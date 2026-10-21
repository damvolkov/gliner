use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

use crate::app::{loaded, router, state};

fn request(body: Body) -> Request<Body> {
    Request::builder().method("POST").uri("/ner/batch").body(body).unwrap()
}

/// Every reply line, sorted by `id` then `line` (replies arrive as items complete, not in order).
async fn replies(state: &gliner::api::state::AppState, lines: &[&str]) -> Vec<Value> {
    let body = lines.join("\n") + "\n";
    let response = router(state).oneshot(request(Body::from(body))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let mut values: Vec<Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    values.sort_by_key(|value| {
        (
            value["id"].as_str().unwrap_or_default().to_owned(),
            value["line"].as_u64(),
        )
    });
    values
}

/// A request body fed line by line through a channel, held open until the sender drops.
fn channel() -> (mpsc::UnboundedSender<String>, Body) {
    let (sender, receiver) = mpsc::unbounded_channel::<String>();
    let lines = stream::unfold(receiver, |mut receiver| async move {
        receiver
            .recv()
            .await
            .map(|line| (Ok::<_, Infallible>(Bytes::from(format!("{line}\n"))), receiver))
    });
    (sender, Body::from_stream(lines))
}

async fn next_line(body: &mut Body) -> Option<Value> {
    let frame = tokio::time::timeout(Duration::from_secs(5), body.frame())
        .await
        .unwrap()?;
    let data = frame.unwrap().into_data().unwrap();
    Some(serde_json::from_slice(&data).unwrap())
}

#[tokio::test]
async fn test_batch_items_header_and_defaults() {
    let (state, _) = loaded(2);
    let got = replies(
        &state,
        &[
            r#"{"labels": ["tool"], "threshold": 0.5}"#,
            r#"{"id": "a", "text": "Docker en Linux"}"#,
            "",
            r#"{"id": "b", "text": "Ana", "labels": ["person"]}"#,
            r#"{"id": "c", "text": "Ana", "threshold": 0.95}"#,
        ],
    )
    .await;
    assert_eq!(
        got,
        [
            json!({"id": "a", "entities": [
                {"start": 0, "end": 6, "text": "Docker", "label": "tool", "score": 0.9},
                {"start": 10, "end": 15, "text": "Linux", "label": "tool", "score": 0.9}
            ]}),
            json!({"id": "b", "entities": [{"start": 0, "end": 3, "text": "Ana", "label": "person", "score": 0.9}]}),
            json!({"id": "c", "entities": []}),
        ]
    );
}

#[tokio::test]
async fn test_batch_without_header_uses_the_configured_labels() {
    let (state, _) = loaded(1);
    let got = replies(&state, &[r#"{"id": "a", "text": "Ana"}"#]).await;
    assert_eq!(got[0]["entities"][0]["label"], "person");
}

#[tokio::test]
async fn test_batch_reports_bad_lines_and_keeps_going() {
    let (state, _) = loaded(2);
    let long = format!(r#"{{"id": "x", "text": "{}"}}"#, "a".repeat(2000));
    let got = replies(
        &state,
        &[
            r#"{"id": "a", "text": "Ana"}"#,
            "{broken",
            r#"{"labels": ["late"]}"#,
            r#"{"text": "no id"}"#,
            r#"{"id": "d"}"#,
            r#"{"id": "e", "text": "Ana", "labels": []}"#,
            r#"{"id": "f", "text": "boom"}"#,
            &long,
            r#"{"id": "g", "text": "Eva", "colour": "red"}"#,
            r#"{"id": "h", "text": "Eva"}"#,
        ],
    )
    .await;
    let summary: Vec<(Value, Value, bool)> = got
        .iter()
        .map(|reply| {
            (
                reply["id"].clone(),
                reply["line"].clone(),
                reply.get("entities").is_some(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            (Value::Null, json!(2), false),
            (Value::Null, json!(3), false),
            (Value::Null, json!(4), false),
            (Value::Null, json!(8), false),
            (Value::Null, json!(9), false),
            (json!("a"), Value::Null, true),
            (json!("d"), Value::Null, false),
            (json!("e"), Value::Null, false),
            (json!("f"), Value::Null, false),
            (json!("h"), Value::Null, true),
        ]
    );
    let error = |index: usize| got[index]["error"].as_str().unwrap().to_owned();
    assert!(error(1).contains("header"), "{}", error(1));
    assert_eq!(error(2), "missing id");
    assert_eq!(error(3), "line exceeds 1024 bytes");
    assert!(error(4).contains("colour"), "{}", error(4));
    assert_eq!(error(6), "missing text");
    assert_eq!(error(7), "labels must not be empty");
    assert_eq!(error(8), "onnx runtime failed: boom");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_batch_bounds_items_in_flight() {
    let (state, fake) = loaded(3);
    let lines: Vec<String> = (0..40)
        .map(|n| format!(r#"{{"id": "{n}", "text": "sleep:5 Ana"}}"#))
        .collect();
    let got = replies(&state, &lines.iter().map(String::as_str).collect::<Vec<_>>()).await;
    assert_eq!(got.len(), 40);
    assert_eq!(fake.calls.load(Ordering::SeqCst), 40);
    assert!(
        fake.peak.load(Ordering::SeqCst) <= 3,
        "peak {}",
        fake.peak.load(Ordering::SeqCst)
    );
    assert!(fake.peak.load(Ordering::SeqCst) >= 2, "no concurrency at all");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_batch_streams_each_reply_before_the_body_ends() {
    let (state, _) = loaded(1);
    let (sender, body) = channel();
    let mut response = router(&state).oneshot(request(body)).await.unwrap().into_body();
    sender.send(r#"{"id": "a", "text": "Ana"}"#.to_owned()).unwrap();
    assert_eq!(next_line(&mut response).await.unwrap()["id"], "a");
    sender.send(r#"{"id": "b", "text": "Eva"}"#.to_owned()).unwrap();
    assert_eq!(next_line(&mut response).await.unwrap()["id"], "b");
    drop(sender);
    assert!(next_line(&mut response).await.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_batch_stops_when_the_client_goes_away() {
    let (state, fake) = loaded(1);
    let (sender, body) = channel();
    (0..10).for_each(|n| {
        sender
            .send(format!(r#"{{"id": "{n}", "text": "sleep:50 Ana"}}"#))
            .unwrap();
    });
    let mut response = router(&state).oneshot(request(body)).await.unwrap().into_body();
    assert!(next_line(&mut response).await.is_some());
    drop(response);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        fake.calls.load(Ordering::SeqCst) <= 3,
        "calls {}",
        fake.calls.load(Ordering::SeqCst)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_batch_ends_on_shutdown() {
    let (state, _) = loaded(1);
    let (sender, body) = channel();
    let mut response = router(&state).oneshot(request(body)).await.unwrap().into_body();
    sender.send(r#"{"id": "a", "text": "Ana"}"#.to_owned()).unwrap();
    assert!(next_line(&mut response).await.is_some());
    state.shutdown.cancel();
    assert!(next_line(&mut response).await.is_none());
    drop(sender);
}

#[tokio::test]
async fn test_batch_waits_for_the_model() {
    let response = router(&state(1)).oneshot(request(Body::empty())).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
