use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::http::StatusCode;
use serde_json::json;

use crate::app::{call, lab};

#[tokio::test]
async fn test_promote_serves_the_version_and_moves_current() {
    let lab = lab();
    let (status, card) = call(&lab.state, "POST", "/promote", r#"{"version": 2}"#).await;
    assert_eq!(status, StatusCode::OK, "{card}");
    assert_eq!(
        (card["model"].as_str(), card["served"].as_u64()),
        (Some("tiny"), Some(2))
    );
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 2);
    let current = std::fs::read_to_string(lab.root.path().join("models/current")).unwrap();
    assert_eq!(current, "2\n");
}

#[tokio::test]
async fn test_promote_failure_keeps_the_champion() {
    let lab = lab();
    let (status, reply) = call(&lab.state, "POST", "/promote", r#"{"version": 7}"#).await;
    assert_eq!(
        (status, reply),
        (StatusCode::NOT_FOUND, json!({"error": "version 7 is not registered"}))
    );
    let broken = lab.root.path().join("models/2/text.onnx");
    std::fs::write(&broken, b"not a model").unwrap();
    let (status, reply) = call(&lab.state, "POST", "/promote", r#"{"version": 2}"#).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        reply["error"]
            .as_str()
            .unwrap()
            .starts_with("version 2 cannot be served"),
        "{reply}"
    );
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 1);
    let current = std::fs::read_to_string(lab.root.path().join("models/current")).unwrap();
    assert_eq!(current, "1\n");
    let (status, _) = call(&lab.state, "POST", "/ner", r#"{"text": "Ana", "labels": ["person"]}"#).await;
    assert_eq!(status, StatusCode::OK);
}

/// Requests keep flowing while the served model is swapped back and forth: none fails, none waits on a half
/// loaded model.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_promote_under_load_drops_no_request() {
    let lab = lab();
    let (served, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let stop = tokio_util::sync::CancellationToken::new();
    let clients: Vec<_> = (0..4)
        .map(|_| {
            let (state, served, failed, stop) = (
                lab.state.clone(),
                Arc::clone(&served),
                Arc::clone(&failed),
                stop.clone(),
            );
            tokio::spawn(async move {
                while !stop.is_cancelled() {
                    let (status, body) = call(
                        &state,
                        "POST",
                        "/ner",
                        r#"{"text": "Ana vive en Málaga", "labels": ["person"]}"#,
                    )
                    .await;
                    let counter = if status == StatusCode::OK && body["entities"].as_array().map(Vec::len) == Some(4) {
                        &served
                    } else {
                        &failed
                    };
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();
    for version in [2, 1, 2, 1, 2] {
        let (status, card) = call(&lab.state, "POST", "/promote", &format!(r#"{{"version": {version}}}"#)).await;
        assert_eq!((status, card["served"].as_u64()), (StatusCode::OK, Some(version)));
    }
    stop.cancel();
    for client in clients {
        client.await.unwrap();
    }
    assert_eq!(failed.load(Ordering::SeqCst), 0);
    assert!(
        served.load(Ordering::SeqCst) >= 5,
        "served {}",
        served.load(Ordering::SeqCst)
    );
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 2);
}
