use axum::http::StatusCode;
use serde_json::json;

use crate::app::{call, loaded, state};

#[tokio::test]
async fn test_ner_returns_character_offsets() {
    let (state, _) = loaded(1);
    let (status, body) = call(
        &state,
        "POST",
        "/ner",
        r#"{"text": "Visité Málaga y Óbidos", "labels": ["location"]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        json!({"entities": [
            {"start": 0, "end": 6, "text": "Visité", "label": "location", "score": 0.9},
            {"start": 7, "end": 13, "text": "Málaga", "label": "location", "score": 0.9},
            {"start": 16, "end": 22, "text": "Óbidos", "label": "location", "score": 0.9}
        ]})
    );
}

#[tokio::test]
async fn test_ner_threshold_overrides_the_default() {
    let (state, _) = loaded(1);
    let (_, body) = call(
        &state,
        "POST",
        "/ner",
        r#"{"text": "Ana", "labels": ["person"], "threshold": 0.95}"#,
    )
    .await;
    assert_eq!(body, json!({"entities": []}));
}

#[tokio::test]
async fn test_ner_rejects_invalid_requests() {
    let (state, _) = loaded(1);
    let cases = [
        (
            r#"{"text": "a", "labels": []}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "labels must not be empty",
        ),
        (
            r#"{"text": "a", "labels": ["x", " "]}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "label 1 is blank",
        ),
        (
            r#"{"text": "a", "labels": ["x"], "threshold": 1.5}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "threshold 1.5 must be in (0, 1)",
        ),
    ];
    for (body, status, error) in cases {
        assert_eq!(
            call(&state, "POST", "/ner", body).await,
            (status, json!({"error": error}))
        );
    }
    let (status, body) = call(
        &state,
        "POST",
        "/ner",
        r#"{"text": "a", "labels": ["x"], "lang": "es"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["error"].as_str().unwrap().contains("lang"), "{body}");
    let (status, _) = call(&state, "POST", "/ner", "{not json").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_ner_reports_inference_failures_and_loading() {
    let (ready, _) = loaded(1);
    assert_eq!(
        call(&ready, "POST", "/ner", r#"{"text": "boom", "labels": ["x"]}"#).await,
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error": "onnx runtime failed: boom"})
        )
    );
    assert_eq!(
        call(&state(1), "POST", "/ner", r#"{"text": "a", "labels": ["x"]}"#).await,
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error": "the model is still loading"})
        )
    );
}
