use axum::http::StatusCode;
use serde_json::json;

use crate::app::{call, loaded, state};

#[tokio::test]
async fn test_health_answers_while_loading() {
    let (status, body) = call(&state(1), "GET", "/health", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}));
}

#[tokio::test]
async fn test_ready_and_models_follow_the_load() {
    let loading = state(1);
    assert_eq!(
        call(&loading, "GET", "/ready", "").await,
        (StatusCode::SERVICE_UNAVAILABLE, json!({"status": "loading"}))
    );
    assert_eq!(
        call(&loading, "GET", "/models", "").await,
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error": "the model is still loading"})
        )
    );
    let (ready, _) = loaded(1);
    assert_eq!(
        call(&ready, "GET", "/ready", "").await,
        (StatusCode::OK, json!({"status": "ready"}))
    );
    assert_eq!(
        call(&ready, "GET", "/models", "").await,
        (
            StatusCode::OK,
            json!({"model": "fake", "version": "0123456789abcdef", "labels_cached": 0})
        )
    );
}
