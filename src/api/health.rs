use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde_json::{Value, json};

use crate::api::state::AppState;

/// Liveness and the service version; answers while the model loads.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

/// Readiness: 200 once the model is loaded and predictions are served, 503 before.
pub async fn ready(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    match state.slot.get() {
        Some(_) => (StatusCode::OK, Json(json!({ "status": "ready" }))),
        None => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "status": "loading" }))),
    }
}
