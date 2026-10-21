use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use gliner::api::server::HttpServer;
use gliner::api::state::AppState;
use gliner::core::settings::Settings;
use gliner::runtime::executor::{BoundedExecutor, ServedExecutor};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

use crate::fake::FakeRecognizer;

/// Default settings with `concurrency` predictions in flight and 1 KiB lines; the model not loaded yet.
pub fn state(concurrency: usize) -> AppState {
    let mut settings = Settings::default();
    settings.model.concurrency = concurrency;
    settings.server.line = 1024;
    AppState::new(&settings).unwrap()
}

/// The state with the fake loaded.
pub fn loaded(concurrency: usize) -> (AppState, Arc<FakeRecognizer>) {
    let state = state(concurrency);
    let fake = Arc::new(FakeRecognizer::default());
    state.slot.set(ServedExecutor {
        version: 1,
        executor: BoundedExecutor::new(fake.clone(), concurrency),
    });
    (state, fake)
}

pub fn router(state: &AppState) -> Router {
    HttpServer::router(state.clone())
}

/// One request through the router (no network); the status and the JSON body.
pub async fn call(state: &AppState, method: &str, uri: &str, body: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = router(state).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}
