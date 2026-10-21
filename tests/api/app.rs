use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use gliner::api::server::HttpServer;
use gliner::api::state::AppState;
use gliner::core::settings::Settings;
use gliner::runtime::artifact::ModelArtifact;
use gliner::runtime::executor::{BoundedExecutor, ServedExecutor};
use gliner::runtime::lab::TrainLab;
use gliner::runtime::onnx::OnnxRuntime;
use gliner::runtime::registry::ModelRegistry;
use gliner::runtime::store::DataStore;
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

use crate::fake::FakeRecognizer;
use crate::toolchain::{FakeToolchain, ModeFake};

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

/// The train role over temporary dirs: a registry holding the tiny artifact as versions 1 and 2 (1 served,
/// loaded for real), and a data dir holding the neutral eval set. Dropping it removes both.
#[derive(Debug)]
pub struct Lab {
    pub state: AppState,
    pub root: tempfile::TempDir,
}

pub fn resource(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources").join(path)
}

pub fn copy_tiny(to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    std::fs::read_dir(resource("tiny"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .for_each(|path| {
            std::fs::copy(&path, to.join(path.file_name().unwrap())).unwrap();
        });
}

/// The tiny model's settings: one thread, windows of its 6 words sharing 2.
pub fn tiny_settings() -> Settings {
    let mut settings = Settings::default();
    settings.model.threads = Some(1);
    settings.model.overlap = 2;
    settings.model.concurrency = 2;
    settings.server.line = 4096;
    settings
}

pub fn lab() -> Lab {
    lab_with(Arc::new(FakeToolchain::new(ModeFake::Pass)))
}

pub fn lab_with(toolchain: Arc<FakeToolchain>) -> Lab {
    let root = tempfile::tempdir().unwrap();
    let models = root.path().join("models");
    copy_tiny(&models.join("1"));
    copy_tiny(&models.join("2"));
    std::fs::write(models.join("current"), "1\n").unwrap();
    let mut settings = tiny_settings();
    settings.model.dir.clone_from(&models);
    settings.train.enabled = true;
    settings.train.data = root.path().join("train");
    let registry = ModelRegistry::open(&models).unwrap();
    let store = DataStore::open(&settings.train.data, settings.server.line).unwrap();
    std::fs::copy(
        resource("eval/neutral.jsonl"),
        settings.train.data.join("evals/neutral.jsonl"),
    )
    .unwrap();
    let mut state = AppState::new(&settings).unwrap();
    let runtime = OnnxRuntime::load(&ModelArtifact::open(&models.join("1")).unwrap(), &settings.model).unwrap();
    state.slot.set(ServedExecutor {
        version: 1,
        executor: BoundedExecutor::new(Arc::new(runtime), settings.model.concurrency),
    });
    state.lab = Some(Arc::new(TrainLab::new(
        registry,
        store,
        Arc::clone(&state.slot),
        toolchain,
        &settings,
    )));
    Lab { state, root }
}

/// A raw-body request (NDJSON uploads) through the router.
pub async fn send(state: &AppState, method: &str, uri: &str, body: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/x-ndjson")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = router(state).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}
