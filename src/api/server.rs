use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::api::batch::batch;
use crate::api::health::{health, ready};
use crate::api::models::models;
use crate::api::ner::ner;
use crate::api::state::AppState;
use crate::config::model::ModelConfig;
use crate::core::settings::Settings;
use crate::runtime::artifact::{ArtifactError, ModelArtifact};
use crate::runtime::executor::BoundedExecutor;
use crate::runtime::onnx::{OnnxError, OnnxRuntime};
use crate::schema::error::RequestError;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error(transparent)]
    Onnx(#[from] OnnxError),
    #[error("invalid defaults: {0}")]
    Defaults(#[from] RequestError),
    #[error("model loader stopped: {0}")]
    Loader(String),
    #[error("cannot bind or serve: {0}")]
    Io(#[from] std::io::Error),
}

/// The HTTP server: listening as soon as the artifact checks out, ready once the model is loaded; SIGINT/SIGTERM
/// (or a failed load) stop new work and end open batch streams after their items in flight.
#[derive(Debug)]
pub struct HttpServer;

impl HttpServer {
    // ##### PRIVATE #####

    async fn serve_signal(shutdown: CancellationToken) {
        let interrupt = tokio::signal::ctrl_c();
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
        let term = async move {
            match terminate.as_mut() {
                Some(signal) => signal.recv().await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = interrupt => {}
            () = async { term.await; } => {}
            () = shutdown.cancelled() => {}
        }
        tracing::info!("server.stopping");
        shutdown.cancel();
    }

    async fn serve_load(artifact: ModelArtifact, config: ModelConfig, state: AppState) -> Result<(), ServerError> {
        let concurrency = config.concurrency;
        let loaded = tokio::task::spawn_blocking(move || OnnxRuntime::load(&artifact, &config))
            .await
            .map_err(|error| ServerError::Loader(error.to_string()))
            .and_then(|loaded| loaded.map_err(ServerError::from));
        match loaded {
            Ok(runtime) => {
                state
                    .executor
                    .get_or_init(|| BoundedExecutor::new(Arc::new(runtime), concurrency));
                tracing::info!("server.ready");
                Ok(())
            }
            Err(error) => {
                tracing::error!(%error, "server.load");
                state.shutdown.cancel();
                Err(error)
            }
        }
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// Every route over a state; `/ner/batch` streams, so its body is unbounded (lines are not).
    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/models", get(models))
            .route("/ner", post(ner))
            .route("/ner/batch", post(batch).layer(DefaultBodyLimit::disable()))
            .with_state(state)
    }

    /// # Errors
    /// Invalid artifact or defaults, an address that cannot be bound, or a model that fails to load.
    pub async fn serve(settings: &Settings) -> Result<(), ServerError> {
        let artifact = ModelArtifact::open(&settings.model.dir)?;
        let state = AppState::new(settings)?;
        let address = SocketAddr::new(settings.server.host, settings.server.port);
        let listener = TcpListener::bind(address).await?;
        tracing::info!(%address, model = %artifact.manifest.model, "server.listening");
        let app = axum::serve(listener, Self::router(state.clone()))
            .with_graceful_shutdown(Self::serve_signal(state.shutdown.clone()))
            .into_future();
        let load = Self::serve_load(artifact, settings.model.clone(), state);
        let (served, loaded) = tokio::join!(app, load);
        served?;
        loaded
    }
}
