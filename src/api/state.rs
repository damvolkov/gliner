use std::sync::{Arc, OnceLock};

use tokio_util::sync::CancellationToken;

use crate::api::error::ApiError;
use crate::core::query::Query;
use crate::core::settings::Settings;
use crate::runtime::executor::BoundedExecutor;
use crate::schema::error::RequestError;

/// Shared by every request: the executor once the model is loaded (empty until then, so `/ready` and the
/// prediction routes answer 503 meanwhile), the default query, `/ner/batch` limits, and the shutdown signal
/// every stream obeys.
#[derive(Debug, Clone)]
pub struct AppState {
    pub executor: Arc<OnceLock<BoundedExecutor>>,
    pub defaults: Query,
    pub concurrency: usize,
    pub line: usize,
    pub shutdown: CancellationToken,
}

impl AppState {
    /// # Errors
    /// The configured default labels or threshold are not a valid query.
    pub fn new(settings: &Settings) -> Result<Self, RequestError> {
        Ok(Self {
            executor: Arc::new(OnceLock::new()),
            defaults: Query::new(
                settings.model.labels.iter().cloned().collect(),
                settings.model.threshold,
            )?,
            concurrency: settings.model.concurrency,
            line: settings.server.line,
            shutdown: CancellationToken::new(),
        })
    }

    /// # Errors
    /// The model is not loaded yet.
    pub fn ready(&self) -> Result<&BoundedExecutor, ApiError> {
        self.executor.get().ok_or(ApiError::NotReady)
    }
}
