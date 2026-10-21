use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::api::error::ApiError;
use crate::core::query::Query;
use crate::core::settings::Settings;
use crate::runtime::executor::{ServedExecutor, SlotExecutor};
use crate::schema::error::RequestError;

/// Shared by every request: the served model once loaded (empty until then, so `/ready` and the prediction routes
/// answer 503 meanwhile; swapped whole on promote), the default query, `/ner/batch` limits, and the shutdown
/// signal every stream obeys.
#[derive(Debug, Clone)]
pub struct AppState {
    pub slot: Arc<SlotExecutor>,
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
            slot: Arc::new(SlotExecutor::default()),
            defaults: Query::new(
                settings.model.labels.iter().cloned().collect(),
                settings.model.threshold,
            )?,
            concurrency: settings.model.concurrency,
            line: settings.server.line,
            shutdown: CancellationToken::new(),
        })
    }

    /// The served model, held for the caller's whole request.
    ///
    /// # Errors
    /// The model is not loaded yet.
    pub fn ready(&self) -> Result<ServedExecutor, ApiError> {
        self.slot.get().ok_or(ApiError::NotReady)
    }
}
