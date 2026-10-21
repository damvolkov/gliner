use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::core::query::Query;
use crate::schema::ner::{NerRequest, NerResponse};

/// Entities of one text for the labels asked, character offsets, ordered by start.
///
/// # Errors
/// Malformed body (4xx), invalid labels or threshold (422), model loading (503), inference failure (500).
pub async fn ner(
    State(state): State<AppState>,
    payload: Result<Json<NerRequest>, JsonRejection>,
) -> Result<Json<NerResponse>, ApiError> {
    let Json(request) = payload?;
    let query = Query::new(
        request.labels.into(),
        request.threshold.unwrap_or(state.defaults.threshold),
    )?;
    let entities = state.ready()?.predict(request.text, query).await?;
    Ok(Json(NerResponse { entities }))
}
