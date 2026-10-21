use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::schema::card::ModelCard;
use crate::schema::promote::PromoteRequest;

/// `POST /promote`: loads a registered version and serves it; requests in flight finish on the one they began
/// with. A version that fails to load leaves the champion serving.
///
/// # Errors
/// Train role off (501), unknown version (404), a version that does not load (500), loading (503).
pub async fn promote(
    State(state): State<AppState>,
    payload: Result<Json<PromoteRequest>, JsonRejection>,
) -> Result<Json<ModelCard>, ApiError> {
    let lab = state.lab()?;
    let Json(request) = payload?;
    Ok(Json(lab.promote(request.version).await?))
}
