use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::schema::eval::{EvalRequest, ReplyEval};

/// `POST /eval`: an eval set scored on a registered version (default: the served one), strict, lenient and per
/// label.
///
/// # Errors
/// Train role off (501), unknown set or version (404), a set without labels (422), loading (503).
pub async fn eval(
    State(state): State<AppState>,
    payload: Result<Json<EvalRequest>, JsonRejection>,
) -> Result<Json<ReplyEval>, ApiError> {
    let lab = state.lab()?;
    let Json(request) = payload?;
    Ok(Json(lab.eval(&request.set, request.version).await?))
}
