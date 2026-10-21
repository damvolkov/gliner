use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::runtime::job::{JobError, TrainJob};
use crate::schema::train::{JobTrain, TrainRequest};

/// `POST /train`: a job validating, training, exporting, gating and registering a candidate on its own; answers
/// 202 with the job as it starts.
///
/// # Errors
/// Train role off (501), unknown dataset or base (404), invalid params (422), a job already running (409),
/// loading (503).
pub async fn start(
    State(state): State<AppState>,
    payload: Result<Json<TrainRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<JobTrain>), ApiError> {
    let lab = state.lab()?;
    let Json(request) = payload?;
    Ok((StatusCode::ACCEPTED, Json(TrainJob::start(lab, request).await?)))
}

/// `GET /train`: every job of this process, oldest first.
///
/// # Errors
/// Train role off (501).
pub async fn list(State(state): State<AppState>) -> Result<Json<Vec<JobTrain>>, ApiError> {
    Ok(Json(state.lab()?.board.list()))
}

/// `GET /train/{id}`: state, step, log tail and, once validated, the report and the version registered.
///
/// # Errors
/// Train role off (501), unknown job (404).
pub async fn job(State(state): State<AppState>, Path(id): Path<u64>) -> Result<Json<JobTrain>, ApiError> {
    Ok(Json(state.lab()?.board.get(id).map_err(JobError::from)?))
}

/// `DELETE /train/{id}`: cancels the running job, the tool it runs killed with everything it started, and
/// answers once it has stopped.
///
/// # Errors
/// Train role off (501), unknown job (404), a job not running (409).
pub async fn cancel(State(state): State<AppState>, Path(id): Path<u64>) -> Result<Json<JobTrain>, ApiError> {
    Ok(Json(state.lab()?.board.cancel(id).await.map_err(JobError::from)?))
}
