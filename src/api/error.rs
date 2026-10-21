use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::runtime::board::BoardError;
use crate::runtime::job::JobError;
use crate::runtime::lab::LabError;
use crate::runtime::registry::RegistryError;
use crate::runtime::store::StoreError;
use crate::schema::error::{RecognizerError, RequestError};

/// Every failure a handler returns, as `{"error": "..."}` with its status.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("the model is still loading")]
    NotReady,
    #[error(transparent)]
    Body(#[from] JsonRejection),
    #[error(transparent)]
    Request(#[from] RequestError),
    #[error(transparent)]
    Recognizer(#[from] RecognizerError),
    #[error("training is disabled: train.enabled = true turns it on (the gliner-train image does)")]
    Disabled,
    #[error(transparent)]
    Lab(#[from] LabError),
    #[error(transparent)]
    Job(#[from] JobError),
}

impl ApiError {
    // ##### PRIVATE #####

    const fn into_job_status(error: &JobError) -> StatusCode {
        match error {
            JobError::Lab(error) => Self::into_lab_status(error),
            JobError::Board(BoardError::Unknown(_)) => StatusCode::NOT_FOUND,
            JobError::Board(BoardError::Busy(_) | BoardError::Finished(_)) => StatusCode::CONFLICT,
            JobError::Params(_) => StatusCode::UNPROCESSABLE_ENTITY,
            JobError::Toolchain(_)
            | JobError::Unvalidated
            | JobError::Leakage(_)
            | JobError::Changed(_)
            | JobError::Candidate(_)
            | JobError::Parity { .. }
            | JobError::File { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    const fn into_lab_status(error: &LabError) -> StatusCode {
        match error {
            LabError::Registry(RegistryError::Unknown(_)) | LabError::Store(StoreError::Missing { .. }) => {
                StatusCode::NOT_FOUND
            }
            LabError::Store(
                StoreError::Name(_) | StoreError::Line { .. } | StoreError::Long { .. } | StoreError::Empty,
            )
            | LabError::Unlabelled(_) => StatusCode::UNPROCESSABLE_ENTITY,
            LabError::Store(StoreError::Body(_)) => StatusCode::BAD_REQUEST,
            LabError::NotReady => StatusCode::SERVICE_UNAVAILABLE,
            LabError::Registry(_)
            | LabError::Store(StoreError::Io { .. })
            | LabError::Load { .. }
            | LabError::Recognizer(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::NotReady => StatusCode::SERVICE_UNAVAILABLE,
            Self::Body(rejection) => rejection.status(),
            Self::Request(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Recognizer(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Disabled => StatusCode::NOT_IMPLEMENTED,
            Self::Lab(error) => Self::into_lab_status(error),
            Self::Job(error) => Self::into_job_status(error),
        };
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}
