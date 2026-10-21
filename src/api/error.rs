use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

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
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::NotReady => StatusCode::SERVICE_UNAVAILABLE,
            Self::Body(rejection) => rejection.status(),
            Self::Request(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Recognizer(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}
