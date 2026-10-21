use axum::Json;
use axum::extract::State;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::schema::card::ModelCard;

/// The served model id, the version of its files, the label embeddings cached, and the registry version served.
///
/// # Errors
/// The model is not loaded yet.
pub async fn models(State(state): State<AppState>) -> Result<Json<ModelCard>, ApiError> {
    Ok(Json(state.ready()?.card()))
}
