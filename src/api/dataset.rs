use std::io;

use axum::Json;
use axum::body::Body;
use futures_util::TryStreamExt;
use tokio_util::io::StreamReader;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::runtime::lab::LabError;
use crate::runtime::store::KindStore;
use crate::schema::dataset::CardDataset;

/// `PUT /datasets/{name}`, `PUT /evals/{name}`: an NDJSON body streamed, every line checked, stored whole or not
/// at all.
///
/// # Errors
/// Train role off (501), an invalid name or line, no examples (422), an unreadable body (400).
pub async fn put(state: AppState, kind: KindStore, name: String, body: Body) -> Result<Json<CardDataset>, ApiError> {
    let reader = StreamReader::new(body.into_data_stream().map_err(io::Error::other));
    let card = state
        .lab()?
        .store
        .put(kind, &name, reader)
        .await
        .map_err(LabError::from)?;
    Ok(Json(card))
}

/// `GET /datasets`, `GET /evals`: name, lines and sha256 of every stored set.
///
/// # Errors
/// Train role off (501), unreadable store (500).
pub async fn list(state: AppState, kind: KindStore) -> Result<Json<Vec<CardDataset>>, ApiError> {
    let cards = state.lab()?.store.list(kind).await.map_err(LabError::from)?;
    Ok(Json(cards))
}
