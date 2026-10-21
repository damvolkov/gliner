use serde::{Deserialize, Serialize};

/// `GET /models`: the served checkpoint, the version of its files, and how many label embeddings are cached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    pub model: String,
    pub version: String,
    pub labels_cached: usize,
}
