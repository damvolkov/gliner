use serde::{Deserialize, Serialize};

/// `GET /models`: the served checkpoint, the version of its files, how many label embeddings are cached, and the
/// registry version served (a recognizer alone does not know it: `None`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    pub model: String,
    pub version: String,
    pub labels_cached: usize,
    pub served: Option<u32>,
}
