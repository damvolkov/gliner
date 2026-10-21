use serde::{Deserialize, Serialize};

use crate::schema::entity::Entity;

/// One NDJSON line of `POST /ner/batch`. With `id` and `text` it is an item; with neither, and only as the first
/// line, a header whose `labels`/`threshold` become the defaults of every item after it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineBatch {
    pub id: Option<String>,
    pub text: Option<String>,
    pub labels: Option<Vec<String>>,
    pub threshold: Option<f32>,
}

/// One NDJSON line of the response, written as each item completes (not in request order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReplyBatch {
    Entities { id: String, entities: Vec<Entity> },
    Failed { id: String, error: String },
    Invalid { line: usize, error: String },
}
