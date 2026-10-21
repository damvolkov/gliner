use serde::{Deserialize, Serialize};

use crate::schema::entity::Entity;

/// `POST /ner`: one text, the labels to look for, and an optional threshold (default from the config).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NerRequest {
    pub text: String,
    pub labels: Vec<String>,
    pub threshold: Option<f32>,
}

/// Entities ordered by start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NerResponse {
    pub entities: Vec<Entity>,
}
