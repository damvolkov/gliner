use serde::{Deserialize, Serialize};

/// One recognized entity: `start..end` is a half-open range of characters (Unicode scalar values) of the text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub label: String,
    pub score: f32,
}
