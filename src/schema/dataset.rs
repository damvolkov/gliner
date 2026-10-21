use serde::{Deserialize, Serialize};

/// One gold entity: `start..end` is a half-open range of characters of the text, `label` free text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanDataset {
    pub start: usize,
    pub end: usize,
    pub label: String,
}

/// One NDJSON line of a dataset or an eval set: a text and every entity it holds (`[]` for none).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineDataset {
    pub text: String,
    pub entities: Vec<SpanDataset>,
}

/// A stored dataset or eval set: its name, examples, and the sha256 of its file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardDataset {
    pub name: String,
    pub lines: usize,
    pub sha256: String,
}
