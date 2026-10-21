use std::path::PathBuf;
use std::thread::available_parallelism;

use serde::{Deserialize, Serialize};

/// The served model and how it runs: its artifact dir, ONNX Runtime intra-op threads (default: the CPUs this
/// process may use, cgroup-aware), predictions in flight, the default threshold and labels (`/ner/batch` lines
/// without their own), the label embedding cache capacity, and the word window over long texts (default: the
/// model's `max_len`) with the words consecutive windows share.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub dir: PathBuf,
    pub threads: Option<usize>,
    pub concurrency: usize,
    pub threshold: f32,
    pub labels: Vec<String>,
    pub cache: usize,
    pub window: Option<usize>,
    pub overlap: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("data/models/gliner-bi-base-v2.0"),
            threads: None,
            concurrency: 2,
            threshold: 0.5,
            labels: ["person", "organization", "location"].map(str::to_owned).to_vec(),
            cache: 4096,
            window: None,
            overlap: 32,
        }
    }
}

impl ModelConfig {
    /// Intra-op threads ONNX Runtime gets, always explicit: its own default sizes to the host, not the cgroup.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
            .or_else(|| available_parallelism().ok().map(std::num::NonZero::get))
            .unwrap_or(1)
    }
}
