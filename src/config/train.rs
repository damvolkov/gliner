use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The regression gate, in F1 points (0–100) of candidate minus champion on the pooled eval sets: the strict
/// delta's point estimate, the lower bound of its 95 % bootstrap interval, and the lenient delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GateConfig {
    pub strict: f64,
    pub floor: f64,
    pub lenient: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            strict: 0.0,
            floor: -1.0,
            lenient: -1.0,
        }
    }
}

/// The train role: off by default (its routes answer 501). `data` holds `datasets/` and `evals/`; `trainer`,
/// `exporter` and `reference` are the command lines (program first) of the Python tools, their arguments
/// appended; `parity` is the share of identical predictions the exported candidate must keep against `PyTorch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    pub enabled: bool,
    pub data: PathBuf,
    pub trainer: Vec<String>,
    pub exporter: Vec<String>,
    pub reference: Vec<String>,
    pub parity: f64,
    pub gate: GateConfig,
}

impl TrainConfig {
    fn tool(project: &str, script: &str) -> Vec<String> {
        ["uv", "run", "-q", "--project", project, "python", script]
            .map(str::to_owned)
            .to_vec()
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            data: PathBuf::from("data/train"),
            trainer: Self::tool("tools/train", "tools/train/train.py"),
            exporter: Self::tool("tools/export", "tools/export/export.py"),
            reference: Self::tool("tools/export", "tools/export/parity.py"),
            parity: 0.99,
            gate: GateConfig::default(),
        }
    }
}
