use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::schema::dataset::CardDataset;
use crate::schema::eval::CompareEval;

/// Fine-tuning hyperparameters handed to the trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParamsTrain {
    pub epochs: u32,
    pub lr: f64,
    pub batch: u32,
}

impl Default for ParamsTrain {
    fn default() -> Self {
        Self {
            epochs: 3,
            lr: 5e-6,
            batch: 8,
        }
    }
}

/// `POST /train`: a stored dataset, the version to start from (default: the served one), hyperparameters, and
/// whether a candidate that passes the gate is promoted at once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainRequest {
    pub dataset: String,
    pub base: Option<u32>,
    #[serde(default)]
    pub params: ParamsTrain,
    #[serde(default)]
    pub auto_promote: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateTrain {
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// The pipeline, in order; a job reports the step it is in (or ended in).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepTrain {
    Leakage,
    Train,
    Export,
    Parity,
    Regression,
    Gate,
    Register,
    Promote,
}

/// One clause of the gate: the measured value (F1 points) against its configured minimum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckTrain {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateTrain {
    pub passed: bool,
    pub checks: Vec<CheckTrain>,
}

/// What validated a candidate: the champion it was measured against, export parity with `PyTorch`, every eval
/// set compared, the pooled comparison, and the gate's verdict on the pooled one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportTrain {
    pub champion: u32,
    pub parity: f64,
    pub sets: BTreeMap<String, CompareEval>,
    pub pooled: CompareEval,
    pub gate: GateTrain,
}

/// `GET /train/{id}`: a job, its step, the tail of its log, and once validated its report and version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobTrain {
    pub id: u64,
    pub state: StateTrain,
    pub step: StepTrain,
    pub dataset: CardDataset,
    pub base: u32,
    pub params: ParamsTrain,
    pub auto_promote: bool,
    pub version: Option<u32>,
    pub promoted: bool,
    pub error: Option<String>,
    pub report: Option<ReportTrain>,
    pub log: Vec<String>,
    pub created: u64,
    pub finished: Option<u64>,
}
