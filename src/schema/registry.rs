use serde::{Deserialize, Serialize};

use crate::schema::dataset::CardDataset;
use crate::schema::train::{ParamsTrain, ReportTrain};

/// A version's lineage, kept under `registry` in its `manifest.json`: the version it was trained from, the
/// checkpoint id the lineage started at, the dataset and parameters, when it was registered (unix seconds), and
/// the report that validated it. An exported model without the block is a root version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryRegistry {
    pub version: u32,
    pub parent: Option<u32>,
    pub base: String,
    pub dataset: Option<CardDataset>,
    pub params: Option<ParamsTrain>,
    pub created: u64,
    pub report: Option<ReportTrain>,
}
