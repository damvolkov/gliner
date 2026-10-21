use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// `POST /eval`: an eval set scored on a registered version (default: the served one).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalRequest {
    pub set: String,
    pub version: Option<u32>,
}

/// Precision, recall and F1 (0–1, zero when undefined) from true positives, false positives and false negatives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreEval {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
}

/// Texts scored, micro-averaged: strict (span and label), lenient (distinct names per text, label ignored), and
/// strict per label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportEval {
    pub texts: usize,
    pub strict: ScoreEval,
    pub lenient: ScoreEval,
    pub labels: BTreeMap<String, ScoreEval>,
}

/// The `POST /eval` reply: the set, the version scored, and its report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplyEval {
    pub set: String,
    pub version: u32,
    #[serde(flatten)]
    pub report: ReportEval,
}

/// Candidate minus champion F1, in points (0–100): the estimate on the texts as they are and the 95 % paired
/// bootstrap interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaEval {
    pub point: f64,
    pub low: f64,
    pub high: f64,
}

/// Champion and candidate on the same texts, and the strict and lenient deltas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareEval {
    pub champion: ReportEval,
    pub candidate: ReportEval,
    pub strict: DeltaEval,
    pub lenient: DeltaEval,
}
