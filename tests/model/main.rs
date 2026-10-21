#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

mod bench;
mod parity;

use std::path::PathBuf;

use gliner::core::settings::Settings;
use gliner::runtime::artifact::ModelArtifact;
use gliner::runtime::onnx::OnnxRuntime;
use gliner::schema::entity::Entity;
use serde::Deserialize;

/// Reference predictions of `PyTorch` GLiNER, written by `make fixture`.
#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub labels: Vec<String>,
    pub threshold: f32,
    pub items: Vec<Item>,
}

#[derive(Debug, Deserialize)]
pub struct Item {
    pub id: String,
    pub text: String,
    pub entities: Vec<Entity>,
}

/// `PARITY_FIXTURE`, else the public fixture.
pub fn fixture() -> Fixture {
    let path = std::env::var_os("PARITY_FIXTURE").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/resources/parity/torch.json"),
        PathBuf::from,
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The exported model the settings point at (`GLINER_MODEL__DIR`, `GLINER_MODEL__THREADS`, …), loaded.
pub fn runtime() -> OnnxRuntime {
    let settings = Settings::load(None).unwrap();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&settings.model.dir);
    OnnxRuntime::load(&ModelArtifact::open(&dir).unwrap(), &settings.model).unwrap()
}
