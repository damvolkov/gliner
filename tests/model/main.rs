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
use gliner::runtime::registry::ModelRegistry;
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

/// The registry the settings point at (`GLINER_MODEL__DIR`; `make export` writes version 1).
pub fn registry() -> ModelRegistry {
    let settings = Settings::load(None).unwrap();
    ModelRegistry::open(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&settings.model.dir)).unwrap()
}

/// The registry's served version, loaded with the settings (`GLINER_MODEL__THREADS`, …).
pub fn runtime() -> OnnxRuntime {
    let (settings, registry) = (Settings::load(None).unwrap(), registry());
    let dir = registry.path(registry.current().unwrap());
    OnnxRuntime::load(&ModelArtifact::open(&dir).unwrap(), &settings.model).unwrap()
}
