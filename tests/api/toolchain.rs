use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::future::BoxFuture;
use gliner::core::recognizer::Recognizer;
use gliner::runtime::artifact::ModelArtifact;
use gliner::runtime::onnx::OnnxRuntime;
use gliner::runtime::toolchain::{CallToolchain, Toolchain, ToolchainError};
use serde_json::{Value, json};

use crate::app::{copy_tiny, resource, tiny_settings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeFake {
    /// Every tool succeeds; the reference predicts exactly what the tiny runtime does.
    Pass,
    /// The reference finds nothing: the export "diverges" from `PyTorch`.
    Diverge,
    /// Training never ends until the run is dropped.
    Hang,
    /// Training fails.
    Fail,
}

/// The tools without Python: training writes a checkpoint marker, the export copies the tiny artifact, the
/// reference predicts with the tiny runtime. Records the tools called and whether a run was dropped mid-way.
#[derive(Debug)]
pub struct FakeToolchain {
    pub mode: ModeFake,
    pub calls: Mutex<Vec<&'static str>>,
    pub dropped: AtomicBool,
    runtime: OnnxRuntime,
}

/// Sets its flag when dropped: a run cut short.
struct DropFlag<'a>(&'a AtomicBool);

impl Drop for DropFlag<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl FakeToolchain {
    pub fn new(mode: ModeFake) -> Self {
        let artifact = ModelArtifact::open(&resource("tiny")).unwrap();
        Self {
            mode,
            calls: Mutex::new(Vec::new()),
            dropped: AtomicBool::new(false),
            runtime: OnnxRuntime::load(&artifact, &tiny_settings().model).unwrap(),
        }
    }

    pub fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().clone()
    }

    fn reference(&self, texts: &Path, labels: &[String], threshold: f32, out: &Path) {
        let rows: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(texts).unwrap()).unwrap();
        let items: Vec<Value> = rows
            .iter()
            .map(|row| {
                let text = row["text"].as_str().unwrap();
                let entities = match self.mode {
                    ModeFake::Diverge => Vec::new(),
                    _ => self.runtime.predict(text, labels, threshold).unwrap(),
                };
                json!({"id": row["id"], "text": text, "entities": entities})
            })
            .collect();
        std::fs::write(out, json!({"model": "tiny", "items": items}).to_string()).unwrap();
    }
}

impl Toolchain for FakeToolchain {
    fn run<'a>(
        &'a self,
        call: &'a CallToolchain,
        log: &'a (dyn Fn(String) + Send + Sync),
    ) -> BoxFuture<'a, Result<(), ToolchainError>> {
        Box::pin(async move {
            match call {
                CallToolchain::Train { out, .. } => {
                    self.calls.lock().unwrap().push("train");
                    log("epoch 1".to_owned());
                    match self.mode {
                        ModeFake::Hang => {
                            let _flag = DropFlag(&self.dropped);
                            std::future::pending::<()>().await;
                        }
                        ModeFake::Fail => {
                            return Err(ToolchainError::Exit {
                                tool: "trainer",
                                status: "exit status: 1".to_owned(),
                                last: "CUDA out of memory".to_owned(),
                            });
                        }
                        ModeFake::Pass | ModeFake::Diverge => {
                            std::fs::create_dir_all(out).unwrap();
                            std::fs::write(out.join("pytorch_model.bin"), "weights").unwrap();
                        }
                    }
                }
                CallToolchain::Export { out, .. } => {
                    self.calls.lock().unwrap().push("export");
                    copy_tiny(out);
                }
                CallToolchain::Reference {
                    texts,
                    labels,
                    threshold,
                    out,
                    ..
                } => {
                    self.calls.lock().unwrap().push("reference");
                    self.reference(texts, labels, *threshold, out);
                }
            }
            Ok(())
        })
    }
}
