use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError};

use ndarray::{Array2, Axis, Ix4};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, TensorRef};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::config::model::ModelConfig;
use crate::core::cache::LabelCache;
use crate::core::recognizer::Recognizer;
use crate::core::span::Span;
use crate::core::splitter::WhitespaceSplitter;
use crate::core::tensor::TextTensor;
use crate::core::window::Window;
use crate::runtime::artifact::{ArtifactError, ModelArtifact};
use crate::schema::card::ModelCard;
use crate::schema::entity::Entity;
use crate::schema::error::RecognizerError;

#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("cannot load {file}: {reason}")]
    Load { file: &'static str, reason: String },
    #[error("window of {window} words must be larger than the overlap of {overlap}")]
    Window { window: usize, overlap: usize },
    #[error("label cache capacity must be at least 1")]
    Cache,
}

/// A GLiNER bi-encoder on ONNX Runtime: the label tower encodes labels once (cached), the text tower scores every
/// span of a window against them. Sessions are shared, one run at a time each; intra-op threads always explicit.
pub struct OnnxRuntime {
    _environment: Environment,
    text: Mutex<Session>,
    labels: Mutex<Session>,
    tokenizer: Tokenizer,
    labels_tokenizer: Tokenizer,
    splitter: WhitespaceSplitter,
    cache: LabelCache,
    width: usize,
    window: usize,
    overlap: usize,
    model: String,
    version: String,
}

impl OnnxRuntime {
    // ##### PRIVATE #####

    fn load_session(
        environment: &Environment,
        artifact: &ModelArtifact,
        file: &'static str,
        threads: usize,
    ) -> Result<Session, OnnxError> {
        let reason = |error: ort::Error<_>| OnnxError::Load {
            file,
            reason: error.to_string(),
        };
        let unit = |error: ort::Error| OnnxError::Load {
            file,
            reason: error.to_string(),
        };
        let builder = Session::builder(environment)
            .map_err(unit)?
            .with_intra_threads(threads)
            .map_err(reason)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(reason)?;
        #[cfg(feature = "cuda")]
        let builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .map_err(reason)?;
        let mut builder = builder;
        builder.commit_from_file(artifact.path(file)).map_err(unit)
    }

    fn load_tokenizer(artifact: &ModelArtifact, file: &'static str) -> Result<Tokenizer, OnnxError> {
        Tokenizer::from_file(artifact.path(file)).map_err(|error| OnnxError::Load {
            file,
            reason: error.to_string(),
        })
    }

    fn predict_lock(session: &Mutex<Session>) -> MutexGuard<'_, Session> {
        session.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn predict_onnx(error: impl std::fmt::Display) -> RecognizerError {
        RecognizerError::Onnx(error.to_string())
    }

    fn predict_shape(error: impl std::fmt::Display) -> RecognizerError {
        RecognizerError::Shape(error.to_string())
    }

    fn predict_output<'s>(
        outputs: &'s ort::session::SessionOutputs<'_>,
        name: &str,
    ) -> Result<ndarray::ArrayViewD<'s, f32>, RecognizerError> {
        outputs
            .get(name)
            .ok_or_else(|| RecognizerError::Shape(format!("the model has no output {name}")))?
            .try_extract_array::<f32>()
            .map_err(Self::predict_onnx)
    }

    fn predict_encode(&self, missing: &[&str]) -> Result<Vec<Vec<f32>>, RecognizerError> {
        let encodings = self
            .labels_tokenizer
            .encode_batch(missing.to_vec(), true)
            .map_err(|error| RecognizerError::Tokenizer(error.to_string()))?;
        let columns = encodings.first().map_or(0, tokenizers::Encoding::len);
        let rows = |pick: fn(&tokenizers::Encoding) -> &[u32]| {
            let flat = encodings
                .iter()
                .flat_map(|encoding| pick(encoding).iter().map(|&id| i64::from(id)));
            Array2::from_shape_vec((encodings.len(), columns), flat.collect()).map_err(Self::predict_shape)
        };
        let (ids, mask) = (
            rows(tokenizers::Encoding::get_ids)?,
            rows(tokenizers::Encoding::get_attention_mask)?,
        );
        let mut session = Self::predict_lock(&self.labels);
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(ids).map_err(Self::predict_onnx)?,
                "attention_mask" => Tensor::from_array(mask).map_err(Self::predict_onnx)?,
            ])
            .map_err(Self::predict_onnx)?;
        let embeddings = Self::predict_output(&outputs, "embeddings")?;
        Ok(embeddings
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect())
    }

    fn predict_labels(&self, labels: &[String]) -> Result<Array2<f32>, RecognizerError> {
        let rows = self.cache.resolve(labels, |missing| self.predict_encode(missing))?;
        let dims = rows.first().map_or(0, |row| row.len());
        let flat: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Array2::from_shape_vec((rows.len(), dims), flat).map_err(Self::predict_shape)
    }

    fn predict_window(
        &self,
        words: &[&str],
        window: Window,
        embeddings: &Array2<f32>,
        threshold: f32,
    ) -> Result<Vec<Span>, RecognizerError> {
        let tensor = TextTensor::encode(
            &self.tokenizer,
            words.get(window.range()).unwrap_or_default(),
            self.width,
        )?;
        let mut session = Self::predict_lock(&self.text);
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(tensor.input_ids).map_err(Self::predict_onnx)?,
                "attention_mask" => Tensor::from_array(tensor.attention_mask).map_err(Self::predict_onnx)?,
                "words_mask" => Tensor::from_array(tensor.words_mask).map_err(Self::predict_onnx)?,
                "text_lengths" => Tensor::from_array(tensor.text_lengths).map_err(Self::predict_onnx)?,
                "span_idx" => Tensor::from_array(tensor.span_idx).map_err(Self::predict_onnx)?,
                "span_mask" => Tensor::from_array(tensor.span_mask).map_err(Self::predict_onnx)?,
                "labels_embeddings" => TensorRef::from_array_view(embeddings).map_err(Self::predict_onnx)?,
            ])
            .map_err(Self::predict_onnx)?;
        let logits = Self::predict_output(&outputs, "logits")?
            .into_dimensionality::<Ix4>()
            .map_err(Self::predict_shape)?;
        let words = window.end.saturating_sub(window.start);
        Ok(Span::scan(
            &logits.index_axis(Axis(0), 0),
            window.start,
            words,
            threshold,
        ))
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// Verifies the artifact, then builds both sessions and tokenizers; the window defaults to the model's
    /// `max_len`, so texts the model would truncate are windowed instead.
    ///
    /// # Errors
    /// Integrity, session, tokenizer or window failures.
    pub fn load(artifact: &ModelArtifact, config: &ModelConfig) -> Result<Self, OnnxError> {
        let version = artifact.verify()?;
        let window = config.window.unwrap_or(artifact.config.max_len);
        (window > config.overlap).then_some(()).ok_or(OnnxError::Window {
            window,
            overlap: config.overlap,
        })?;
        let threads = config.threads();
        let mut labels_tokenizer = Self::load_tokenizer(artifact, ModelArtifact::LABELS_TOKENIZER)?;
        // ### Pads are masked out of attention and pooling: the pad id only has to be a valid token.
        let pad_id = labels_tokenizer.get_padding().map_or(0, |padding| padding.pad_id);
        labels_tokenizer
            .with_padding(Some(PaddingParams {
                pad_id,
                ..PaddingParams::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: artifact.config.labels_max_len,
                ..TruncationParams::default()
            }))
            .map_err(|error| OnnxError::Load {
                file: ModelArtifact::LABELS_TOKENIZER,
                reason: error.to_string(),
            })?;
        let environment = ort::init()
            .with_name("gliner")
            .build()
            .map_err(|error| OnnxError::Load {
                file: ModelArtifact::TEXT,
                reason: error.to_string(),
            })?;
        let runtime = Self {
            text: Mutex::new(Self::load_session(
                &environment,
                artifact,
                ModelArtifact::TEXT,
                threads,
            )?),
            labels: Mutex::new(Self::load_session(
                &environment,
                artifact,
                ModelArtifact::LABELS,
                threads,
            )?),
            _environment: environment,
            tokenizer: Self::load_tokenizer(artifact, ModelArtifact::TOKENIZER)?,
            labels_tokenizer,
            splitter: WhitespaceSplitter::new().map_err(|error| OnnxError::Load {
                file: ModelArtifact::CONFIG,
                reason: error.to_string(),
            })?,
            cache: LabelCache::new(NonZeroUsize::new(config.cache).ok_or(OnnxError::Cache)?),
            width: artifact.config.max_width,
            window,
            overlap: config.overlap,
            model: artifact.manifest.model.clone(),
            version,
        };
        tracing::info!(model = %runtime.model, version = %runtime.version, threads, window, "runtime.loaded");
        Ok(runtime)
    }
}

impl std::fmt::Debug for OnnxRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnnxRuntime")
            .field("model", &self.model)
            .field("version", &self.version)
            .field("width", &self.width)
            .field("window", &self.window)
            .field("overlap", &self.overlap)
            .finish_non_exhaustive()
    }
}

impl Recognizer for OnnxRuntime {
    fn predict(&self, text: &str, labels: &[String], threshold: f32) -> Result<Vec<Entity>, RecognizerError> {
        let words = self.splitter.split(text);
        let windows = Window::plan(words.len(), self.window, self.overlap);
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        let embeddings = self.predict_labels(labels)?;
        let pieces: Vec<&str> = words.iter().map(|word| word.of(text)).collect();
        let candidates = windows
            .into_iter()
            .map(|window| self.predict_window(&pieces, window, &embeddings, threshold))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Span::select(candidates.into_iter().flatten().collect())
            .iter()
            .filter_map(|span| span.entity(text, &words, labels))
            .collect())
    }

    fn card(&self) -> ModelCard {
        ModelCard {
            model: self.model.clone(),
            version: self.version.clone(),
            labels_cached: self.cache.len(),
            served: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::config::model::ModelConfig;
    use crate::core::recognizer::Recognizer;
    use crate::runtime::artifact::ModelArtifact;
    use crate::runtime::onnx::{OnnxError, OnnxRuntime};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn runtime(window: Option<usize>, overlap: usize) -> Result<OnnxRuntime, OnnxError> {
        let artifact =
            ModelArtifact::open(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/tiny")).unwrap();
        let config = ModelConfig {
            threads: Some(1),
            window,
            overlap,
            ..ModelConfig::default()
        };
        OnnxRuntime::load(&artifact, &config)
    }

    #[test]
    fn test_predict_every_word_with_the_strongest_label() {
        let runtime = runtime(None, 2).unwrap();
        let text = "Ana vive en Málaga 🚀";
        let entities = runtime
            .predict(text, &labels(&["person", "software tool"]), 0.5)
            .unwrap();
        let found: Vec<(usize, usize, &str, &str)> = entities
            .iter()
            .map(|entity| (entity.start, entity.end, entity.text.as_str(), entity.label.as_str()))
            .collect();
        assert_eq!(
            found,
            [
                (0, 3, "Ana", "software tool"),
                (4, 8, "vive", "software tool"),
                (9, 11, "en", "software tool"),
                (12, 18, "Málaga", "software tool"),
                (19, 20, "🚀", "software tool")
            ]
        );
        assert!(entities.iter().all(|entity| (entity.score - sigmoid(4.0)).abs() < 1e-6));
    }

    #[test]
    fn test_predict_ties_go_to_the_first_label_and_threshold_is_strict() {
        let runtime = runtime(None, 2).unwrap();
        let entities = runtime.predict("Ana", &labels(&["location", "person"]), 0.5).unwrap();
        assert_eq!(entities[0].label, "location");
        assert!(
            runtime
                .predict("Ana", &labels(&["person"]), sigmoid(3.0))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_predict_windows_long_texts_without_duplicates() {
        let runtime = runtime(None, 2).unwrap();
        let text = "uno dos tres cuatro cinco seis siete ocho nueve diez once";
        let entities = runtime.predict(text, &labels(&["person"]), 0.5).unwrap();
        let words: Vec<&str> = entities.iter().map(|entity| entity.text.as_str()).collect();
        assert_eq!(words, text.split(' ').collect::<Vec<_>>());
    }

    #[test]
    fn test_predict_empty_text_encodes_nothing() {
        let runtime = runtime(Some(4), 1).unwrap();
        assert!(runtime.predict(" \n", &labels(&["person"]), 0.5).unwrap().is_empty());
        assert_eq!(runtime.card().labels_cached, 0);
    }

    #[test]
    fn test_card_reports_model_version_and_cache() {
        let runtime = runtime(None, 2).unwrap();
        runtime
            .predict("Ana", &labels(&["person", "location", "person"]), 0.5)
            .unwrap();
        let card = runtime.card();
        assert_eq!(
            (card.model.as_str(), card.version.len(), card.labels_cached),
            ("tiny", 16, 2)
        );
        assert!(format!("{runtime:?}").contains("tiny"));
    }

    #[test]
    fn test_load_rejects_a_window_within_the_overlap() {
        assert!(matches!(
            runtime(None, 32),
            Err(OnnxError::Window { window: 6, overlap: 32 })
        ));
        assert!(matches!(runtime(Some(3), 3), Err(OnnxError::Window { .. })));
    }
}
