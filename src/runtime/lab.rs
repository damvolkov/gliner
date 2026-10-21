use std::collections::BTreeSet;
use std::sync::Arc;

use futures_util::{StreamExt, TryStreamExt, stream};

use crate::config::model::ModelConfig;
use crate::config::train::TrainConfig;
use crate::core::query::Query;
use crate::core::recognizer::Recognizer;
use crate::core::score::TextScore;
use crate::core::settings::Settings;
use crate::runtime::artifact::ModelArtifact;
use crate::runtime::executor::{ServedExecutor, SlotExecutor};
use crate::runtime::onnx::OnnxRuntime;
use crate::runtime::registry::{ModelRegistry, RegistryError};
use crate::runtime::store::{DataStore, KindStore, StoreError};
use crate::schema::card::ModelCard;
use crate::schema::dataset::LineDataset;
use crate::schema::entity::Entity;
use crate::schema::error::RecognizerError;
use crate::schema::eval::ReplyEval;

#[derive(Debug, thiserror::Error)]
pub enum LabError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("version {version} cannot be served: {reason}")]
    Load { version: u32, reason: String },
    #[error("eval set {0} has no labels to look for")]
    Unlabelled(String),
    #[error(transparent)]
    Recognizer(#[from] RecognizerError),
    #[error("the model is still loading")]
    NotReady,
}

/// The train role's shared ground: the registry, the data store, the served slot, and the model settings every
/// version is loaded with. Promotions are serialized; a version is loaded off the async runtime and swapped in
/// only once whole, so a failure leaves the champion serving.
#[derive(Debug)]
pub struct TrainLab {
    pub registry: ModelRegistry,
    pub store: DataStore,
    pub slot: Arc<SlotExecutor>,
    pub model: ModelConfig,
    pub train: TrainConfig,
    promoting: tokio::sync::Mutex<()>,
}

impl TrainLab {
    // ##### PRIVATE #####

    fn common_served(&self) -> Result<ServedExecutor, LabError> {
        self.slot.get().ok_or(LabError::NotReady)
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(registry: ModelRegistry, store: DataStore, slot: Arc<SlotExecutor>, settings: &Settings) -> Self {
        Self {
            registry,
            store,
            slot,
            model: settings.model.clone(),
            train: settings.train.clone(),
            promoting: tokio::sync::Mutex::new(()),
        }
    }

    /// A registered version, verified and loaded on a blocking thread.
    ///
    /// # Errors
    /// The version is not registered, or its artifact does not load.
    pub async fn load(&self, version: u32) -> Result<Arc<OnnxRuntime>, LabError> {
        let dir = self.registry.path(version);
        dir.is_dir().then_some(()).ok_or(RegistryError::Unknown(version))?;
        let config = self.model.clone();
        let failed = |reason: String| LabError::Load { version, reason };
        let loaded = tokio::task::spawn_blocking(move || {
            ModelArtifact::open(&dir)
                .map_err(|error| error.to_string())
                .and_then(|artifact| OnnxRuntime::load(&artifact, &config).map_err(|error| error.to_string()))
        })
        .await
        .map_err(|error| failed(error.to_string()))?
        .map_err(failed)?;
        Ok(Arc::new(loaded))
    }

    /// Serves a loaded version: `current` is rewritten first, then the slot swapped, behind the same permits.
    ///
    /// # Errors
    /// Nothing is served yet, or the pointer cannot be written (the champion keeps serving).
    pub fn install(&self, version: u32, recognizer: Arc<dyn Recognizer>) -> Result<ModelCard, LabError> {
        let champion = self.common_served()?;
        self.registry.point(version)?;
        let served = ServedExecutor {
            version,
            executor: champion.executor.share(recognizer),
        };
        self.slot.set(served.clone());
        tracing::info!(from = champion.version, to = version, "lab.promoted");
        Ok(served.card())
    }

    /// Loads a version and serves it; one promotion at a time.
    ///
    /// # Errors
    /// Nothing served yet, an unknown or unloadable version, or an unwritable pointer.
    pub async fn promote(&self, version: u32) -> Result<ModelCard, LabError> {
        let _promoting = self.promoting.lock().await;
        self.common_served()?;
        let recognizer = self.load(version).await?;
        self.install(version, recognizer)
    }

    /// The executor of a version: the served one when it is asked (or none is), else the version loaded behind
    /// the served permits, so evaluating never exceeds `model.concurrency`.
    ///
    /// # Errors
    /// Nothing served yet, or an unknown or unloadable version.
    pub async fn executor(&self, version: Option<u32>) -> Result<ServedExecutor, LabError> {
        let served = self.common_served()?;
        match version {
            None => Ok(served),
            Some(asked) if asked == served.version => Ok(served),
            Some(asked) => Ok(ServedExecutor {
                version: asked,
                executor: served.executor.share(self.load(asked).await?),
            }),
        }
    }

    /// Predictions for every line of a set, in order, looking for the set's own labels (sorted) at the default
    /// threshold.
    ///
    /// # Errors
    /// The set names no label, or a prediction failed.
    pub async fn predict(
        &self,
        served: &ServedExecutor,
        set: &str,
        lines: &[LineDataset],
    ) -> Result<Vec<Vec<Entity>>, LabError> {
        let labels: BTreeSet<&str> = lines
            .iter()
            .flat_map(|line| line.entities.iter().map(|span| span.label.as_str()))
            .collect();
        let query = Query::new(labels.into_iter().map(str::to_owned).collect(), self.model.threshold)
            .map_err(|_| LabError::Unlabelled(set.to_owned()))?;
        let texts: Vec<String> = lines.iter().map(|line| line.text.clone()).collect();
        stream::iter(texts)
            .map(|text| served.executor.predict(text, query.clone()))
            .buffered(self.model.concurrency.max(1))
            .map_err(LabError::from)
            .try_collect()
            .await
    }

    /// Scores a version (default: the served one) on an eval set.
    ///
    /// # Errors
    /// Unknown set or version, an unloadable version, a set without labels, or a failed prediction.
    pub async fn eval(&self, set: &str, version: Option<u32>) -> Result<ReplyEval, LabError> {
        let lines: Vec<LineDataset> = self.store.read(KindStore::Evals, set, |line| line).await?;
        let served = self.executor(version).await?;
        let predicted = self.predict(&served, set, &lines).await?;
        let scores: Vec<TextScore> = lines
            .iter()
            .zip(&predicted)
            .map(|(gold, found)| TextScore::new(gold, found))
            .collect();
        Ok(ReplyEval {
            set: set.to_owned(),
            version: served.version,
            report: TextScore::report(&scores),
        })
    }
}
