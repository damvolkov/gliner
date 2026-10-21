use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::core::bootstrap::Bootstrap;
use crate::core::example::Example;
use crate::core::gate::Gate;
use crate::core::recognizer::Recognizer;
use crate::core::score::TextScore;
use crate::runtime::board::BoardError;
use crate::runtime::executor::BoundedExecutor;
use crate::runtime::lab::{LabError, TrainLab};
use crate::runtime::onnx::OnnxRuntime;
use crate::runtime::registry::{ModelRegistry, RegistryError};
use crate::runtime::store::KindStore;
use crate::runtime::toolchain::{CallToolchain, ToolchainError};
use crate::schema::dataset::{CardDataset, LineDataset};
use crate::schema::entity::Entity;
use crate::schema::eval::CompareEval;
use crate::schema::registry::EntryRegistry;
use crate::schema::train::{JobTrain, ReportTrain, StateTrain, StepTrain, TrainRequest};

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error(transparent)]
    Lab(#[from] LabError),
    #[error(transparent)]
    Board(#[from] BoardError),
    #[error("params: {0}")]
    Params(&'static str),
    #[error(transparent)]
    Toolchain(#[from] ToolchainError),
    #[error("no eval set registered: a candidate is never accepted unvalidated")]
    Unvalidated,
    #[error("leakage: {0}")]
    Leakage(String),
    #[error("the candidate does not load: {0}")]
    Candidate(String),
    #[error("parity {parity:.4} is below {min}: the export does not reproduce PyTorch")]
    Parity { parity: f64, min: f64 },
    #[error("{path}: {reason}")]
    File { path: PathBuf, reason: String },
}

impl From<RegistryError> for JobError {
    fn from(error: RegistryError) -> Self {
        Self::Lab(error.into())
    }
}

/// `PyTorch`'s predictions as `tools/export/parity.py` writes them, one item per text in order.
#[derive(Debug, Deserialize)]
struct FixtureJob {
    items: Vec<ItemJob>,
}

#[derive(Debug, Deserialize)]
struct ItemJob {
    entities: Vec<Entity>,
}

/// An eval set in the pipeline: its gold lines and the candidate's predictions on them.
#[derive(Debug)]
struct SetJob {
    name: String,
    lines: Vec<LineDataset>,
    candidate: Vec<Vec<Entity>>,
}

/// One train job, run start to end without a human: leakage check, fine-tuning, export, export parity against
/// `PyTorch`, regression against the champion on every eval set, the gate, registration, and promotion when
/// asked and earned. Its scratch dir (the candidate until registered, the parity files) is removed however it
/// ends; cancelling drops the pipeline, which kills the tool running.
#[derive(Debug)]
pub struct TrainJob {
    lab: Arc<TrainLab>,
    id: u64,
    request: TrainRequest,
    base: u32,
    dataset: CardDataset,
    staging: PathBuf,
}

impl TrainJob {
    // ##### PRIVATE #####

    fn common_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs())
    }

    fn common_log(&self, line: String) {
        tracing::info!(job = self.id, %line, "job.log");
        self.lab.board.log(self.id, line);
    }

    fn common_step(&self, step: StepTrain) {
        self.lab.board.update(self.id, |job| job.step = step);
        self.common_log(format!("step {}", json!(step)));
    }

    async fn common_write(path: &Path, bytes: Vec<u8>) -> Result<(), JobError> {
        tokio::fs::write(path, bytes).await.map_err(|error| JobError::File {
            path: path.to_owned(),
            reason: error.to_string(),
        })
    }

    async fn run_tool(&self, call: CallToolchain) -> Result<(), JobError> {
        let log = |line: String| self.lab.board.log(self.id, line);
        Ok(self.lab.toolchain.run(&call, &log).await?)
    }

    /// Every eval set, read, once no dataset text appears in any of them.
    async fn run_leakage(&self) -> Result<BTreeMap<String, Vec<LineDataset>>, JobError> {
        self.common_step(StepTrain::Leakage);
        let store = &self.lab.store;
        let cards = store.list(KindStore::Evals).await.map_err(LabError::from)?;
        (!cards.is_empty()).then_some(()).ok_or(JobError::Unvalidated)?;
        let mut evals = BTreeMap::new();
        for card in cards {
            let lines: Vec<LineDataset> = store
                .read(KindStore::Evals, &card.name, |line| line)
                .await
                .map_err(LabError::from)?;
            evals.insert(card.name, lines);
        }
        let seen: HashSet<[u8; 32]> = store
            .read(KindStore::Datasets, &self.dataset.name, |line| {
                Example::fingerprint(&line.text)
            })
            .await
            .map_err(LabError::from)?;
        let leaks = Example::leaks(&seen, &evals);
        let plural = |count: usize| if count == 1 { "text" } else { "texts" };
        let shared: Vec<String> = leaks
            .iter()
            .map(|(set, count)| format!("eval set {set} shares {count} {} with the dataset", plural(*count)))
            .collect();
        shared
            .is_empty()
            .then_some(())
            .ok_or_else(|| JobError::Leakage(shared.join("; ")))?;
        self.common_log(format!("{} eval sets, no shared text", evals.len()));
        Ok(evals)
    }

    async fn run_train(&self, checkpoint: &Path) -> Result<(), JobError> {
        self.common_step(StepTrain::Train);
        tokio::fs::create_dir_all(&self.staging)
            .await
            .map_err(|error| JobError::File {
                path: self.staging.clone(),
                reason: error.to_string(),
            })?;
        let base = self.lab.registry.source(self.base)?;
        let dataset = self
            .lab
            .store
            .path(KindStore::Datasets, &self.dataset.name)
            .map_err(LabError::from)?;
        self.run_tool(CallToolchain::Train {
            base,
            dataset,
            out: checkpoint.to_owned(),
            params: self.request.params.clone(),
        })
        .await
    }

    async fn run_export(&self, checkpoint: &Path, candidate: &Path) -> Result<Arc<OnnxRuntime>, JobError> {
        self.common_step(StepTrain::Export);
        self.run_tool(CallToolchain::Export {
            checkpoint: checkpoint.to_owned(),
            name: self.lab.registry.model(self.base)?,
            out: candidate.to_owned(),
        })
        .await?;
        self.lab.open(candidate.to_owned()).await.map_err(JobError::Candidate)
    }

    /// The candidate's predictions on every eval set, once they match `PyTorch`'s on enough of them: the share
    /// of identical `(text, start, end, label)` over all sets (Jaccard).
    async fn run_parity(
        &self,
        checkpoint: &Path,
        executor: &BoundedExecutor,
        evals: BTreeMap<String, Vec<LineDataset>>,
    ) -> Result<(f64, Vec<SetJob>), JobError> {
        self.common_step(StepTrain::Parity);
        let (mut torch, mut ours) = (BTreeSet::new(), BTreeSet::new());
        let mut sets = Vec::with_capacity(evals.len());
        for (name, lines) in evals {
            let reference = self.run_reference(checkpoint, &name, &lines).await?;
            let candidate = self.lab.predict(executor, &name, &lines).await?;
            let keyed = |found: &[Vec<Entity>]| -> Vec<(String, usize, usize, usize, String)> {
                found
                    .iter()
                    .enumerate()
                    .flat_map(|(text, entities)| {
                        entities
                            .iter()
                            .map(move |entity| (text, entity.start, entity.end, entity.label.clone()))
                    })
                    .map(|(text, start, end, label)| (name.clone(), text, start, end, label))
                    .collect()
            };
            torch.extend(keyed(&reference));
            ours.extend(keyed(&candidate));
            sets.push(SetJob { name, lines, candidate });
        }
        let union = torch.union(&ours).count();
        let parity = Self::run_ratio(torch.intersection(&ours).count(), union);
        self.common_log(format!(
            "parity {parity:.4}: {} PyTorch entities, {} exported",
            torch.len(),
            ours.len()
        ));
        let min = self.lab.train.parity;
        (parity >= min).then_some(()).ok_or(JobError::Parity { parity, min })?;
        Ok((parity, sets))
    }

    #[allow(clippy::cast_precision_loss)]
    fn run_ratio(shared: usize, union: usize) -> f64 {
        // ### Nothing found by either side is perfect agreement; counts stay exact in f64.
        if union == 0 { 1.0 } else { shared as f64 / union as f64 }
    }

    async fn run_reference(
        &self,
        checkpoint: &Path,
        name: &str,
        lines: &[LineDataset],
    ) -> Result<Vec<Vec<Entity>>, JobError> {
        let (texts, out) = (
            self.staging.join(format!("{name}.texts.json")),
            self.staging.join(format!("{name}.torch.json")),
        );
        let rows: Vec<_> = lines
            .iter()
            .enumerate()
            .map(|(id, line)| json!({"id": id.to_string(), "text": line.text}))
            .collect();
        Self::common_write(&texts, serde_json::to_vec(&rows).unwrap_or_default()).await?;
        self.run_tool(CallToolchain::Reference {
            checkpoint: checkpoint.to_owned(),
            texts,
            labels: TrainLab::labels(lines),
            threshold: self.lab.model.threshold,
            out: out.clone(),
        })
        .await?;
        let unreadable = |reason: String| JobError::File {
            path: out.clone(),
            reason,
        };
        let bytes = tokio::fs::read(&out)
            .await
            .map_err(|error| unreadable(error.to_string()))?;
        let fixture: FixtureJob = serde_json::from_slice(&bytes).map_err(|error| unreadable(error.to_string()))?;
        (fixture.items.len() == lines.len())
            .then_some(())
            .ok_or_else(|| unreadable(format!("{} items for {} texts", fixture.items.len(), lines.len())))?;
        Ok(fixture.items.into_iter().map(|item| item.entities).collect())
    }

    fn run_compare(champion: &[TextScore], candidate: &[TextScore]) -> CompareEval {
        let pairs = |pick: fn(&TextScore) -> crate::core::tally::Tally| -> Vec<_> {
            champion
                .iter()
                .zip(candidate)
                .map(|(a, b)| (pick(a), pick(b)))
                .collect()
        };
        let bootstrap = Bootstrap::default();
        CompareEval {
            champion: TextScore::report(champion),
            candidate: TextScore::report(candidate),
            strict: bootstrap.delta(&pairs(|score| score.strict)),
            lenient: bootstrap.delta(&pairs(|score| score.lenient)),
        }
    }

    /// Champion and candidate on every set, and on all of them pooled.
    async fn run_regression(
        &self,
        sets: &[SetJob],
    ) -> Result<(u32, BTreeMap<String, CompareEval>, CompareEval), JobError> {
        self.common_step(StepTrain::Regression);
        let champion = self.lab.slot.get().ok_or(LabError::NotReady)?;
        let (mut pooled_champion, mut pooled_candidate) = (Vec::new(), Vec::new());
        let mut compared = BTreeMap::new();
        for set in sets {
            let found = self.lab.predict(&champion.executor, &set.name, &set.lines).await?;
            let score = |predicted: &[Vec<Entity>]| -> Vec<TextScore> {
                set.lines
                    .iter()
                    .zip(predicted)
                    .map(|(gold, found)| TextScore::new(gold, found))
                    .collect()
            };
            let (old, new) = (score(&found), score(&set.candidate));
            compared.insert(set.name.clone(), Self::run_compare(&old, &new));
            pooled_champion.extend(old);
            pooled_candidate.extend(new);
        }
        let pooled = Self::run_compare(&pooled_champion, &pooled_candidate);
        self.common_log(format!(
            "pooled over {} texts: strict {:+.2} [{:+.2}, {:+.2}], lenient {:+.2} F1 points against version {}",
            pooled_champion.len(),
            pooled.strict.point,
            pooled.strict.low,
            pooled.strict.high,
            pooled.lenient.point,
            champion.version
        ));
        Ok((champion.version, compared, pooled))
    }

    async fn run_pipeline(&self) -> Result<(), JobError> {
        let evals = self.run_leakage().await?;
        let candidate = self.staging.join("candidate");
        let checkpoint = candidate.join(ModelRegistry::CHECKPOINT);
        self.run_train(&checkpoint).await?;
        let runtime = self.run_export(&checkpoint, &candidate).await?;
        let served = self.lab.slot.get().ok_or(LabError::NotReady)?;
        let executor = served.executor.share(Arc::clone(&runtime) as Arc<dyn Recognizer>);
        let (parity, sets) = self.run_parity(&checkpoint, &executor, evals).await?;
        let (champion, compared, pooled) = self.run_regression(&sets).await?;
        self.common_step(StepTrain::Gate);
        let gate = Gate::verdict(&self.lab.train.gate, &pooled.strict, &pooled.lenient);
        self.common_log(format!("gate {}", if gate.passed { "passed" } else { "failed" }));
        let report = ReportTrain {
            champion,
            parity,
            sets: compared,
            pooled,
            gate,
        };
        self.lab.board.update(self.id, |job| job.report = Some(report.clone()));
        self.common_step(StepTrain::Register);
        let passed = report.gate.passed;
        let entry = EntryRegistry {
            version: 0,
            parent: Some(self.base),
            base: self.lab.registry.model(self.base)?,
            dataset: Some(self.dataset.clone()),
            params: Some(self.request.params.clone()),
            created: Self::common_now(),
            report: Some(report),
        };
        let registry = self.lab.registry.clone();
        let version = tokio::task::spawn_blocking(move || registry.register(&candidate, entry))
            .await
            .map_err(|error| JobError::Candidate(error.to_string()))??;
        self.lab.board.update(self.id, |job| job.version = Some(version));
        self.common_log(format!("registered as version {version}"));
        if self.request.auto_promote && passed {
            self.common_step(StepTrain::Promote);
            self.lab.serve(version, runtime).await?;
            self.lab.board.update(self.id, |job| job.promoted = true);
            self.common_log(format!("version {version} served"));
        }
        Ok(())
    }

    async fn run(self, cancel: CancellationToken) {
        let ended = tokio::select! {
            result = self.run_pipeline() => Some(result),
            () = cancel.cancelled() => None,
        };
        let (state, error) = match ended {
            Some(Ok(())) => (StateTrain::Succeeded, None),
            Some(Err(error)) => (StateTrain::Failed, Some(error.to_string())),
            None => (StateTrain::Cancelled, None),
        };
        // ### Absent when the job never reached training: nothing to clean.
        let _ = tokio::fs::remove_dir_all(&self.staging).await;
        self.common_log(format!("{}", json!(state)));
        self.lab.board.finish(self.id, state, error, Self::common_now());
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// Checks the request, opens the job and starts it in the background: what `POST /train` answers with.
    ///
    /// # Errors
    /// Invalid parameters, an unknown dataset or base version, nothing served yet, or a job already running.
    pub async fn start(lab: Arc<TrainLab>, request: TrainRequest) -> Result<JobTrain, JobError> {
        let params = &request.params;
        (params.epochs >= 1)
            .then_some(())
            .ok_or(JobError::Params("epochs must be at least 1"))?;
        (params.batch >= 1)
            .then_some(())
            .ok_or(JobError::Params("batch must be at least 1"))?;
        (params.lr > 0.0 && params.lr < 1.0)
            .then_some(())
            .ok_or(JobError::Params("lr must be in (0, 1)"))?;
        let dataset = lab
            .store
            .card(KindStore::Datasets, &request.dataset)
            .await
            .map_err(LabError::from)?;
        let served = lab.slot.get().ok_or(LabError::NotReady)?;
        let base = request.base.unwrap_or(served.version);
        lab.registry
            .path(base)
            .is_dir()
            .then_some(())
            .ok_or(RegistryError::Unknown(base))?;
        let (job, cancel) = lab.board.claim(|id| JobTrain {
            id,
            state: StateTrain::Running,
            step: StepTrain::Leakage,
            dataset: dataset.clone(),
            base,
            params: request.params.clone(),
            auto_promote: request.auto_promote,
            version: None,
            promoted: false,
            error: None,
            report: None,
            log: Vec::new(),
            created: Self::common_now(),
            finished: None,
        })?;
        let train = Self {
            staging: lab.registry.staging(job.id),
            lab: Arc::clone(&lab),
            id: job.id,
            request,
            base,
            dataset,
        };
        tracing::info!(job = job.id, base, dataset = %job.dataset.name, "job.started");
        lab.board.attach(job.id, tokio::spawn(train.run(cancel)));
        Ok(job)
    }
}
