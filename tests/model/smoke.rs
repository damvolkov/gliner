use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gliner::core::settings::Settings;
use gliner::runtime::artifact::ModelArtifact;
use gliner::runtime::executor::{BoundedExecutor, ServedExecutor, SlotExecutor};
use gliner::runtime::job::TrainJob;
use gliner::runtime::lab::TrainLab;
use gliner::runtime::onnx::OnnxRuntime;
use gliner::runtime::registry::ModelRegistry;
use gliner::runtime::store::DataStore;
use gliner::runtime::toolchain::ProcessToolchain;
use gliner::schema::train::{ParamsTrain, StateTrain, TrainRequest};

use crate::registry;

fn resource(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources").join(path)
}

/// The whole job for real: one epoch on the neutral training lines on CPU, export, parity with `PyTorch` on the
/// neutral eval set, regression against the exported model, gate, registration as version 2. Runs on a scratch
/// registry whose version 1 links to the exported model's files, so the real registry is never touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the exported model (make export) and uv; trains for real on CPU"]
async fn test_smoke_train_one_epoch_through_the_job() {
    let source = registry();
    let served = source.path(source.current().unwrap());
    let root = tempfile::tempdir().unwrap();
    let models = root.path().join("models");
    std::fs::create_dir_all(models.join("1")).unwrap();
    std::fs::read_dir(&served)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .for_each(|file| {
            std::os::unix::fs::symlink(&file, models.join("1").join(file.file_name().unwrap())).unwrap();
        });
    std::fs::write(models.join("current"), "1\n").unwrap();
    let mut settings = Settings::load(None).unwrap();
    settings.model.dir.clone_from(&models);
    settings.train.enabled = true;
    settings.train.data = root.path().join("train");
    let store = DataStore::open(&settings.train.data, settings.server.line).unwrap();
    std::fs::copy(
        resource("eval/neutral.jsonl"),
        settings.train.data.join("evals/neutral.jsonl"),
    )
    .unwrap();
    std::fs::copy(
        resource("train/neutral.jsonl"),
        settings.train.data.join("datasets/neutral.jsonl"),
    )
    .unwrap();
    let runtime = OnnxRuntime::load(&ModelArtifact::open(&models.join("1")).unwrap(), &settings.model).unwrap();
    let slot = Arc::new(SlotExecutor::default());
    slot.set(ServedExecutor {
        version: 1,
        executor: BoundedExecutor::new(Arc::new(runtime), settings.model.concurrency),
    });
    let lab = Arc::new(TrainLab::new(
        ModelRegistry::open(&models).unwrap(),
        store,
        slot,
        Arc::new(ProcessToolchain::new(&settings.train)),
        &settings,
    ));
    let request = TrainRequest {
        dataset: "neutral".to_owned(),
        base: None,
        params: ParamsTrain {
            epochs: 1,
            lr: 5e-6,
            batch: 4,
        },
        auto_promote: false,
    };
    let started = Instant::now();
    let id = TrainJob::start(Arc::clone(&lab), request).await.unwrap().id;
    let job = loop {
        let job = lab.board.get(id).unwrap();
        if job.state != StateTrain::Running {
            break job;
        }
        assert!(
            started.elapsed() < Duration::from_secs(1800),
            "still running after 30 min"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    };
    job.log
        .iter()
        .rev()
        .take(12)
        .rev()
        .for_each(|line| eprintln!("  | {line}"));
    let report = job.report.as_ref();
    eprintln!(
        "smoke: {:?} in {:.0}s, version {:?}, parity {:?}, pooled strict {:?}, lenient {:?}, gate passed {:?}, error {:?}",
        job.state,
        started.elapsed().as_secs_f64(),
        job.version,
        report.map(|report| report.parity),
        report.map(|report| report.pooled.strict),
        report.map(|report| report.pooled.lenient),
        report.map(|report| report.gate.passed),
        job.error
    );
    assert_eq!(job.state, StateTrain::Succeeded, "{:?}", job.error);
    assert_eq!(job.version, Some(2));
    assert!(report.unwrap().parity >= settings.train.parity);
    assert!(models.join("2/checkpoint").is_dir());
    assert!(ModelArtifact::open(&models.join("2")).unwrap().verify().is_ok());
}
