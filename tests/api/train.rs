use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::http::StatusCode;
use gliner::runtime::executor::{BoundedExecutor, ServedExecutor};
use serde_json::{Value, json};

use crate::app::{Lab, call, lab, lab_with, resource, send};
use crate::fake::GoldRecognizer;
use crate::toolchain::{FakeToolchain, ModeFake};

/// The job once it left `running`.
async fn finished(lab: &Lab, id: u64) -> Value {
    let wait = async {
        loop {
            let (status, job) = call(&lab.state, "GET", &format!("/train/{id}"), "").await;
            assert_eq!(status, StatusCode::OK, "{job}");
            if job["state"] != "running" {
                return job;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(30), wait).await.unwrap()
}

/// A dataset named `web` holding the repo's neutral training lines (none of them an eval text).
async fn dataset(lab: &Lab) {
    let lines = std::fs::read_to_string(resource("train/neutral.jsonl")).unwrap();
    assert_eq!(send(&lab.state, "PUT", "/datasets/web", &lines).await.0, StatusCode::OK);
}

async fn train(lab: &Lab, body: &str) -> Value {
    let (status, job) = call(&lab.state, "POST", "/train", body).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{job}");
    finished(lab, job["id"].as_u64().unwrap()).await
}

fn registered(lab: &Lab) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(lab.root.path().join("models"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_validates_registers_and_promotes() {
    let toolchain = Arc::new(FakeToolchain::new(ModeFake::Pass));
    let lab = lab_with(Arc::clone(&toolchain));
    dataset(&lab).await;
    let (status, started) = call(
        &lab.state,
        "POST",
        "/train",
        r#"{"dataset": "web", "auto_promote": true, "params": {"epochs": 1}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{started}");
    assert_eq!(
        (
            &started["id"],
            &started["state"],
            &started["base"],
            &started["dataset"]["lines"]
        ),
        (&json!(1), &json!("running"), &json!(1), &json!(8))
    );
    let job = finished(&lab, 1).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    assert_eq!(
        (&job["version"], &job["promoted"], &job["step"]),
        (&json!(3), &json!(true), &json!("promote"))
    );
    let report = &job["report"];
    assert_eq!(
        (&report["champion"], &report["parity"], &report["gate"]["passed"]),
        (&json!(1), &json!(1.0), &json!(true))
    );
    assert_eq!(
        report["pooled"]["strict"],
        json!({"point": 0.0, "low": 0.0, "high": 0.0})
    );
    assert_eq!(report["sets"]["neutral"]["candidate"]["strict"]["tp"], 5);
    assert!(job["log"].as_array().unwrap().contains(&json!("epoch 1")), "{job}");
    assert_eq!(toolchain.calls(), ["train", "export", "reference"]);
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 3);
    let root = lab.root.path().join("models");
    assert_eq!(std::fs::read_to_string(root.join("current")).unwrap(), "3\n");
    assert!(root.join("3/checkpoint/pytorch_model.bin").is_file());
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("3/manifest.json")).unwrap()).unwrap();
    let entry = &manifest["registry"];
    assert_eq!(
        (&entry["version"], &entry["parent"], &entry["base"]),
        (&json!(3), &json!(1), &json!("tiny"))
    );
    assert_eq!(
        (&entry["dataset"]["name"], &entry["params"]["epochs"]),
        (&json!("web"), &json!(1))
    );
    assert_eq!(entry["report"], job["report"]);
    assert_eq!(registered(&lab), ["1", "2", "3", "current"]);
    assert_eq!(
        call(&lab.state, "GET", "/train", "").await.1.as_array().unwrap().len(),
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_without_auto_promote_waits_for_promote() {
    let lab = lab();
    dataset(&lab).await;
    let job = train(&lab, r#"{"dataset": "web", "base": 2}"#).await;
    assert_eq!(
        (&job["state"], &job["version"], &job["promoted"]),
        (&json!("succeeded"), &json!(3), &json!(false))
    );
    assert_eq!(job["base"], 2);
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 1);
    let (status, card) = call(&lab.state, "POST", "/promote", r#"{"version": 3}"#).await;
    assert_eq!((status, &card["served"]), (StatusCode::OK, &json!(3)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_gate_failure_registers_without_promoting() {
    let lab = lab();
    let champion = lab.state.ready().unwrap();
    lab.state.slot.set(ServedExecutor {
        version: 1,
        executor: champion.executor.share(Arc::new(GoldRecognizer::new())),
    });
    dataset(&lab).await;
    let job = train(&lab, r#"{"dataset": "web", "auto_promote": true}"#).await;
    assert_eq!(
        (&job["state"], &job["version"], &job["promoted"]),
        (&json!("succeeded"), &json!(3), &json!(false))
    );
    let gate = &job["report"]["gate"];
    assert_eq!(gate["passed"], false);
    assert_eq!(gate["checks"][0]["name"], "strict");
    assert!(gate["checks"][0]["value"].as_f64().unwrap() < -50.0, "{gate}");
    assert_eq!(job["report"]["sets"]["neutral"]["champion"]["strict"]["f1"], 1.0);
    assert_eq!(call(&lab.state, "GET", "/models", "").await.1["model"], "gold");
    let _ = BoundedExecutor::share;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_refuses_leakage_and_an_empty_eval_base() {
    let lab = lab();
    let leaky = concat!(
        r#"{"text": "Alan Turing   nació en LONDRES y trabajó en Bletchley Park.", "entities": []}"#,
        "\n"
    );
    send(&lab.state, "PUT", "/datasets/leaky", leaky).await;
    let job = train(&lab, r#"{"dataset": "leaky"}"#).await;
    assert_eq!((&job["state"], &job["step"]), (&json!("failed"), &json!("leakage")));
    assert_eq!(job["error"], "leakage: eval set neutral shares 1 text with the dataset");
    std::fs::remove_file(lab.root.path().join("train/evals/neutral.jsonl")).unwrap();
    dataset(&lab).await;
    let job = train(&lab, r#"{"dataset": "web"}"#).await;
    assert_eq!(
        job["error"],
        "no eval set registered: a candidate is never accepted unvalidated"
    );
    assert_eq!(registered(&lab), ["1", "2", "current"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_fails_on_a_tool_failure_or_lost_parity() {
    for (mode, step, error) in [
        (
            ModeFake::Fail,
            "train",
            "trainer failed (exit status: 1): CUDA out of memory",
        ),
        (
            ModeFake::Diverge,
            "parity",
            "parity 0.0000 is below 0.99: the export does not reproduce PyTorch",
        ),
    ] {
        let lab = lab_with(Arc::new(FakeToolchain::new(mode)));
        dataset(&lab).await;
        let job = train(&lab, r#"{"dataset": "web", "auto_promote": true}"#).await;
        assert_eq!(
            (&job["state"], &job["step"], &job["error"]),
            (&json!("failed"), &json!(step), &json!(error))
        );
        assert_eq!(
            registered(&lab),
            ["1", "2", "current"],
            "nothing registered, no scratch left"
        );
        assert_eq!(call(&lab.state, "GET", "/models", "").await.1["served"], 1);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_one_at_a_time_and_cancel() {
    let toolchain = Arc::new(FakeToolchain::new(ModeFake::Hang));
    let lab = lab_with(Arc::clone(&toolchain));
    dataset(&lab).await;
    assert_eq!(
        call(&lab.state, "POST", "/train", r#"{"dataset": "web"}"#).await.0,
        StatusCode::ACCEPTED
    );
    assert_eq!(
        call(&lab.state, "POST", "/train", r#"{"dataset": "web"}"#).await,
        (
            StatusCode::CONFLICT,
            json!({"error": "job 1 is running: one job at a time"})
        )
    );
    let training = async {
        while toolchain.calls().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), training).await.unwrap();
    let (status, job) = call(&lab.state, "DELETE", "/train/1", "").await;
    assert_eq!(
        (status, &job["state"], &job["step"]),
        (StatusCode::OK, &json!("cancelled"), &json!("train"))
    );
    assert!(toolchain.dropped.load(Ordering::SeqCst), "the running tool was stopped");
    assert_eq!(registered(&lab), ["1", "2", "current"]);
    assert_eq!(
        call(&lab.state, "DELETE", "/train/1", "").await,
        (StatusCode::CONFLICT, json!({"error": "job 1 is not running"}))
    );
    assert_eq!(call(&lab.state, "GET", "/train/9", "").await.0, StatusCode::NOT_FOUND);
    assert_eq!(
        call(&lab.state, "DELETE", "/train/9", "").await.0,
        StatusCode::NOT_FOUND
    );
    let (status, next) = call(&lab.state, "POST", "/train", r#"{"dataset": "web"}"#).await;
    assert_eq!((status, &next["id"]), (StatusCode::ACCEPTED, &json!(2)));
    assert_eq!(call(&lab.state, "DELETE", "/train/2", "").await.1["state"], "cancelled");
}

#[tokio::test]
async fn test_train_refusals() {
    let lab = lab();
    dataset(&lab).await;
    let cases = [
        (
            r#"{"dataset": "absent"}"#,
            StatusCode::NOT_FOUND,
            "datasets absent does not exist",
        ),
        (
            r#"{"dataset": "web", "base": 9}"#,
            StatusCode::NOT_FOUND,
            "version 9 is not registered",
        ),
        (
            r#"{"dataset": "web", "params": {"epochs": 0}}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "params: epochs must be at least 1",
        ),
        (
            r#"{"dataset": "web", "params": {"batch": 0}}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "params: batch must be at least 1",
        ),
        (
            r#"{"dataset": "web", "params": {"lr": 2.0}}"#,
            StatusCode::UNPROCESSABLE_ENTITY,
            "params: lr must be in (0, 1)",
        ),
    ];
    for (body, status, error) in cases {
        assert_eq!(
            call(&lab.state, "POST", "/train", body).await,
            (status, json!({"error": error})),
            "{body}"
        );
    }
    let (status, _) = call(&lab.state, "POST", "/train", r#"{"dataset": "web", "epochs": 3}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(call(&lab.state, "GET", "/train", "").await.1, json!([]));
}
