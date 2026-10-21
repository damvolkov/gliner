use axum::http::StatusCode;
use serde_json::{Value, json};

use crate::app::{call, lab, send, state};

const LINES: &str = concat!(
    r#"{"text": "Ana vive en Málaga", "entities": [{"start": 0, "end": 3, "label": "person"}]}"#,
    "\n",
    r#"{"text": "nada", "entities": []}"#,
    "\n"
);

#[tokio::test]
async fn test_datasets_and_evals_put_then_list() {
    let lab = lab();
    let (status, card) = send(&lab.state, "PUT", "/datasets/web", LINES).await;
    assert_eq!(status, StatusCode::OK, "{card}");
    assert_eq!((card["name"].as_str(), card["lines"].as_u64()), (Some("web"), Some(2)));
    assert_eq!(card["sha256"].as_str().unwrap().len(), 64);
    let (status, listed) = call(&lab.state, "GET", "/datasets", "").await;
    assert_eq!((status, listed), (StatusCode::OK, json!([card])));
    let (_, evals) = call(&lab.state, "GET", "/evals", "").await;
    let names: Vec<&str> = evals
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["neutral"]);
    let (status, _) = send(&lab.state, "PUT", "/evals/second", LINES).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        call(&lab.state, "GET", "/evals", "").await.1.as_array().unwrap().len(),
        2
    );
}

#[tokio::test]
async fn test_put_refuses_bad_uploads_whole() {
    let lab = lab();
    let bad = format!(
        "{LINES}{}\n",
        r#"{"text": "Ana", "entities": [{"start": 0, "end": 3, "label": ""}]}"#
    );
    let refusals: [(&str, &str, &str); 3] = [
        ("/datasets/web", &bad, "line 3: entity 0: the label is blank"),
        ("/datasets/web", "\n", "no examples: an empty set is refused"),
        ("/datasets/.web", LINES, "name"),
    ];
    for (uri, body, error) in refusals {
        let (status, reply) = send(&lab.state, "PUT", uri, body).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
        assert!(reply["error"].as_str().unwrap().contains(error), "{reply}");
    }
    assert_eq!(call(&lab.state, "GET", "/datasets", "").await.1, json!([]));
}

#[tokio::test]
async fn test_train_routes_answer_501_when_disabled() {
    let serving = state(1);
    let routes = [
        ("PUT", "/datasets/web"),
        ("GET", "/datasets"),
        ("PUT", "/evals/web"),
        ("GET", "/evals"),
        ("POST", "/eval"),
        ("POST", "/promote"),
    ];
    for (method, uri) in routes {
        let (status, reply): (StatusCode, Value) = send(&serving, method, uri, "{}").await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{method} {uri}");
        assert!(reply["error"].as_str().unwrap().contains("train.enabled"), "{reply}");
    }
}
