use axum::http::StatusCode;
use serde_json::json;

use crate::app::{call, lab, send};

/// Every word is predicted as `software tool` by the tiny model (its longest label): counts checked by hand
/// against the neutral eval set.
#[tokio::test]
async fn test_eval_scores_a_set_on_the_served_or_any_version() {
    let lab = lab();
    let (status, reply) = call(&lab.state, "POST", "/eval", r#"{"set": "neutral"}"#).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(
        (
            reply["set"].as_str(),
            reply["version"].as_u64(),
            reply["texts"].as_u64()
        ),
        (Some("neutral"), Some(1), Some(10))
    );
    assert_eq!(
        (&reply["strict"]["tp"], &reply["strict"]["fp"], &reply["strict"]["fn"]),
        (&json!(5), &json!(105), &json!(23))
    );
    assert_eq!(
        (
            &reply["lenient"]["tp"],
            &reply["lenient"]["fp"],
            &reply["lenient"]["fn"]
        ),
        (&json!(17), &json!(88), &json!(11))
    );
    assert_eq!(reply["labels"]["person"]["fn"], 6);
    assert_eq!(reply["labels"]["software tool"]["tp"], 5);
    let (status, other) = call(&lab.state, "POST", "/eval", r#"{"set": "neutral", "version": 2}"#).await;
    assert_eq!(status, StatusCode::OK, "{other}");
    assert_eq!(other["version"], 2);
    assert_eq!(other["strict"], reply["strict"]);
}

#[tokio::test]
async fn test_eval_refusals() {
    let lab = lab();
    let (status, _) = call(&lab.state, "POST", "/eval", r#"{"set": "absent"}"#).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, reply) = call(&lab.state, "POST", "/eval", r#"{"set": "neutral", "version": 9}"#).await;
    assert_eq!(
        (status, reply),
        (StatusCode::NOT_FOUND, json!({"error": "version 9 is not registered"}))
    );
    send(
        &lab.state,
        "PUT",
        "/evals/bare",
        "{\"text\": \"nada\", \"entities\": []}\n",
    )
    .await;
    let (status, reply) = call(&lab.state, "POST", "/eval", r#"{"set": "bare"}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    let (status, _) = call(&lab.state, "POST", "/eval", r#"{"sett": "neutral"}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}
