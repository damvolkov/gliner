#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use std::path::Path;
use std::time::Duration;

use gliner::api::server::{HttpServer, ServerError};
use gliner::core::settings::Settings;
use gliner::runtime::onnx::OnnxError;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn settings(overlap: usize) -> Settings {
    let mut settings = Settings::default();
    settings.server.host = [127, 0, 0, 1].into();
    settings.server.port = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    settings.model.dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/tiny");
    settings.model.threads = Some(1);
    settings.model.overlap = overlap;
    settings
}

/// One HTTP/1.1 exchange over TCP: the status code and the JSON body, or `None` while nothing listens.
async fn http(port: u16, method: &str, path: &str, body: &str) -> Option<(u16, Value)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.ok()?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).await.ok()?;
    let status = response.get(9..12)?.parse().ok()?;
    let (_, payload) = response.split_once("\r\n\r\n")?;
    Some((status, serde_json::from_str(payload).ok()?))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_serve_loads_answers_and_stops_on_sigterm() {
    let settings = settings(2).await;
    let port = settings.server.port;
    let server = tokio::spawn(async move { HttpServer::serve(&settings).await });
    let ready = async {
        loop {
            if let Some((200, _)) = http(port, "GET", "/ready", "").await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), ready).await.unwrap();
    let (status, body) = http(port, "POST", "/ner", r#"{"text": "Ana", "labels": ["person"]}"#)
        .await
        .unwrap();
    assert_eq!((status, &body["entities"][0]["text"]), (200, &json!("Ana")));
    std::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status()
        .unwrap();
    let stopped = tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .unwrap()
        .unwrap();
    assert!(stopped.is_ok(), "{stopped:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_serve_fails_when_the_model_cannot_load() {
    let settings = settings(64).await;
    let failed = tokio::time::timeout(Duration::from_secs(10), HttpServer::serve(&settings))
        .await
        .unwrap();
    assert!(
        matches!(failed, Err(ServerError::Onnx(OnnxError::Window { .. }))),
        "{failed:?}"
    );
    let mut missing = settings.clone();
    missing.model.dir = "/nonexistent".into();
    assert!(matches!(
        HttpServer::serve(&missing).await,
        Err(ServerError::Artifact(_))
    ));
}
