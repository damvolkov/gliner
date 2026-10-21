use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use gliner::core::recognizer::Recognizer;
use gliner::schema::card::ModelCard;
use gliner::schema::entity::Entity;
use gliner::schema::error::RecognizerError;

/// Every capitalised word is an entity of the first label at 0.9; `boom` fails; `sleep:<ms> …` takes that long.
/// Counts calls and the most predictions ever in flight at once.
#[derive(Debug, Default)]
pub struct FakeRecognizer {
    pub calls: AtomicUsize,
    pub inflight: AtomicUsize,
    pub peak: AtomicUsize,
}

impl Recognizer for FakeRecognizer {
    fn predict(&self, text: &str, labels: &[String], threshold: f32) -> Result<Vec<Entity>, RecognizerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let now = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let pause = text
            .strip_prefix("sleep:")
            .and_then(|rest| rest.split(' ').next()?.parse().ok());
        std::thread::sleep(Duration::from_millis(pause.unwrap_or(0)));
        self.inflight.fetch_sub(1, Ordering::SeqCst);
        if text == "boom" {
            return Err(RecognizerError::Onnx("boom".to_owned()));
        }
        let mut start = 0;
        Ok(text
            .split(' ')
            .filter_map(|word| {
                let at = start;
                start += word.chars().count() + 1;
                (word.chars().next().is_some_and(char::is_uppercase) && 0.9 > threshold).then(|| Entity {
                    start: at,
                    end: at + word.chars().count(),
                    text: word.to_owned(),
                    label: labels[0].clone(),
                    score: 0.9,
                })
            })
            .collect())
    }

    fn card(&self) -> ModelCard {
        ModelCard {
            model: "fake".to_owned(),
            version: "0123456789abcdef".to_owned(),
            labels_cached: self.calls.load(Ordering::SeqCst),
            served: None,
        }
    }
}
