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

/// The champion that cannot be beaten: the neutral eval set's gold entities, for the labels asked.
#[derive(Debug)]
pub struct GoldRecognizer {
    gold: std::collections::HashMap<String, Vec<gliner::schema::dataset::SpanDataset>>,
}

impl GoldRecognizer {
    pub fn new() -> Self {
        let lines = std::fs::read_to_string(crate::app::resource("eval/neutral.jsonl")).unwrap();
        Self {
            gold: lines
                .lines()
                .map(|line| serde_json::from_str::<gliner::schema::dataset::LineDataset>(line).unwrap())
                .map(|line| (line.text, line.entities))
                .collect(),
        }
    }
}

impl Recognizer for GoldRecognizer {
    fn predict(&self, text: &str, labels: &[String], _: f32) -> Result<Vec<Entity>, RecognizerError> {
        Ok(self
            .gold
            .get(text)
            .into_iter()
            .flatten()
            .filter(|span| labels.contains(&span.label))
            .map(|span| Entity {
                start: span.start,
                end: span.end,
                text: text.chars().skip(span.start).take(span.end - span.start).collect(),
                label: span.label.clone(),
                score: 1.0,
            })
            .collect())
    }

    fn card(&self) -> ModelCard {
        ModelCard {
            model: "gold".to_owned(),
            version: String::new(),
            labels_cached: 0,
            served: None,
        }
    }
}
