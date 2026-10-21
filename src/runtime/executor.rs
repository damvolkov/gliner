use std::sync::{Arc, PoisonError, RwLock};

use tokio::sync::Semaphore;

use crate::core::query::Query;
use crate::core::recognizer::Recognizer;
use crate::schema::card::ModelCard;
use crate::schema::entity::Entity;
use crate::schema::error::RecognizerError;

/// Runs a [`Recognizer`] off the async runtime with at most `concurrency` predictions in flight; callers past
/// the bound wait for a permit (backpressure, never refusal). A permit lives as long as the blocking work, so a
/// caller that goes away mid-prediction still counts until the work ends.
#[derive(Debug, Clone)]
pub struct BoundedExecutor {
    recognizer: Arc<dyn Recognizer>,
    permits: Arc<Semaphore>,
}

impl BoundedExecutor {
    #[must_use]
    pub fn new(recognizer: Arc<dyn Recognizer>, concurrency: usize) -> Self {
        Self {
            recognizer,
            permits: Arc::new(Semaphore::new(concurrency.max(1))),
        }
    }

    /// # Errors
    /// The recognizer failed, or its worker died.
    pub async fn predict(&self, text: String, query: Query) -> Result<Vec<Entity>, RecognizerError> {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|error| RecognizerError::Worker(error.to_string()))?;
        let recognizer = Arc::clone(&self.recognizer);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            recognizer.predict(&text, &query.labels, query.threshold)
        })
        .await
        .map_err(|error| RecognizerError::Worker(error.to_string()))?
    }

    /// Another recognizer behind the same permits: the bound holds across a swap of the served model.
    #[must_use]
    pub fn share(&self, recognizer: Arc<dyn Recognizer>) -> Self {
        Self {
            recognizer,
            permits: Arc::clone(&self.permits),
        }
    }

    #[must_use]
    pub fn card(&self) -> ModelCard {
        self.recognizer.card()
    }
}

/// The model answering predictions: its registry version and its executor.
#[derive(Debug, Clone)]
pub struct ServedExecutor {
    pub version: u32,
    pub executor: BoundedExecutor,
}

impl ServedExecutor {
    #[must_use]
    pub fn card(&self) -> ModelCard {
        ModelCard {
            served: Some(self.version),
            ..self.executor.card()
        }
    }
}

/// The served model, replaced whole: a request clones what it finds (two `Arc`s) and keeps it to the end, so a
/// swap never cuts a prediction short and never shows half a model. Empty until the first load.
#[derive(Debug, Default)]
pub struct SlotExecutor {
    served: RwLock<Option<ServedExecutor>>,
}

impl SlotExecutor {
    #[must_use]
    pub fn get(&self) -> Option<ServedExecutor> {
        self.served.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Serves `served` from now on; the previous one, if any, lives on in the requests that hold it.
    pub fn set(&self, served: ServedExecutor) -> Option<ServedExecutor> {
        self.served
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .replace(served)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::query::Query;
    use crate::core::recognizer::Recognizer;
    use crate::runtime::executor::{BoundedExecutor, ServedExecutor, SlotExecutor};
    use crate::schema::card::ModelCard;
    use crate::schema::entity::Entity;
    use crate::schema::error::RecognizerError;

    #[derive(Debug)]
    struct PanickingRecognizer;

    impl Recognizer for PanickingRecognizer {
        fn predict(&self, text: &str, _: &[String], _: f32) -> Result<Vec<Entity>, RecognizerError> {
            panic!("{text}");
        }

        fn card(&self) -> ModelCard {
            ModelCard {
                model: "panic".to_owned(),
                version: String::new(),
                labels_cached: 0,
                served: None,
            }
        }
    }

    #[derive(Debug, Default)]
    struct SlowRecognizer {
        name: &'static str,
    }

    impl Recognizer for SlowRecognizer {
        fn predict(&self, _: &str, _: &[String], _: f32) -> Result<Vec<Entity>, RecognizerError> {
            std::thread::sleep(std::time::Duration::from_millis(30));
            Ok(Vec::new())
        }

        fn card(&self) -> ModelCard {
            ModelCard {
                model: self.name.to_owned(),
                version: String::new(),
                labels_cached: 0,
                served: None,
            }
        }
    }

    #[tokio::test]
    async fn test_slot_swaps_whole_and_readers_keep_theirs() {
        let slot = SlotExecutor::default();
        assert!(slot.get().is_none());
        let first = BoundedExecutor::new(Arc::new(SlowRecognizer { name: "one" }), 1);
        assert!(
            slot.set(ServedExecutor {
                version: 1,
                executor: first.clone()
            })
            .is_none()
        );
        let held = slot.get().unwrap();
        let second = first.share(Arc::new(SlowRecognizer { name: "two" }));
        let old = slot
            .set(ServedExecutor {
                version: 2,
                executor: second,
            })
            .unwrap();
        assert_eq!((old.version, held.version, held.card().served), (1, 1, Some(1)));
        assert_eq!(held.card().model, "one");
        let served = slot.get().unwrap();
        assert_eq!(
            (served.version, served.card().model.as_str(), served.card().served),
            (2, "two", Some(2))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_share_keeps_the_concurrency_bound() {
        let first = BoundedExecutor::new(Arc::new(SlowRecognizer { name: "one" }), 1);
        let second = first.share(Arc::new(SlowRecognizer { name: "two" }));
        let query = Query::new(Arc::from(["x".to_owned()]), 0.5).unwrap();
        let started = std::time::Instant::now();
        let (a, b) = tokio::join!(
            first.predict("a".to_owned(), query.clone()),
            second.predict("b".to_owned(), query)
        );
        assert!(a.is_ok() && b.is_ok());
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(60),
            "ran side by side"
        );
    }

    #[tokio::test]
    async fn test_predict_turns_a_dead_worker_into_an_error() {
        let executor = BoundedExecutor::new(Arc::new(PanickingRecognizer), 0);
        let query = Query::new(Arc::from(["x".to_owned()]), 0.5).unwrap();
        let failed = executor.predict("oops".to_owned(), query.clone()).await;
        assert!(matches!(failed, Err(RecognizerError::Worker(_))));
        assert!(matches!(
            executor.predict("again".to_owned(), query).await,
            Err(RecognizerError::Worker(_))
        ));
        assert_eq!(executor.card().model, "panic");
    }
}
