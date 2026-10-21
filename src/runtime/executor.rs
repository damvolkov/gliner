use std::sync::Arc;

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

    #[must_use]
    pub fn card(&self) -> ModelCard {
        self.recognizer.card()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::query::Query;
    use crate::core::recognizer::Recognizer;
    use crate::runtime::executor::BoundedExecutor;
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
            }
        }
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
