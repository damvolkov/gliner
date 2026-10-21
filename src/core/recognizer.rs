use crate::schema::card::ModelCard;
use crate::schema::entity::Entity;
use crate::schema::error::RecognizerError;

/// The port every model backend implements: CPU-bound and blocking, called off the async runtime.
pub trait Recognizer: Send + Sync + std::fmt::Debug {
    /// Entities of `text` for `labels` scoring above `threshold`, non-overlapping, ordered by start.
    ///
    /// # Errors
    /// Tokenization or inference failed.
    fn predict(&self, text: &str, labels: &[String], threshold: f32) -> Result<Vec<Entity>, RecognizerError>;

    /// The served model, its version and cache state.
    fn card(&self) -> ModelCard;
}
