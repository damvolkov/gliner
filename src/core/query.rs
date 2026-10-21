use std::sync::Arc;

use crate::schema::error::RequestError;

/// What to look for: labels (non-empty, none blank) and a threshold in (0, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub labels: Arc<[String]>,
    pub threshold: f32,
}

impl Query {
    /// # Errors
    /// No labels, a blank label, or a threshold outside (0, 1).
    pub fn new(labels: Arc<[String]>, threshold: f32) -> Result<Self, RequestError> {
        let blank = labels.iter().position(|label| label.trim().is_empty());
        match (labels.is_empty(), blank, Self::bounded(threshold)) {
            (true, _, _) => Err(RequestError::Empty),
            (false, Some(index), _) => Err(RequestError::Blank(index)),
            (false, None, false) => Err(RequestError::Threshold(threshold)),
            (false, None, true) => Ok(Self { labels, threshold }),
        }
    }

    /// Whether a threshold is usable: strictly between 0 and 1.
    #[must_use]
    pub fn bounded(threshold: f32) -> bool {
        threshold > 0.0 && threshold < 1.0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::query::Query;
    use crate::schema::error::RequestError;

    fn query(labels: &[&str], threshold: f32) -> Result<Query, RequestError> {
        Query::new(
            labels
                .iter()
                .map(|label| (*label).to_owned())
                .collect::<Arc<[String]>>(),
            threshold,
        )
    }

    #[test]
    fn test_new_validates_labels_and_threshold() {
        assert!(query(&["person"], 0.5).is_ok());
        assert_eq!(query(&[], 0.5), Err(RequestError::Empty));
        assert_eq!(query(&["person", " "], 0.5), Err(RequestError::Blank(1)));
        assert_eq!(query(&["person"], 1.0), Err(RequestError::Threshold(1.0)));
        assert_eq!(query(&["person"], 0.0), Err(RequestError::Threshold(0.0)));
        assert!(query(&["person"], f32::NAN).is_err());
    }
}
