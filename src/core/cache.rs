use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, PoisonError};

use lru::LruCache;

use crate::schema::error::RecognizerError;

/// Embedding per label, least recently used evicted first: labels are free text per request, encoded once.
#[derive(Debug)]
pub struct LabelCache {
    entries: Mutex<LruCache<String, Arc<[f32]>>>,
}

impl LabelCache {
    #[must_use]
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            entries: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Labels currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One embedding per label, in order. Misses are encoded together by `encode` (one row per distinct missing
    /// label, outside the lock) and cached.
    ///
    /// # Errors
    /// `encode` failed or returned fewer rows than asked.
    pub fn resolve(
        &self,
        labels: &[String],
        encode: impl FnOnce(&[&str]) -> Result<Vec<Vec<f32>>, RecognizerError>,
    ) -> Result<Vec<Arc<[f32]>>, RecognizerError> {
        let hits: Vec<Option<Arc<[f32]>>> = {
            let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            labels.iter().map(|label| entries.get(label).cloned()).collect()
        };
        let mut seen = HashSet::new();
        let missing: Vec<&str> = labels
            .iter()
            .zip(&hits)
            .filter(|(label, hit)| hit.is_none() && seen.insert(label.as_str()))
            .map(|(label, _)| label.as_str())
            .collect();
        let rows = if missing.is_empty() {
            Vec::new()
        } else {
            encode(&missing)?
        };
        if rows.len() != missing.len() {
            return Err(RecognizerError::Shape(format!(
                "{} labels encoded, {} asked",
                rows.len(),
                missing.len()
            )));
        }
        let fresh: HashMap<&str, Arc<[f32]>> = missing.iter().copied().zip(rows.into_iter().map(Arc::from)).collect();
        {
            let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            for (label, row) in &fresh {
                entries.put((*label).to_owned(), Arc::clone(row));
            }
        }
        labels
            .iter()
            .zip(hits)
            .map(|(label, hit)| hit.or_else(|| fresh.get(label.as_str()).cloned()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| RecognizerError::Shape("a label lost its embedding".to_owned()))
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::cell::RefCell;
    use std::num::NonZeroUsize;

    use crate::core::cache::LabelCache;
    use crate::schema::error::RecognizerError;

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn encode(calls: &RefCell<Vec<Vec<String>>>) -> impl FnOnce(&[&str]) -> Result<Vec<Vec<f32>>, RecognizerError> {
        move |missing: &[&str]| {
            calls
                .borrow_mut()
                .push(missing.iter().map(|m| (*m).to_owned()).collect());
            Ok(missing.iter().map(|m| vec![m.len() as f32]).collect())
        }
    }

    #[test]
    fn test_resolve_encodes_each_miss_once() {
        let cache = LabelCache::new(NonZeroUsize::new(8).unwrap());
        let calls = RefCell::new(Vec::new());
        let first = cache
            .resolve(&labels(&["person", "city", "person"]), encode(&calls))
            .unwrap();
        let second = cache.resolve(&labels(&["city", "tool"]), encode(&calls)).unwrap();
        assert_eq!(first.iter().map(|row| row[0]).collect::<Vec<_>>(), [6.0, 4.0, 6.0]);
        assert_eq!(second.iter().map(|row| row[0]).collect::<Vec<_>>(), [4.0, 4.0]);
        assert_eq!(*calls.borrow(), [labels(&["person", "city"]), labels(&["tool"])]);
        assert_eq!(cache.len(), 3);
        cache.resolve(&labels(&["tool"]), encode(&calls)).unwrap();
        assert_eq!(calls.borrow().len(), 2);
    }

    #[test]
    fn test_resolve_evicts_least_recent_but_answers_in_full() {
        let cache = LabelCache::new(NonZeroUsize::new(1).unwrap());
        let calls = RefCell::new(Vec::new());
        let rows = cache.resolve(&labels(&["ab", "abc"]), encode(&calls)).unwrap();
        assert_eq!(rows.iter().map(|row| row[0]).collect::<Vec<_>>(), [2.0, 3.0]);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_resolve_rejects_short_encodings_and_propagates_errors() {
        let cache = LabelCache::new(NonZeroUsize::new(4).unwrap());
        let short = cache.resolve(&labels(&["a", "b"]), |_| Ok(vec![vec![1.0]]));
        assert!(matches!(short, Err(RecognizerError::Shape(_))));
        let failed = cache.resolve(&labels(&["a"]), |_| Err(RecognizerError::Onnx("boom".to_owned())));
        assert_eq!(failed, Err(RecognizerError::Onnx("boom".to_owned())));
        assert!(cache.is_empty());
    }
}
