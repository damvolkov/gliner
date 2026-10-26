use ndarray::ArrayView3;

use crate::core::word::Word;
use crate::schema::entity::Entity;

/// A scored candidate: words `start..end` (half-open, over the whole text) labelled with `label` (an index).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub label: usize,
    pub score: f32,
}

impl Span {
    /// Shares at least one word: half-open ranges, so adjacent spans do not overlap.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Candidates of one window's logits `[words, width, labels]` scoring strictly above `threshold`, in
    /// (start, width, label) order, offset to text positions; spans running past the window's `words` are dropped.
    #[must_use]
    pub fn scan(logits: &ArrayView3<'_, f32>, offset: usize, words: usize, threshold: f32) -> Vec<Self> {
        logits
            .indexed_iter()
            .filter_map(|((start, width, label), &logit)| {
                let score = 1.0 / (1.0 + (-logit).exp());
                let end = start.saturating_add(width).saturating_add(1);
                (score > threshold && end <= words).then(|| Self {
                    start: offset.saturating_add(start),
                    end: offset.saturating_add(end),
                    label,
                    score,
                })
            })
            .collect()
    }

    /// Flat greedy decoding: by descending score (stable on ties), keep each span that overlaps none kept;
    /// the result is ordered by start.
    #[must_use]
    pub fn select(mut candidates: Vec<Self>) -> Vec<Self> {
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept = candidates.into_iter().fold(Vec::<Self>::new(), |mut kept, candidate| {
            if kept.iter().all(|span| !span.overlaps(&candidate)) {
                kept.push(candidate);
            }
            kept
        });
        kept.sort_by_key(|span| span.start);
        kept
    }

    /// The entity this span names in `text`, split into `words`, with `labels` by index.
    #[must_use]
    pub fn entity(&self, text: &str, words: &[Word], labels: &[String]) -> Option<Entity> {
        let first = words.get(self.start)?;
        let last = words.get(self.end.checked_sub(1)?)?;
        Some(Entity {
            start: first.chars.start,
            end: last.chars.end,
            text: text.get(first.bytes.start..last.bytes.end)?.to_owned(),
            label: labels.get(self.label)?.clone(),
            score: self.score,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use crate::core::span::Span;
    use crate::core::splitter::WhitespaceSplitter;

    const fn span(start: usize, end: usize, label: usize, score: f32) -> Span {
        Span {
            start,
            end,
            label,
            score,
        }
    }

    #[test]
    fn test_overlaps_is_half_open() {
        assert!(!span(0, 2, 0, 0.9).overlaps(&span(2, 3, 0, 0.9)));
        assert!(span(0, 2, 0, 0.9).overlaps(&span(1, 3, 0, 0.9)));
        assert!(span(1, 2, 0, 0.9).overlaps(&span(0, 3, 1, 0.9)));
    }

    #[test]
    fn test_select_keeps_adjacent_spans() {
        let kept = Span::select(vec![span(2, 3, 1, 0.7), span(0, 2, 0, 0.9)]);
        assert_eq!(kept, [span(0, 2, 0, 0.9), span(2, 3, 1, 0.7)]);
    }

    #[test]
    fn test_select_greedy_by_score_then_ordered_by_start() {
        let kept = Span::select(vec![
            span(0, 1, 0, 0.6),
            span(0, 2, 1, 0.8),
            span(1, 3, 0, 0.95),
            span(4, 5, 2, 0.55),
            span(4, 5, 0, 0.65),
        ]);
        assert_eq!(kept, [span(0, 1, 0, 0.6), span(1, 3, 0, 0.95), span(4, 5, 0, 0.65)]);
    }

    #[test]
    fn test_select_ties_keep_the_first_candidate() {
        let kept = Span::select(vec![span(0, 1, 1, 0.7), span(0, 1, 0, 0.7)]);
        assert_eq!(kept, [span(0, 1, 1, 0.7)]);
        assert_eq!(Span::select(Vec::new()), []);
    }

    #[test]
    fn test_scan_thresholds_bounds_and_offsets() {
        let mut logits = Array3::from_elem((3, 2, 2), -5.0_f32);
        logits[[0, 0, 1]] = 2.0;
        logits[[1, 1, 0]] = 1.0;
        logits[[2, 1, 0]] = 4.0;
        logits[[2, 0, 0]] = 0.0;
        let found = Span::scan(&logits.view(), 10, 3, 0.5);
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        assert_eq!(found, [span(10, 11, 1, sigmoid(2.0)), span(11, 13, 0, sigmoid(1.0))]);
    }

    #[test]
    fn test_entity_uses_character_offsets() {
        let text = "Visité 🚀 Málaga Este";
        let words = WhitespaceSplitter::new().unwrap().split(text);
        let labels = ["person".to_owned(), "location".to_owned()];
        let entity = span(2, 4, 1, 0.75).entity(text, &words, &labels).unwrap();
        assert_eq!((entity.start, entity.end), (9, 20));
        assert_eq!(entity.text, "Málaga Este");
        assert_eq!(entity.label, "location");
        assert_eq!(
            text.chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect::<String>(),
            entity.text
        );
        assert!(span(2, 9, 1, 0.75).entity(text, &words, &labels).is_none());
        assert!(span(2, 3, 7, 0.75).entity(text, &words, &labels).is_none());
    }
}
