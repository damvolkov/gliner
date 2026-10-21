use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::core::tally::Tally;
use crate::schema::dataset::LineDataset;
use crate::schema::entity::Entity;
use crate::schema::eval::ReportEval;

/// One text's predictions against its gold: strict (same span, same label), lenient (the distinct names the text
/// mentions, casefolded and trimmed, label ignored), and strict per label.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextScore {
    pub strict: Tally,
    pub lenient: Tally,
    pub labels: BTreeMap<String, Tally>,
}

impl TextScore {
    // ##### PRIVATE #####

    fn new_name(name: &str) -> String {
        name.trim().to_lowercase()
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(gold: &LineDataset, predicted: &[Entity]) -> Self {
        let truth: HashSet<(usize, usize, &str)> = gold
            .entities
            .iter()
            .map(|span| (span.start, span.end, span.label.as_str()))
            .collect();
        let found: HashSet<(usize, usize, &str)> = predicted
            .iter()
            .map(|entity| (entity.start, entity.end, entity.label.as_str()))
            .collect();
        let chars: Vec<char> = gold.text.chars().collect();
        let mentioned: HashSet<String> = gold
            .entities
            .iter()
            .filter_map(|span| chars.get(span.start..span.end))
            .map(|name| Self::new_name(&name.iter().collect::<String>()))
            .collect();
        let guessed: HashSet<String> = predicted.iter().map(|entity| Self::new_name(&entity.text)).collect();
        let only = |set: &HashSet<(usize, usize, &str)>, label: &str| -> HashSet<(usize, usize)> {
            set.iter()
                .filter(|(_, _, own)| *own == label)
                .map(|&(start, end, _)| (start, end))
                .collect()
        };
        let labels: BTreeSet<&str> = truth.iter().chain(&found).map(|&(_, _, label)| label).collect();
        Self {
            strict: Tally::of(&truth, &found),
            lenient: Tally::of(&mentioned, &guessed),
            labels: labels
                .into_iter()
                .map(|label| (label.to_owned(), Tally::of(&only(&truth, label), &only(&found, label))))
                .collect(),
        }
    }

    /// Micro-averaged over texts: tallies summed, then scored.
    #[must_use]
    pub fn report(scores: &[Self]) -> ReportEval {
        let labels: BTreeSet<&str> = scores
            .iter()
            .flat_map(|score| score.labels.keys())
            .map(String::as_str)
            .collect();
        ReportEval {
            texts: scores.len(),
            strict: scores.iter().map(|score| score.strict).sum::<Tally>().score(),
            lenient: scores.iter().map(|score| score.lenient).sum::<Tally>().score(),
            labels: labels
                .into_iter()
                .map(|label| {
                    let tally: Tally = scores.iter().filter_map(|score| score.labels.get(label)).copied().sum();
                    (label.to_owned(), tally.score())
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::score::TextScore;
    use crate::core::tally::Tally;
    use crate::schema::dataset::{LineDataset, SpanDataset};
    use crate::schema::entity::Entity;

    const TEXT: &str = "Ana vive en Málaga 🚀 con Ana";

    fn gold(spans: &[(usize, usize, &str)]) -> LineDataset {
        LineDataset {
            text: TEXT.to_owned(),
            entities: spans
                .iter()
                .map(|&(start, end, label)| SpanDataset {
                    start,
                    end,
                    label: label.to_owned(),
                })
                .collect(),
        }
    }

    fn predicted(spans: &[(usize, usize, &str)]) -> Vec<Entity> {
        spans
            .iter()
            .map(|&(start, end, label)| Entity {
                start,
                end,
                text: TEXT.chars().skip(start).take(end.saturating_sub(start)).collect(),
                label: label.to_owned(),
                score: 0.9,
            })
            .collect()
    }

    const fn tally(tp: usize, fp: usize, fn_: usize) -> Tally {
        Tally { tp, fp, fn_ }
    }

    #[test]
    fn test_new_strict_lenient_and_per_label() {
        let score = TextScore::new(
            &gold(&[(0, 3, "person"), (12, 18, "city"), (0, 3, "person")]),
            &predicted(&[(0, 3, "person"), (12, 18, "person"), (4, 8, "verb")]),
        );
        assert_eq!(score.strict, tally(1, 2, 1));
        assert_eq!(score.lenient, tally(2, 1, 0));
        assert_eq!(
            score.labels,
            BTreeMap::from([
                ("city".to_owned(), tally(0, 0, 1)),
                ("person".to_owned(), tally(1, 1, 0)),
                ("verb".to_owned(), tally(0, 1, 0)),
            ])
        );
    }

    #[test]
    fn test_new_lenient_names_are_distinct_per_text() {
        let score = TextScore::new(
            &gold(&[(0, 3, "person"), (25, 28, "person"), (19, 20, "emoji")]),
            &predicted(&[(25, 28, "city"), (19, 20, "emoji")]),
        );
        assert_eq!(score.lenient, tally(2, 0, 0));
        assert_eq!(score.strict, tally(1, 1, 2));
    }

    #[test]
    fn test_new_empty_text_scores_nothing() {
        assert_eq!(TextScore::new(&gold(&[]), &[]), TextScore::default());
        let missed = TextScore::new(&gold(&[(0, 3, "person")]), &[]);
        assert_eq!((missed.strict, missed.lenient), (tally(0, 0, 1), tally(0, 0, 1)));
    }

    #[test]
    fn test_report_sums_before_scoring() {
        let first = TextScore::new(&gold(&[(0, 3, "person")]), &predicted(&[(0, 3, "person")]));
        let second = TextScore::new(&gold(&[(12, 18, "city")]), &predicted(&[(12, 18, "person")]));
        let report = TextScore::report(&[first, second]);
        assert_eq!(report.texts, 2);
        assert_eq!((report.strict.tp, report.strict.fp, report.strict.fn_), (1, 1, 1));
        assert!((report.strict.f1 - 0.5).abs() < 1e-12);
        assert!((report.lenient.f1 - 1.0).abs() < 1e-12);
        assert_eq!(report.labels["person"].fp, 1);
        assert!((report.labels["person"].precision - 0.5).abs() < 1e-12);
        assert!((report.labels["city"].recall).abs() < 1e-12);
        assert_eq!(TextScore::report(&[]).texts, 0);
    }
}
