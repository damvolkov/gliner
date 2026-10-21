use std::collections::{BTreeMap, HashSet};

use sha2::{Digest, Sha256};

use crate::schema::dataset::{LineDataset, SpanDataset};
use crate::schema::error::ExampleError;

/// Lines of datasets and eval sets: parsed and checked one by one, and fingerprinted to find texts shared
/// between a dataset and an eval set.
#[derive(Debug)]
pub struct Example;

impl Example {
    // ##### PRIVATE #####

    fn parse_entity(index: usize, entity: &SpanDataset, chars: usize) -> Option<ExampleError> {
        let (start, end) = (entity.start, entity.end);
        match (start < end, end <= chars, entity.label.trim().is_empty()) {
            (false, _, _) => Some(ExampleError::Empty { index, start, end }),
            (true, false, _) => Some(ExampleError::Past { index, end, chars }),
            (true, true, true) => Some(ExampleError::Blank(index)),
            (true, true, false) => None,
        }
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// # Errors
    /// Not a `{text, entities}` object, or an entity that is empty, reversed, past the text, or blank-labelled.
    pub fn parse(line: &str) -> Result<LineDataset, ExampleError> {
        let parsed: LineDataset = serde_json::from_str(line).map_err(|error| ExampleError::Json(error.to_string()))?;
        let chars = parsed.text.chars().count();
        parsed
            .entities
            .iter()
            .enumerate()
            .find_map(|(index, entity)| Self::parse_entity(index, entity, chars))
            .map_or(Ok(parsed), Err)
    }

    /// The sha256 of a text casefolded with its whitespace collapsed: the same text, however typed.
    #[must_use]
    pub fn fingerprint(text: &str) -> [u8; 32] {
        let normalized = text
            .split_whitespace()
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
            .join(" ");
        Sha256::digest(normalized.as_bytes()).into()
    }

    /// Per eval set, its texts whose fingerprint is among a dataset's (sets without any left out).
    #[must_use]
    pub fn leaks(dataset: &HashSet<[u8; 32]>, evals: &BTreeMap<String, Vec<LineDataset>>) -> BTreeMap<String, usize> {
        evals
            .iter()
            .map(|(name, lines)| {
                let shared = lines
                    .iter()
                    .filter(|line| dataset.contains(&Self::fingerprint(&line.text)))
                    .count();
                (name.clone(), shared)
            })
            .filter(|(_, shared)| *shared > 0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use crate::core::example::Example;
    use crate::schema::dataset::{LineDataset, SpanDataset};
    use crate::schema::error::ExampleError;

    fn line(text: &str) -> LineDataset {
        LineDataset {
            text: text.to_owned(),
            entities: Vec::new(),
        }
    }

    #[test]
    fn test_parse_accepts_character_offsets() {
        let parsed =
            Example::parse(r#"{"text": "Visité Málaga 🚀", "entities": [{"start": 7, "end": 13, "label": "city"}]}"#)
                .unwrap();
        assert_eq!(
            parsed.entities,
            [SpanDataset {
                start: 7,
                end: 13,
                label: "city".to_owned()
            }]
        );
        let edge = Example::parse(r#"{"text": "Ana 🚀", "entities": [{"start": 4, "end": 5, "label": "x"}]}"#);
        assert!(edge.is_ok(), "{edge:?}");
        assert!(Example::parse(r#"{"text": "", "entities": []}"#).is_ok());
    }

    #[test]
    fn test_parse_rejects_bad_lines() {
        let entity = |start: usize, end: usize, label: &str| {
            format!(
                r#"{{"text": "Ana 🚀", "entities": [{{"start": 0, "end": 3, "label": "p"}}, {{"start": {start}, "end": {end}, "label": "{label}"}}]}}"#
            )
        };
        assert_eq!(
            Example::parse(&entity(2, 2, "x")),
            Err(ExampleError::Empty {
                index: 1,
                start: 2,
                end: 2
            })
        );
        assert!(matches!(
            Example::parse(&entity(3, 1, "x")),
            Err(ExampleError::Empty { .. })
        ));
        assert_eq!(
            Example::parse(&entity(4, 6, "x")),
            Err(ExampleError::Past {
                index: 1,
                end: 6,
                chars: 5
            })
        );
        assert_eq!(Example::parse(&entity(0, 1, " ")), Err(ExampleError::Blank(1)));
        let json = |line: &str| matches!(Example::parse(line), Err(ExampleError::Json(_)));
        assert!(json("{broken"));
        assert!(json(r#"{"text": "a"}"#));
        assert!(json(r#"{"text": "a", "entities": [], "lang": "es"}"#));
        assert!(json(
            r#"{"text": "a", "entities": [{"start": -1, "end": 1, "label": "x"}]}"#
        ));
    }

    #[test]
    fn test_fingerprint_ignores_case_and_spacing() {
        assert_eq!(
            Example::fingerprint("Ana  vive\ten\nMÁLAGA "),
            Example::fingerprint(" ana vive en málaga")
        );
        assert_ne!(Example::fingerprint("Ana vive"), Example::fingerprint("Ana vive."));
    }

    #[test]
    fn test_leaks_counts_shared_texts_per_set() {
        let dataset: HashSet<[u8; 32]> = ["Ana vive en Málaga", "Eva  VIVE en Roma", "nadie"]
            .into_iter()
            .map(Example::fingerprint)
            .collect();
        let evals = BTreeMap::from([
            ("a".to_owned(), vec![line("eva vive en roma"), line("otra")]),
            ("b".to_owned(), vec![line("otra más")]),
            (
                "c".to_owned(),
                vec![line("ana vive en málaga"), line("Eva vive en Roma")],
            ),
        ]);
        assert_eq!(
            Example::leaks(&dataset, &evals),
            BTreeMap::from([("a".to_owned(), 1), ("c".to_owned(), 2)])
        );
        assert!(Example::leaks(&dataset, &BTreeMap::new()).is_empty());
    }
}
