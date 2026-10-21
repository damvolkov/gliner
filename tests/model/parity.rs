use std::collections::BTreeSet;

use gliner::core::recognizer::Recognizer;

use crate::{fixture, runtime};

#[test]
#[ignore = "requires the exported model: make export"]
fn test_parity_with_pytorch() {
    let (fixture, runtime) = (fixture(), runtime());
    let predicted: Vec<_> = fixture
        .items
        .iter()
        .map(|item| runtime.predict(&item.text, &fixture.labels, fixture.threshold).unwrap())
        .collect();
    let strict = |entities: &mut dyn Iterator<Item = (&str, &gliner::schema::entity::Entity)>| {
        entities
            .map(|(id, e)| (id.to_owned(), e.start, e.end, e.label.clone()))
            .collect::<BTreeSet<_>>()
    };
    let lenient = |entities: &mut dyn Iterator<Item = (&str, &gliner::schema::entity::Entity)>| {
        entities
            .map(|(id, e)| (id.to_owned(), e.text.trim().to_lowercase()))
            .collect::<BTreeSet<_>>()
    };
    let torch = || {
        fixture
            .items
            .iter()
            .flat_map(|item| item.entities.iter().map(|e| (item.id.as_str(), e)))
    };
    let ours = || {
        fixture
            .items
            .iter()
            .zip(&predicted)
            .flat_map(|(item, entities)| entities.iter().map(|e| (item.id.as_str(), e)))
    };
    let (torch_strict, ours_strict) = (strict(&mut torch()), strict(&mut ours()));
    let jaccard =
        torch_strict.intersection(&ours_strict).count() as f64 / torch_strict.union(&ours_strict).count().max(1) as f64;
    let drift = torch()
        .flat_map(|(id, t)| {
            ours()
                .filter(move |(i, o)| *i == id && o.start == t.start && o.end == t.end && o.label == t.label)
                .map(move |(_, o)| (o.score - t.score).abs())
        })
        .fold(0.0_f32, f32::max);
    eprintln!(
        "parity: {} texts, torch {} entities, rust {}, identical (Jaccard) {:.4}, max score drift {drift:.2e}",
        fixture.items.len(),
        torch_strict.len(),
        ours_strict.len(),
        jaccard
    );
    assert!(jaccard >= 0.99, "identical predictions {jaccard:.4} < 0.99");
    assert_eq!(lenient(&mut torch()), lenient(&mut ours()), "lenient sets differ");
}
