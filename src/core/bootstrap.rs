use crate::core::tally::Tally;
use crate::schema::eval::DeltaEval;

/// Paired bootstrap of a difference in F1: texts resampled with replacement, the same draw for both models, the
/// interval read off the sorted deltas (nearest rank, 2.5 % and 97.5 %). Seeded, so a report never changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bootstrap {
    draws: usize,
    seed: u64,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Self::new(Self::DRAWS, Self::SEED)
    }
}

impl Bootstrap {
    // ##### PRIVATE #####

    /// `SplitMix64`: a few lines that fix the sequence forever, where a library generator may change between
    /// releases and silently change every past report.
    fn delta_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mixed = (*state ^ state.wrapping_shr(30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mixed = (mixed ^ mixed.wrapping_shr(27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ mixed.wrapping_shr(31)
    }

    /// An index below `len`, by multiply-shift (Lemire): no modulo bias worth the name, no rejection loop.
    fn delta_index(state: &mut u64, len: usize) -> usize {
        let wide = u128::from(Self::delta_next(state)).wrapping_mul(len as u128);
        usize::try_from(wide.wrapping_shr(64)).unwrap_or(0)
    }

    fn delta_points<'p>(pairs: impl Iterator<Item = &'p (Tally, Tally)>) -> f64 {
        let (champion, candidate) = pairs.fold(
            (Tally::default(), Tally::default()),
            |(champion, candidate), (one, other)| (champion.plus(*one), candidate.plus(*other)),
        );
        (candidate.f1() - champion.f1()) * 100.0
    }

    fn delta_rank(sorted: &[f64], per_mille: usize) -> f64 {
        let rank = sorted.len().saturating_mul(per_mille).div_ceil(1000).saturating_sub(1);
        sorted.get(rank).copied().unwrap_or(0.0)
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    pub const DRAWS: usize = 1000;
    pub const SEED: u64 = 0x6C69_6E65_7231;

    #[must_use]
    pub const fn new(draws: usize, seed: u64) -> Self {
        Self { draws, seed }
    }

    /// Candidate minus champion F1 in points, from one `(champion, candidate)` tally per text.
    #[must_use]
    pub fn delta(&self, pairs: &[(Tally, Tally)]) -> DeltaEval {
        let mut state = self.seed;
        let mut deltas: Vec<f64> = (0..self.draws)
            .map(|_| {
                Self::delta_points(
                    (0..pairs.len()).filter_map(|_| pairs.get(Self::delta_index(&mut state, pairs.len()))),
                )
            })
            .collect();
        deltas.sort_by(f64::total_cmp);
        DeltaEval {
            point: Self::delta_points(pairs.iter()),
            low: Self::delta_rank(&deltas, 25),
            high: Self::delta_rank(&deltas, 975),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::bootstrap::Bootstrap;
    use crate::core::tally::Tally;

    const fn tally(tp: usize, fp: usize, fn_: usize) -> Tally {
        Tally { tp, fp, fn_ }
    }

    fn mixed() -> Vec<(Tally, Tally)> {
        (0..40_usize)
            .map(|n| {
                (
                    tally(n % 5, n % 3, n % 4),
                    tally((n % 5).saturating_add(n % 2), n % 4, n % 3),
                )
            })
            .collect()
    }

    #[test]
    fn test_delta_identical_models_is_zero() {
        let pairs = vec![(tally(3, 1, 2), tally(3, 1, 2)); 10];
        let delta = Bootstrap::default().delta(&pairs);
        assert_eq!((delta.point, delta.low, delta.high), (0.0, 0.0, 0.0));
        let empty = Bootstrap::default().delta(&[]);
        assert_eq!((empty.point, empty.low, empty.high), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_delta_uniform_gain_has_a_point_interval() {
        let pairs = vec![(tally(1, 1, 1), tally(2, 0, 0)); 7];
        let delta = Bootstrap::default().delta(&pairs);
        assert!((delta.point - 50.0).abs() < 1e-9);
        assert!((delta.low - 50.0).abs() < 1e-9 && (delta.high - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_delta_is_seeded_and_brackets_the_estimate() {
        let pairs = mixed();
        let first = Bootstrap::default().delta(&pairs);
        assert_eq!(first, Bootstrap::default().delta(&pairs));
        assert_ne!(first, Bootstrap::new(Bootstrap::DRAWS, 7).delta(&pairs));
        assert!(first.low < first.point && first.point < first.high, "{first:?}");
    }

    #[test]
    fn test_delta_pinned_values() {
        let delta = Bootstrap::default().delta(&mixed());
        let pinned = (5.113_570_330_961_636, 3.280_267_756_795_807, 7.383_888_013_349_404);
        assert!((delta.point - pinned.0).abs() < 1e-9, "{delta:?}");
        assert!((delta.low - pinned.1).abs() < 1e-9, "{delta:?}");
        assert!((delta.high - pinned.2).abs() < 1e-9, "{delta:?}");
    }
}
