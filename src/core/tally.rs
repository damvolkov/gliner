use std::collections::HashSet;
use std::hash::Hash;
use std::iter::Sum;

use crate::schema::eval::ScoreEval;

/// True positives, false positives and false negatives of predictions against gold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tally {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

impl Tally {
    // ##### PRIVATE #####

    #[allow(clippy::cast_precision_loss)]
    fn ratio(part: usize, whole: usize) -> f64 {
        // ### Counts stay far below 2^52, where usize to f64 is exact; 0/0 is 0, as in scikit-learn.
        if whole == 0 { 0.0 } else { part as f64 / whole as f64 }
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    #[must_use]
    pub fn of<T: Eq + Hash>(gold: &HashSet<T>, predicted: &HashSet<T>) -> Self {
        let tp = gold.intersection(predicted).count();
        Self {
            tp,
            fp: predicted.len().saturating_sub(tp),
            fn_: gold.len().saturating_sub(tp),
        }
    }

    /// Field by field, saturating.
    #[must_use]
    pub const fn plus(self, other: Self) -> Self {
        Self {
            tp: self.tp.saturating_add(other.tp),
            fp: self.fp.saturating_add(other.fp),
            fn_: self.fn_.saturating_add(other.fn_),
        }
    }

    #[must_use]
    pub fn precision(&self) -> f64 {
        Self::ratio(self.tp, self.tp.saturating_add(self.fp))
    }

    #[must_use]
    pub fn recall(&self) -> f64 {
        Self::ratio(self.tp, self.tp.saturating_add(self.fn_))
    }

    /// `2·tp / (2·tp + fp + fn)`: the harmonic mean of precision and recall, without dividing twice.
    #[must_use]
    pub fn f1(&self) -> f64 {
        let doubled = self.tp.saturating_mul(2);
        Self::ratio(doubled, doubled.saturating_add(self.fp).saturating_add(self.fn_))
    }

    #[must_use]
    pub fn score(&self) -> ScoreEval {
        ScoreEval {
            precision: self.precision(),
            recall: self.recall(),
            f1: self.f1(),
            tp: self.tp,
            fp: self.fp,
            fn_: self.fn_,
        }
    }
}

impl Sum for Tally {
    fn sum<I: Iterator<Item = Self>>(tallies: I) -> Self {
        tallies.fold(Self::default(), Self::plus)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::core::tally::Tally;

    const fn tally(tp: usize, fp: usize, fn_: usize) -> Tally {
        Tally { tp, fp, fn_ }
    }

    #[test]
    fn test_of_counts_set_differences() {
        let gold = HashSet::from([1, 2, 3]);
        let predicted = HashSet::from([2, 3, 4, 5]);
        assert_eq!(Tally::of(&gold, &predicted), tally(2, 2, 1));
        assert_eq!(Tally::of(&HashSet::<u8>::new(), &HashSet::new()), tally(0, 0, 0));
    }

    #[test]
    fn test_score_ratios_and_zero_division() {
        let score = tally(1, 2, 1).score();
        assert!((score.precision - 1.0 / 3.0).abs() < 1e-12);
        assert!((score.recall - 0.5).abs() < 1e-12);
        assert!((score.f1 - 0.4).abs() < 1e-12);
        assert_eq!((score.tp, score.fp, score.fn_), (1, 2, 1));
        let none = tally(0, 0, 0).score();
        assert_eq!((none.precision, none.recall, none.f1), (0.0, 0.0, 0.0));
        let missed = tally(0, 0, 3);
        assert_eq!((missed.precision(), missed.recall(), missed.f1()), (0.0, 0.0, 0.0));
        assert!((tally(4, 0, 0).f1() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_adds_fieldwise() {
        let total: Tally = [tally(1, 2, 3), tally(4, 5, 6)].into_iter().sum();
        assert_eq!(total, tally(5, 7, 9));
        assert_eq!(std::iter::empty::<Tally>().sum::<Tally>(), Tally::default());
    }
}
