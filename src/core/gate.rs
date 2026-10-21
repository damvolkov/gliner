use crate::config::train::GateConfig;
use crate::schema::eval::DeltaEval;
use crate::schema::train::{CheckTrain, GateTrain};

/// The regression gate: a candidate passes when every clause holds on the pooled comparison — the strict
/// delta's estimate, the lower bound of its interval, and the lenient delta, each at least its minimum.
#[derive(Debug)]
pub struct Gate;

impl Gate {
    #[must_use]
    pub fn verdict(config: &GateConfig, strict: &DeltaEval, lenient: &DeltaEval) -> GateTrain {
        let checks: Vec<CheckTrain> = [
            ("strict", strict.point, config.strict),
            ("floor", strict.low, config.floor),
            ("lenient", lenient.point, config.lenient),
        ]
        .into_iter()
        .map(|(name, value, min)| CheckTrain {
            name: name.to_owned(),
            value,
            min,
            passed: value >= min,
        })
        .collect();
        GateTrain {
            passed: checks.iter().all(|check| check.passed),
            checks,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::train::GateConfig;
    use crate::core::gate::Gate;
    use crate::schema::eval::DeltaEval;

    const fn delta(point: f64, low: f64) -> DeltaEval {
        DeltaEval {
            point,
            low,
            high: point,
        }
    }

    fn passes(strict: DeltaEval, lenient: DeltaEval) -> (bool, Vec<bool>) {
        let verdict = Gate::verdict(&GateConfig::default(), &strict, &lenient);
        (
            verdict.passed,
            verdict.checks.iter().map(|check| check.passed).collect(),
        )
    }

    #[test]
    fn test_verdict_defaults() {
        assert_eq!(
            passes(delta(0.0, -1.0), delta(-1.0, -3.0)),
            (true, vec![true, true, true])
        );
        assert_eq!(
            passes(delta(-0.1, 0.5), delta(2.0, 1.0)),
            (false, vec![false, true, true])
        );
        assert_eq!(
            passes(delta(0.4, -1.2), delta(2.0, 1.0)),
            (false, vec![true, false, true])
        );
        assert_eq!(
            passes(delta(0.4, 0.1), delta(-1.5, -2.0)),
            (false, vec![true, true, false])
        );
        assert!(!passes(delta(f64::NAN, 0.0), delta(0.0, 0.0)).0);
    }

    #[test]
    fn test_verdict_names_values_and_minimums() {
        let config = GateConfig {
            strict: 1.0,
            floor: 0.0,
            lenient: 0.5,
        };
        let verdict = Gate::verdict(&config, &delta(1.5, 0.2), &delta(0.4, 0.0));
        let checks: Vec<(&str, f64, f64)> = verdict
            .checks
            .iter()
            .map(|check| (check.name.as_str(), check.value, check.min))
            .collect();
        assert_eq!(
            checks,
            [("strict", 1.5, 1.0), ("floor", 0.2, 0.0), ("lenient", 0.4, 0.5)]
        );
        assert!(!verdict.passed);
    }
}
