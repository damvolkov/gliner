use std::time::Instant;

use gliner::core::recognizer::Recognizer;

use crate::{fixture, runtime};

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next()?.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

#[test]
#[ignore = "requires the exported model: make export"]
fn test_bench_latency_and_memory() {
    let fixture = fixture();
    let started = Instant::now();
    let runtime = runtime();
    let load_s = started.elapsed().as_secs_f64();
    runtime
        .predict(
            "Warm-up: Docker and Python in Madrid.",
            &fixture.labels,
            fixture.threshold,
        )
        .unwrap();
    let mut ms: Vec<f64> = (0..3)
        .flat_map(|_| fixture.items.iter())
        .map(|item| {
            let tick = Instant::now();
            runtime.predict(&item.text, &fixture.labels, fixture.threshold).unwrap();
            tick.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    eprintln!(
        "bench: {} texts x3, threads {}, load {load_s:.1}s, ms/text mean {mean:.1} p50 {:.1} p95 {:.1}, peak RSS {:.0} MB",
        fixture.items.len(),
        gliner::core::settings::Settings::load(None).unwrap().model.threads(),
        at(0.5),
        at(0.95),
        peak_rss_mb()
    );
}
