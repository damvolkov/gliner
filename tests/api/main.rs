#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

mod app;
mod batch;
mod data;
mod eval;
mod fake;
mod health;
mod ner;
mod promote;
mod toolchain;
mod train;
