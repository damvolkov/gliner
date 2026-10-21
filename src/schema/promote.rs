use serde::{Deserialize, Serialize};

/// `POST /promote`: the registered version to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromoteRequest {
    pub version: u32,
}
