use std::net::{IpAddr, Ipv4Addr};

use serde::{Deserialize, Serialize};

use crate::config::log::LogConfig;

/// The HTTP surface: bind address, port, the longest NDJSON line `/ner/batch` accepts (bytes), and logging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: IpAddr,
    pub port: u16,
    pub line: usize,
    pub log: LogConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 5800,
            line: 1 << 20,
            log: LogConfig::default(),
        }
    }
}
