use std::path::Path;

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use serde::{Deserialize, Serialize};

use crate::config::model::ModelConfig;
use crate::config::server::ServerConfig;
use crate::core::query::Query;

pub const DEFAULT_PATH: &str = "gliner.toml";
pub const ENV_PREFIX: &str = "GLINER_";

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("invalid settings: {0}")]
    Load(#[from] Box<figment::Error>),
    #[error("invalid settings: `{key}` {reason}")]
    Invalid { key: &'static str, reason: &'static str },
}

/// The service contract: serde defaults, then `gliner.toml`, then `GLINER_<SECTION>__<KEY>` overrides
/// (e.g. `GLINER_MODEL__THREADS=8`). Every table rejects unknown keys; values are then range-checked, so a
/// configuration that cannot run never starts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerConfig,
    pub model: ModelConfig,
}

impl Settings {
    // ##### PRIVATE #####

    fn validate_rules(&self) -> [(bool, &'static str, &'static str); 7] {
        let model = &self.model;
        [
            (self.server.line >= 1024, "server.line", "must be at least 1024 bytes"),
            (model.concurrency >= 1, "model.concurrency", "must be at least 1"),
            (
                model.threads.is_none_or(|threads| threads >= 1),
                "model.threads",
                "must be at least 1",
            ),
            (model.cache >= 1, "model.cache", "must be at least 1"),
            (Query::bounded(model.threshold), "model.threshold", "must be in (0, 1)"),
            (
                !model.labels.is_empty() && model.labels.iter().all(|label| !label.trim().is_empty()),
                "model.labels",
                "must be a non-empty list of non-blank labels",
            ),
            (
                model.window.is_none_or(|window| window >= 1 && model.overlap < window),
                "model.window",
                "must be at least 1 and larger than model.overlap",
            ),
        ]
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// An explicit `path` must exist; without one, `gliner.toml` is read only if present.
    ///
    /// # Errors
    /// Missing explicit file, malformed TOML, unknown keys, ill-typed or out-of-range values.
    pub fn load(path: Option<&Path>) -> Result<Self, SettingsError> {
        let file = path.map_or_else(|| Toml::file(DEFAULT_PATH), Toml::file_exact);
        let settings: Self = Figment::from(file)
            .merge(Env::prefixed(ENV_PREFIX).split("__"))
            .extract()
            .map_err(|error| SettingsError::Load(Box::new(error)))?;
        settings.validate()?;
        Ok(settings)
    }

    /// # Errors
    /// The first rule the values break.
    pub fn validate(&self) -> Result<(), SettingsError> {
        self.validate_rules()
            .into_iter()
            .find(|(ok, ..)| !ok)
            .map_or(Ok(()), |(_, key, reason)| Err(SettingsError::Invalid { key, reason }))
    }
}

#[cfg(test)]
#[allow(clippy::result_large_err)]
mod tests {
    use std::path::{Path, PathBuf};

    use figment::Jail;

    use crate::config::log::LogFormat;
    use crate::core::settings::{Settings, SettingsError};

    fn invalid(toml: &str) -> Option<&'static str> {
        let mut key = None;
        Jail::expect_with(|jail| {
            jail.create_file("gliner.toml", toml)?;
            key = match Settings::load(None) {
                Err(SettingsError::Invalid { key, .. }) => Some(key),
                Err(SettingsError::Load(_)) => Some("load"),
                Ok(_) => None,
            };
            Ok(())
        });
        key
    }

    #[test]
    fn test_load_defaults_without_file() {
        Jail::expect_with(|_| {
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings, Settings::default());
            assert!(settings.model.threads() >= 1);
            Ok(())
        });
    }

    #[test]
    fn test_load_file_then_env_overrides() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "gliner.toml",
                "[server]\nport = 6000\nlog = { format = \"text\" }\n[model]\nthreads = 4\nlabels = [\"tool\"]\n",
            )?;
            jail.set_env("GLINER_SERVER__PORT", "7000");
            jail.set_env("GLINER_MODEL__DIR", "/models/bi");
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings.server.port, 7000);
            assert_eq!(settings.server.log.format, LogFormat::Text);
            assert_eq!(settings.model.dir, PathBuf::from("/models/bi"));
            assert_eq!(settings.model.threads(), 4);
            assert_eq!(settings.model.labels, ["tool"]);
            Ok(())
        });
    }

    #[test]
    fn test_rejects_unknown_keys_and_types() {
        assert_eq!(invalid("[model]\nthreshhold = 0.5\n"), Some("load"));
        assert_eq!(invalid("[models]\nthreads = 2\n"), Some("load"));
        assert_eq!(invalid("[server]\nport = \"http\"\n"), Some("load"));
    }

    #[test]
    fn test_rejects_values_that_cannot_run() {
        assert_eq!(invalid("[server]\nline = 10\n"), Some("server.line"));
        assert_eq!(invalid("[model]\nconcurrency = 0\n"), Some("model.concurrency"));
        assert_eq!(invalid("[model]\nthreads = 0\n"), Some("model.threads"));
        assert_eq!(invalid("[model]\ncache = 0\n"), Some("model.cache"));
        assert_eq!(invalid("[model]\nthreshold = 1.0\n"), Some("model.threshold"));
        assert_eq!(invalid("[model]\nlabels = []\n"), Some("model.labels"));
        assert_eq!(invalid("[model]\nlabels = [\"person\", \"\"]\n"), Some("model.labels"));
        assert_eq!(invalid("[model]\nwindow = 32\noverlap = 32\n"), Some("model.window"));
        assert_eq!(invalid("[model]\nwindow = 64\noverlap = 32\n"), None);
    }

    #[test]
    fn test_example_file_is_the_defaults() {
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("gliner.example.toml");
        Jail::expect_with(|_| {
            assert_eq!(Settings::load(Some(&example)).unwrap(), Settings::default());
            Ok(())
        });
    }

    #[test]
    fn test_load_rejects_missing_explicit_file() {
        Jail::expect_with(|_| {
            assert!(Settings::load(Some(Path::new("absent.toml"))).is_err());
            Ok(())
        });
    }
}
