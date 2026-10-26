use std::path::Path;

use serde::{Deserialize, Serialize};
use toml::{Table, Value};

use crate::config::model::ModelConfig;
use crate::config::server::ServerConfig;
use crate::config::train::TrainConfig;
use crate::core::query::Query;

pub const DEFAULT_PATH: &str = "gliner.toml";
pub const ENV_PREFIX: &str = "GLINER_";

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("settings file: {0}")]
    Read(#[from] std::io::Error),
    #[error("invalid settings: {0}")]
    Load(#[from] Box<toml::de::Error>),
    #[error("invalid settings: `{key}` {reason}")]
    Invalid { key: &'static str, reason: &'static str },
}

/// The service contract: serde defaults, then `gliner.toml`, then `GLINER_<SECTION>__<KEY>` overrides
/// (e.g. `GLINER_MODEL__THREADS=8`; a value is read as TOML, so a list is `GLINER_TRAIN__TRAINER='["python", "train.py"]'`,
/// anything that is not TOML a string). Every table rejects unknown keys; values are then range-checked, so a
/// configuration that cannot run never starts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub train: TrainConfig,
}

impl Settings {
    // ##### PRIVATE #####

    fn validate_rules(&self) -> [(bool, &'static str, &'static str); 11] {
        let (model, train) = (&self.model, &self.train);
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
            (
                train.parity > 0.0 && train.parity <= 1.0,
                "train.parity",
                "must be in (0, 1]",
            ),
            (!train.trainer.is_empty(), "train.trainer", "must name a program"),
            (!train.exporter.is_empty(), "train.exporter", "must name a program"),
            (!train.reference.is_empty(), "train.reference", "must name a program"),
        ]
    }

    fn load_file(path: Option<&Path>, default: &Path) -> Result<Table, SettingsError> {
        let text = match path {
            Some(explicit) => std::fs::read_to_string(explicit)?,
            None => std::fs::read_to_string(default).unwrap_or_default(),
        };
        toml::from_str(&text).map_err(|error| SettingsError::Load(Box::new(error)))
    }

    fn load_env(vars: impl IntoIterator<Item = (String, String)>) -> Table {
        vars.into_iter()
            .filter_map(|(key, raw)| key.strip_prefix(ENV_PREFIX).map(|rest| (rest.to_lowercase(), raw)))
            .fold(Table::new(), |mut table, (key, raw)| {
                Self::load_nest(&mut table, &key.split("__").collect::<Vec<_>>(), Self::load_value(&raw));
                table
            })
    }

    /// An override's value as TOML (`8`, `true`, `["a", "b"]`), else the raw string.
    fn load_value(raw: &str) -> Value {
        toml::from_str::<Table>(&format!("value = {raw}"))
            .ok()
            .and_then(|mut table| table.remove("value"))
            .unwrap_or_else(|| Value::String(raw.to_owned()))
    }

    fn load_nest(table: &mut Table, path: &[&str], value: Value) {
        match path {
            [] => {}
            [last] => {
                table.insert((*last).to_owned(), value);
            }
            [head, rest @ ..] => {
                let entry = table
                    .entry((*head).to_owned())
                    .or_insert_with(|| Value::Table(Table::new()));
                if let Value::Table(inner) = entry {
                    Self::load_nest(inner, rest, value);
                }
            }
        }
    }

    /// `over` on `base`, table into table, any other value replacing.
    fn load_merge(base: &mut Table, over: Table) {
        over.into_iter()
            .for_each(|(key, value)| match (base.get_mut(&key), value) {
                (Some(Value::Table(inner)), Value::Table(more)) => Self::load_merge(inner, more),
                (_, value) => {
                    base.insert(key, value);
                }
            });
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// An explicit `path` must exist; without one, `gliner.toml` is read only if present; the process environment
    /// overrides both.
    ///
    /// # Errors
    /// Missing explicit file, malformed TOML, unknown keys, ill-typed or out-of-range values.
    pub fn load(path: Option<&Path>) -> Result<Self, SettingsError> {
        Self::from_sources(path, Path::new(DEFAULT_PATH), std::env::vars())
    }

    /// `load` over given sources: the file, the default one read when no explicit path is given, the variables.
    ///
    /// # Errors
    /// As `load`.
    pub fn from_sources(
        path: Option<&Path>,
        default: &Path,
        vars: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, SettingsError> {
        let mut table = Self::load_file(path, default)?;
        Self::load_merge(&mut table, Self::load_env(vars));
        let settings: Self = Value::Table(table)
            .try_into()
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

    use crate::config::log::LogFormat;
    use crate::core::settings::{Settings, SettingsError};

    /// The settings a `gliner.toml` holding `toml` gives under `vars`, read from a directory of its own.
    fn loaded(toml: &str, vars: &[(&str, &str)]) -> Result<Settings, SettingsError> {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gliner.toml");
        std::fs::write(&file, toml).unwrap();
        let vars = vars.iter().map(|(key, value)| ((*key).to_owned(), (*value).to_owned()));
        Settings::from_sources(None, &file, vars)
    }

    fn invalid(toml: &str) -> Option<&'static str> {
        match loaded(toml, &[]) {
            Err(SettingsError::Invalid { key, .. }) => Some(key),
            Err(SettingsError::Load(_) | SettingsError::Read(_)) => Some("load"),
            Ok(_) => None,
        }
    }

    #[test]
    fn test_load_defaults_without_file() {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings::from_sources(None, &dir.path().join("gliner.toml"), []).unwrap();
        assert_eq!(settings, Settings::default());
        assert!(settings.model.threads() >= 1);
    }

    #[test]
    fn test_load_file_then_env_overrides() {
        let settings = loaded(
            "[server]\nport = 6000\nlog = { format = \"text\" }\n[model]\nthreads = 4\nlabels = [\"tool\"]\n",
            &[
                ("GLINER_SERVER__PORT", "7000"),
                ("GLINER_MODEL__DIR", "/models/bi"),
                ("OTHER_MODEL__DIR", "/x"),
            ],
        )
        .unwrap();
        assert_eq!(settings.server.port, 7000);
        assert_eq!(settings.server.log.format, LogFormat::Text);
        assert_eq!(settings.model.dir, PathBuf::from("/models/bi"));
        assert_eq!(settings.model.threads(), 4);
        assert_eq!(settings.model.labels, ["tool"]);
    }

    #[test]
    fn test_rejects_unknown_keys_and_types() {
        assert_eq!(invalid("[model]\nthreshhold = 0.5\n"), Some("load"));
        assert_eq!(invalid("[models]\nthreads = 2\n"), Some("load"));
        assert_eq!(invalid("[server]\nport = \"http\"\n"), Some("load"));
        assert_eq!(invalid("[server\n"), Some("load"));
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
    fn test_load_train_section_and_rules() {
        let settings = loaded(
            "[train]\nenabled = true\nparity = 0.95\n[train.gate]\nfloor = -2.0\n",
            &[("GLINER_TRAIN__TRAINER", "[\"python\", \"train.py\"]")],
        )
        .unwrap();
        assert!(settings.train.enabled);
        assert_eq!(settings.train.trainer, ["python", "train.py"]);
        assert!((settings.train.parity - 0.95).abs() < f64::EPSILON);
        assert!((settings.train.gate.floor + 2.0).abs() < f64::EPSILON);
        assert_eq!(invalid("[train]\nparity = 0.0\n"), Some("train.parity"));
        assert_eq!(invalid("[train]\nparity = 1.5\n"), Some("train.parity"));
        assert_eq!(invalid("[train]\nexporter = []\n"), Some("train.exporter"));
        assert_eq!(invalid("[train.gate]\nstrict = 0.0\nstrictness = 1\n"), Some("load"));
    }

    #[test]
    fn test_env_value_not_toml_is_a_string() {
        let settings = loaded("", &[("GLINER_MODEL__DIR", "/models/with space")]).unwrap();
        assert_eq!(settings.model.dir, PathBuf::from("/models/with space"));
    }

    #[test]
    fn test_example_file_is_the_defaults() {
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("gliner.example.toml");
        assert_eq!(
            Settings::from_sources(Some(&example), Path::new("unused"), []).unwrap(),
            Settings::default()
        );
    }

    #[test]
    fn test_load_rejects_missing_explicit_file() {
        let missing = Settings::from_sources(Some(Path::new("/nonexistent/absent.toml")), Path::new("unused"), []);
        assert!(matches!(missing, Err(SettingsError::Read(_))));
    }
}
