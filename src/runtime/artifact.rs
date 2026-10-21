use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("model file not found: {0}")]
    Missing(PathBuf),
    #[error("cannot read {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("cannot parse {path}: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
    #[error("unsupported {key} = {value:?}: this runtime implements {expected:?}")]
    Unsupported {
        key: &'static str,
        value: String,
        expected: &'static str,
    },
    #[error("manifest.json does not list {0}")]
    Unlisted(&'static str),
    #[error("{file} does not match manifest.json: sha256 {actual}, expected {expected}")]
    Digest {
        file: &'static str,
        expected: String,
        actual: String,
    },
}

/// The GLiNER config subset the runtime reads (`config.json`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ConfigArtifact {
    pub max_width: usize,
    pub max_len: usize,
    pub words_splitter_type: String,
    pub subtoken_pooling: String,
    pub labels_max_len: usize,
}

/// Provenance and integrity of the files (`manifest.json`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ManifestArtifact {
    pub model: String,
    pub files: BTreeMap<String, String>,
}

/// A model dir honouring the export contract, opened and checked cheaply; [`ModelArtifact::verify`] hashes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    pub dir: PathBuf,
    pub config: ConfigArtifact,
    pub manifest: ManifestArtifact,
}

impl ModelArtifact {
    // ##### PRIVATE #####

    fn open_json<T: DeserializeOwned>(path: &Path) -> Result<T, ArtifactError> {
        let file = File::open(path).map_err(|source| ArtifactError::Read {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_reader(io::BufReader::new(file)).map_err(|source| ArtifactError::Parse {
            path: path.to_owned(),
            source,
        })
    }

    fn open_supported(key: &'static str, value: &str, expected: &'static str) -> Result<(), ArtifactError> {
        (value == expected)
            .then_some(())
            .ok_or_else(|| ArtifactError::Unsupported {
                key,
                value: value.to_owned(),
                expected,
            })
    }

    fn verify_file(&self, name: &'static str) -> Result<String, ArtifactError> {
        let path = self.path(name);
        let read = |source| ArtifactError::Read {
            path: path.clone(),
            source,
        };
        let mut hasher = Sha256::new();
        io::copy(&mut File::open(&path).map_err(read)?, &mut hasher).map_err(read)?;
        let actual = format!("{:x}", hasher.finalize());
        let expected = self.manifest.files.get(name).ok_or(ArtifactError::Unlisted(name))?;
        (actual == *expected)
            .then_some(actual.clone())
            .ok_or_else(|| ArtifactError::Digest {
                file: name,
                expected: expected.clone(),
                actual,
            })
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    pub const TEXT: &str = "text.onnx";
    pub const LABELS: &str = "labels.onnx";
    pub const TOKENIZER: &str = "tokenizer.json";
    pub const LABELS_TOKENIZER: &str = "labels.tokenizer.json";
    pub const CONFIG: &str = "config.json";
    pub const MANIFEST: &str = "manifest.json";
    pub const HASHED: [&str; 5] = [
        Self::TEXT,
        Self::LABELS,
        Self::TOKENIZER,
        Self::LABELS_TOKENIZER,
        Self::CONFIG,
    ];

    /// Every contract file present, `config.json` and `manifest.json` parsed, the words splitter and pooling
    /// the ones this runtime implements, every hashed file listed.
    ///
    /// # Errors
    /// The first broken clause of the contract.
    pub fn open(dir: &Path) -> Result<Self, ArtifactError> {
        Self::HASHED
            .iter()
            .chain([&Self::MANIFEST])
            .map(|name| dir.join(name))
            .find(|path| !path.is_file())
            .map_or(Ok(()), |path| Err(ArtifactError::Missing(path)))?;
        let config: ConfigArtifact = Self::open_json(&dir.join(Self::CONFIG))?;
        let manifest: ManifestArtifact = Self::open_json(&dir.join(Self::MANIFEST))?;
        Self::open_supported("words_splitter_type", &config.words_splitter_type, "whitespace")?;
        Self::open_supported("subtoken_pooling", &config.subtoken_pooling, "first")?;
        Self::HASHED
            .into_iter()
            .find(|name| !manifest.files.contains_key(*name))
            .map_or(Ok(()), |name| Err(ArtifactError::Unlisted(name)))?;
        Ok(Self {
            dir: dir.to_owned(),
            config,
            manifest,
        })
    }

    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// Hashes every file against the manifest; the version is the first 16 hex digits of the sha256 of their
    /// sha256s, so it changes with any byte of the model and nothing else.
    ///
    /// # Errors
    /// A file is unreadable or differs from the manifest.
    pub fn verify(&self) -> Result<String, ArtifactError> {
        let digests = Self::HASHED
            .into_iter()
            .map(|name| self.verify_file(name).map(|digest| format!("{name} {digest}\n")))
            .collect::<Result<String, _>>()?;
        Ok(format!("{:x}", Sha256::digest(digests.as_bytes()))
            .chars()
            .take(16)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::runtime::artifact::{ArtifactError, ModelArtifact};

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/tiny")
    }

    fn copy() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::read_dir(tiny())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .for_each(|path| {
                std::fs::copy(&path, dir.path().join(path.file_name().unwrap())).unwrap();
            });
        dir
    }

    fn edit(dir: &Path, name: &str, from: &str, to: &str) {
        let path = dir.join(name);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(from), "{name} lacks {from}");
        std::fs::write(path, text.replace(from, to)).unwrap();
    }

    #[test]
    fn test_open_reads_config_and_manifest() {
        let artifact = ModelArtifact::open(&tiny()).unwrap();
        assert_eq!((artifact.config.max_width, artifact.config.max_len), (12, 6));
        assert_eq!(artifact.manifest.model, "tiny");
        assert_eq!(artifact.path("text.onnx"), tiny().join("text.onnx"));
    }

    #[test]
    fn test_verify_is_a_stable_version() {
        let artifact = ModelArtifact::open(&tiny()).unwrap();
        let version = artifact.verify().unwrap();
        assert_eq!(version.len(), 16);
        assert_eq!(artifact.verify().unwrap(), version);
    }

    #[test]
    fn test_open_rejects_missing_files() {
        let dir = copy();
        std::fs::remove_file(dir.path().join("labels.tokenizer.json")).unwrap();
        assert!(
            matches!(ModelArtifact::open(dir.path()), Err(ArtifactError::Missing(path)) if path.ends_with("labels.tokenizer.json"))
        );
        assert!(matches!(
            ModelArtifact::open(Path::new("/nonexistent")),
            Err(ArtifactError::Missing(_))
        ));
    }

    #[test]
    fn test_open_rejects_unsupported_and_unlisted() {
        let dir = copy();
        edit(dir.path(), "config.json", "\"whitespace\"", "\"spacy\"");
        assert!(matches!(
            ModelArtifact::open(dir.path()),
            Err(ArtifactError::Unsupported {
                key: "words_splitter_type",
                ..
            })
        ));
        let dir = copy();
        edit(dir.path(), "manifest.json", "\"labels.onnx\"", "\"other.onnx\"");
        assert!(matches!(
            ModelArtifact::open(dir.path()),
            Err(ArtifactError::Unlisted("labels.onnx"))
        ));
        let dir = copy();
        edit(dir.path(), "manifest.json", "\"files\"", "\"filez\"");
        assert!(matches!(
            ModelArtifact::open(dir.path()),
            Err(ArtifactError::Parse { .. })
        ));
    }

    #[test]
    fn test_verify_rejects_tampered_files() {
        let dir = copy();
        edit(dir.path(), "config.json", "\"max_len\": 6", "\"max_len\": 7");
        let artifact = ModelArtifact::open(dir.path()).unwrap();
        assert!(matches!(
            artifact.verify(),
            Err(ArtifactError::Digest {
                file: "config.json",
                ..
            })
        ));
    }
}
