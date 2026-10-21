use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use tempfile::NamedTempFile;

use crate::runtime::artifact::{ArtifactError, ModelArtifact};
use crate::schema::registry::EntryRegistry;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("no model registry at {0}: `make export` writes version 1")]
    Missing(PathBuf),
    #[error("{path} does not name a version: {reason}")]
    Pointer { path: PathBuf, reason: String },
    #[error("version {0} is not registered")]
    Unknown(u32),
    #[error("cannot read {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("cannot write {path}: {source}")]
    Write { path: PathBuf, source: io::Error },
    #[error("{path} is not a JSON object")]
    Manifest { path: PathBuf },
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// Versions of one model lineage under a root: `<root>/<n>/` holds the served artifact, `checkpoint/` (the
/// `PyTorch` weights training continues from, absent on an exported root version) and a `manifest.json` whose
/// `registry` block records the lineage; `<root>/current` names the served version and is only ever replaced
/// whole (write, then rename). Dot-dirs are a job's scratch space, never versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    // ##### PRIVATE #####

    fn common_write(dir: &Path, path: &Path, bytes: &[u8]) -> Result<(), RegistryError> {
        let write = |source| RegistryError::Write {
            path: path.to_owned(),
            source,
        };
        let mut staged = NamedTempFile::new_in(dir).map_err(write)?;
        staged.write_all(bytes).map_err(write)?;
        staged.as_file().sync_all().map_err(write)?;
        staged.persist(path).map_err(|error| write(error.error))?;
        Ok(())
    }

    fn register_manifest(candidate: &Path, entry: &EntryRegistry) -> Result<(), RegistryError> {
        let path = candidate.join(ModelArtifact::MANIFEST);
        let read = |source| RegistryError::Read {
            path: path.clone(),
            source,
        };
        let text = fs::read_to_string(&path).map_err(read)?;
        let mut manifest: Map<String, Value> =
            serde_json::from_str(&text).map_err(|_| RegistryError::Manifest { path: path.clone() })?;
        let block = serde_json::to_value(entry).map_err(|_| RegistryError::Manifest { path: path.clone() })?;
        manifest.insert("registry".to_owned(), block);
        let mut bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|_| RegistryError::Manifest { path: path.clone() })?;
        bytes.push(b'\n');
        Self::common_write(candidate, &path, &bytes)
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    pub const CURRENT: &str = "current";
    pub const CHECKPOINT: &str = "checkpoint";

    /// # Errors
    /// The root is not a directory.
    pub fn open(root: &Path) -> Result<Self, RegistryError> {
        root.is_dir()
            .then(|| Self { root: root.to_owned() })
            .ok_or_else(|| RegistryError::Missing(root.to_owned()))
    }

    #[must_use]
    pub fn path(&self, version: u32) -> PathBuf {
        self.root.join(version.to_string())
    }

    /// A job's scratch dir: on the registry's filesystem, so its candidate is registered by a rename.
    #[must_use]
    pub fn staging(&self, job: u64) -> PathBuf {
        self.root.join(format!(".job-{job}"))
    }

    /// Registered versions, ascending.
    ///
    /// # Errors
    /// The root cannot be listed.
    pub fn versions(&self) -> Result<Vec<u32>, RegistryError> {
        let read = |source| RegistryError::Read {
            path: self.root.clone(),
            source,
        };
        let mut versions = fs::read_dir(&self.root)
            .map_err(read)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
            .collect::<Vec<_>>();
        versions.sort_unstable();
        Ok(versions)
    }

    /// The version `current` names, which must be registered.
    ///
    /// # Errors
    /// `current` is missing, unreadable, not a version number, or names an unregistered version.
    pub fn current(&self) -> Result<u32, RegistryError> {
        let path = self.root.join(Self::CURRENT);
        let pointer = |reason: String| RegistryError::Pointer {
            path: path.clone(),
            reason,
        };
        let text = fs::read_to_string(&path).map_err(|error| pointer(error.to_string()))?;
        let version = text.trim().parse::<u32>().map_err(|error| pointer(error.to_string()))?;
        self.path(version)
            .is_dir()
            .then_some(version)
            .ok_or(RegistryError::Unknown(version))
    }

    /// Points `current` at a registered version, atomically: a reader sees the old pointer or the new one.
    ///
    /// # Errors
    /// The version is not registered, or the pointer cannot be written.
    pub fn point(&self, version: u32) -> Result<(), RegistryError> {
        self.path(version)
            .is_dir()
            .then_some(())
            .ok_or(RegistryError::Unknown(version))?;
        Self::common_write(
            &self.root,
            &self.root.join(Self::CURRENT),
            format!("{version}\n").as_bytes(),
        )
    }

    /// The model a version's training starts from: its checkpoint dir, or for a root version the checkpoint id
    /// it was exported from (its manifest's `model`).
    ///
    /// # Errors
    /// The version is not registered or its manifest is unreadable.
    pub fn source(&self, version: u32) -> Result<String, RegistryError> {
        let dir = self.path(version);
        dir.is_dir().then_some(()).ok_or(RegistryError::Unknown(version))?;
        let checkpoint = dir.join(Self::CHECKPOINT);
        if checkpoint.is_dir() {
            Ok(checkpoint.to_string_lossy().into_owned())
        } else {
            Ok(ModelArtifact::open(&dir)?.manifest.model)
        }
    }

    /// Registers a candidate dir as the next version: its manifest gains the `registry` block, then the dir is
    /// renamed into place, so a version appears whole or not at all.
    ///
    /// # Errors
    /// The candidate's manifest cannot be rewritten, or the dir cannot be moved.
    pub fn register(&self, candidate: &Path, entry: EntryRegistry) -> Result<u32, RegistryError> {
        let version = self.versions()?.last().map_or(1, |last| last.saturating_add(1));
        Self::register_manifest(candidate, &EntryRegistry { version, ..entry })?;
        let target = self.path(version);
        fs::rename(candidate, &target).map_err(|source| RegistryError::Write { path: target, source })?;
        Ok(version)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    use crate::runtime::registry::{ModelRegistry, RegistryError};
    use crate::schema::registry::EntryRegistry;

    fn tiny() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/tiny")
    }

    fn copy(to: &Path) {
        fs::create_dir_all(to).unwrap();
        fs::read_dir(tiny())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .for_each(|path| {
                fs::copy(&path, to.join(path.file_name().unwrap())).unwrap();
            });
    }

    /// A registry holding the tiny artifact as version 1, served.
    fn registry() -> (tempfile::TempDir, ModelRegistry) {
        let root = tempfile::tempdir().unwrap();
        copy(&root.path().join("1"));
        fs::write(root.path().join("current"), "1\n").unwrap();
        let registry = ModelRegistry::open(root.path()).unwrap();
        (root, registry)
    }

    fn entry() -> EntryRegistry {
        EntryRegistry {
            version: 0,
            parent: Some(1),
            base: "tiny".to_owned(),
            dataset: None,
            params: None,
            created: 1,
            report: None,
        }
    }

    #[test]
    fn test_open_requires_a_dir() {
        assert!(matches!(
            ModelRegistry::open(Path::new("/nonexistent")),
            Err(RegistryError::Missing(_))
        ));
    }

    #[test]
    fn test_current_reads_and_point_replaces_the_pointer() {
        let (root, registry) = registry();
        assert_eq!(registry.current().unwrap(), 1);
        copy(&root.path().join("2"));
        fs::create_dir(root.path().join(".job-7")).unwrap();
        fs::create_dir(root.path().join("notes")).unwrap();
        assert_eq!(registry.versions().unwrap(), [1, 2]);
        registry.point(2).unwrap();
        assert_eq!(registry.current().unwrap(), 2);
        assert_eq!(fs::read_to_string(root.path().join("current")).unwrap(), "2\n");
        assert!(matches!(registry.point(9), Err(RegistryError::Unknown(9))));
        assert_eq!(registry.current().unwrap(), 2);
        let leftovers = fs::read_dir(root.path())
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().path().is_file())
            .count();
        assert_eq!(leftovers, 1, "only `current`, no temporary file left behind");
    }

    #[test]
    fn test_current_rejects_bad_pointers() {
        let (root, registry) = registry();
        fs::write(root.path().join("current"), "one\n").unwrap();
        assert!(matches!(registry.current(), Err(RegistryError::Pointer { .. })));
        fs::write(root.path().join("current"), "3").unwrap();
        assert!(matches!(registry.current(), Err(RegistryError::Unknown(3))));
        fs::remove_file(root.path().join("current")).unwrap();
        assert!(matches!(registry.current(), Err(RegistryError::Pointer { .. })));
    }

    #[test]
    fn test_source_prefers_the_checkpoint() {
        let (root, registry) = registry();
        assert_eq!(registry.source(1).unwrap(), "tiny");
        fs::create_dir(root.path().join("1/checkpoint")).unwrap();
        assert_eq!(
            registry.source(1).unwrap(),
            root.path().join("1/checkpoint").to_string_lossy()
        );
        assert!(matches!(registry.source(4), Err(RegistryError::Unknown(4))));
    }

    #[test]
    fn test_register_numbers_annotates_and_moves() {
        let (root, registry) = registry();
        let candidate = registry.staging(3).join("candidate");
        copy(&candidate);
        assert_eq!(registry.register(&candidate, entry()).unwrap(), 2);
        assert!(!candidate.exists());
        assert_eq!(registry.current().unwrap(), 1);
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(root.path().join("2/manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["model"], "tiny");
        assert_eq!(manifest["registry"]["version"], 2);
        assert_eq!(manifest["registry"]["parent"], 1);
        assert!(
            crate::runtime::artifact::ModelArtifact::open(&root.path().join("2"))
                .unwrap()
                .verify()
                .is_ok()
        );
    }

    #[test]
    fn test_register_refuses_a_broken_candidate() {
        let (_root, registry) = registry();
        let candidate = registry.staging(4).join("candidate");
        fs::create_dir_all(&candidate).unwrap();
        assert!(matches!(
            registry.register(&candidate, entry()),
            Err(RegistryError::Read { .. })
        ));
        fs::write(candidate.join("manifest.json"), "[]").unwrap();
        assert!(matches!(
            registry.register(&candidate, entry()),
            Err(RegistryError::Manifest { .. })
        ));
        assert_eq!(registry.versions().unwrap(), [1]);
    }
}
