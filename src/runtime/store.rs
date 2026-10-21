use std::io;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader, BufWriter};
use tokio_util::codec::{FramedRead, LinesCodec, LinesCodecError};

use crate::core::example::Example;
use crate::schema::dataset::{CardDataset, LineDataset};

/// What a stored set is for: training (`datasets/`) or the regression base (`evals/`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindStore {
    Datasets,
    Evals,
}

impl KindStore {
    #[must_use]
    pub const fn dir(self) -> &'static str {
        match self {
            Self::Datasets => "datasets",
            Self::Evals => "evals",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("name {0:?}: 1 to 64 of A-Z a-z 0-9 . _ -, starting with a letter or a digit")]
    Name(String),
    #[error("line {line}: {reason}")]
    Line { line: usize, reason: String },
    #[error("line {line} exceeds {max} bytes")]
    Long { line: usize, max: usize },
    #[error("no examples: an empty set is refused")]
    Empty,
    #[error("{kind} {name} does not exist")]
    Missing { kind: &'static str, name: String },
    #[error("cannot read the body: {0}")]
    Body(String),
    #[error("cannot store {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
}

/// Datasets and eval sets as `<root>/<kind>/<name>.jsonl`: uploads streamed and checked line by line into a
/// temporary file renamed into place only when every line holds, so a set is replaced whole or not at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataStore {
    root: PathBuf,
    line: usize,
}

impl DataStore {
    // ##### PRIVATE #####

    fn path_valid(name: &str) -> bool {
        let mut chars = name.chars();
        name.len() <= 64
            && chars.next().is_some_and(|first| first.is_ascii_alphanumeric())
            && chars.all(|next| next.is_ascii_alphanumeric() || "._-".contains(next))
    }

    fn common_io(path: &Path) -> impl Fn(io::Error) -> StoreError + '_ {
        move |source| StoreError::Io {
            path: path.to_owned(),
            source,
        }
    }

    fn common_line(&self, number: usize, error: LinesCodecError) -> StoreError {
        match error {
            LinesCodecError::MaxLineLengthExceeded => StoreError::Long {
                line: number,
                max: self.line,
            },
            LinesCodecError::Io(error) => StoreError::Body(error.to_string()),
        }
    }

    async fn common_open(&self, kind: KindStore, name: &str) -> Result<(PathBuf, File), StoreError> {
        let path = self.path(kind, name)?;
        match File::open(&path).await {
            Ok(file) => Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Err(StoreError::Missing {
                kind: kind.dir(),
                name: name.to_owned(),
            }),
            Err(error) => Err(Self::common_io(&path)(error)),
        }
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// Creates `datasets/` and `evals/` under `root`; lines longer than `line` bytes are refused.
    ///
    /// # Errors
    /// The directories cannot be created.
    pub fn open(root: &Path, line: usize) -> Result<Self, StoreError> {
        [KindStore::Datasets, KindStore::Evals]
            .map(|kind| root.join(kind.dir()))
            .iter()
            .try_for_each(|dir| std::fs::create_dir_all(dir).map_err(Self::common_io(dir)))?;
        Ok(Self {
            root: root.to_owned(),
            line,
        })
    }

    /// # Errors
    /// An invalid name.
    pub fn path(&self, kind: KindStore, name: &str) -> Result<PathBuf, StoreError> {
        Self::path_valid(name)
            .then(|| self.root.join(kind.dir()).join(format!("{name}.jsonl")))
            .ok_or_else(|| StoreError::Name(name.to_owned()))
    }

    /// Every line checked as it arrives and written to a temporary file beside the target, which replaces it
    /// only once the body ended with every line valid; blank lines are dropped.
    ///
    /// # Errors
    /// An invalid name, an unreadable body, a bad or overlong line, no examples, or a write failure.
    pub async fn put<R: AsyncRead + Unpin>(
        &self,
        kind: KindStore,
        name: &str,
        body: R,
    ) -> Result<CardDataset, StoreError> {
        let path = self.path(kind, name)?;
        let io = Self::common_io(&path);
        let staged = NamedTempFile::new_in(path.parent().unwrap_or(&self.root)).map_err(&io)?;
        let mut writer = BufWriter::new(File::from_std(staged.reopen().map_err(&io)?));
        let mut framed = FramedRead::new(body, LinesCodec::new_with_max_length(self.line)).enumerate();
        let (mut lines, mut hasher) = (0_usize, Sha256::new());
        while let Some((index, line)) = framed.next().await {
            let number = index.saturating_add(1);
            let line = line.map_err(|error| self.common_line(number, error))?;
            if line.trim().is_empty() {
                continue;
            }
            Example::parse(&line).map_err(|error| StoreError::Line {
                line: number,
                reason: error.to_string(),
            })?;
            let record = [line.as_bytes(), b"\n"].concat();
            writer.write_all(&record).await.map_err(&io)?;
            hasher.update(&record);
            lines = lines.saturating_add(1);
        }
        (lines > 0).then_some(()).ok_or(StoreError::Empty)?;
        writer.flush().await.map_err(&io)?;
        writer.into_inner().sync_all().await.map_err(&io)?;
        staged.persist(&path).map_err(|error| io(error.error))?;
        tracing::info!(kind = kind.dir(), name, lines, "store.put");
        Ok(CardDataset {
            name: name.to_owned(),
            lines,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }

    /// The set's lines and the sha256 of its file, read in one streamed pass.
    ///
    /// # Errors
    /// An invalid name, a missing set, or a read failure.
    pub async fn card(&self, kind: KindStore, name: &str) -> Result<CardDataset, StoreError> {
        let (path, file) = self.common_open(kind, name).await?;
        let io = Self::common_io(&path);
        let mut reader = BufReader::new(file);
        let (mut lines, mut hasher) = (0_usize, Sha256::new());
        loop {
            let chunk = reader.fill_buf().await.map_err(&io)?;
            if chunk.is_empty() {
                break;
            }
            hasher.update(chunk);
            lines = lines.saturating_add(memchr::memchr_iter(b'\n', chunk).count());
            let read = chunk.len();
            reader.consume(read);
        }
        Ok(CardDataset {
            name: name.to_owned(),
            lines,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }

    /// Every set of a kind, by name.
    ///
    /// # Errors
    /// A read failure.
    pub async fn list(&self, kind: KindStore) -> Result<Vec<CardDataset>, StoreError> {
        let dir = self.root.join(kind.dir());
        let io = Self::common_io(&dir);
        let mut entries = tokio::fs::read_dir(&dir).await.map_err(&io)?;
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(&io)? {
            let file = entry.file_name();
            let stem = file.to_str().and_then(|file| file.strip_suffix(".jsonl"));
            names.extend(stem.filter(|stem| Self::path_valid(stem)).map(str::to_owned));
        }
        names.sort();
        let mut cards = Vec::with_capacity(names.len());
        for name in names {
            cards.push(self.card(kind, &name).await?);
        }
        Ok(cards)
    }

    /// Every line of a set, mapped as it is read and collected into `C`: a caller keeps only what it needs.
    ///
    /// # Errors
    /// An invalid name, a missing set, a read failure, or a line that no longer parses.
    pub async fn read<T, C: Default + Extend<T>>(
        &self,
        kind: KindStore,
        name: &str,
        mut map: impl FnMut(LineDataset) -> T,
    ) -> Result<C, StoreError> {
        let (_, file) = self.common_open(kind, name).await?;
        let mut framed = FramedRead::new(file, LinesCodec::new_with_max_length(self.line)).enumerate();
        let mut collected = C::default();
        while let Some((index, line)) = framed.next().await {
            let number = index.saturating_add(1);
            let line = line.map_err(|error| self.common_line(number, error))?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed = Example::parse(&line).map_err(|error| StoreError::Line {
                line: number,
                reason: error.to_string(),
            })?;
            collected.extend(std::iter::once(map(parsed)));
        }
        Ok(collected)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use sha2::{Digest, Sha256};

    use crate::core::example::Example;
    use crate::runtime::store::{DataStore, KindStore, StoreError};
    use crate::schema::dataset::LineDataset;

    const GOOD: &str = concat!(
        r#"{"text": "Ana vive en Málaga", "entities": [{"start": 0, "end": 3, "label": "person"}]}"#,
        "\n\n",
        r#"{"text": "nada", "entities": []}"#,
        "\n"
    );

    fn store() -> (tempfile::TempDir, DataStore) {
        let root = tempfile::tempdir().unwrap();
        let store = DataStore::open(&root.path().join("train"), 1024).unwrap();
        (root, store)
    }

    fn files(store: &DataStore, kind: KindStore) -> Vec<String> {
        let dir = store.path(kind, "probe").unwrap().parent().unwrap().to_owned();
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn test_put_stores_valid_lines_and_their_digest() {
        let (_root, store) = store();
        let card = store.put(KindStore::Evals, "news", GOOD.as_bytes()).await.unwrap();
        let path = store.path(KindStore::Evals, "news").unwrap();
        let stored = std::fs::read_to_string(&path).unwrap();
        assert_eq!(stored.lines().count(), 2, "blank lines dropped");
        assert_eq!((card.name.as_str(), card.lines), ("news", 2));
        assert_eq!(card.sha256, format!("{:x}", Sha256::digest(stored.as_bytes())));
        assert_eq!(store.card(KindStore::Evals, "news").await.unwrap(), card);
        assert!(files(&store, KindStore::Datasets).is_empty());
    }

    #[tokio::test]
    async fn test_put_refuses_a_bad_line_and_keeps_the_old_set() {
        let (_root, store) = store();
        let old = store.put(KindStore::Datasets, "web", GOOD.as_bytes()).await.unwrap();
        let bad = format!(
            "{GOOD}{}\n",
            r#"{"text": "Ana", "entities": [{"start": 0, "end": 9, "label": "x"}]}"#
        );
        let refused = store.put(KindStore::Datasets, "web", bad.as_bytes()).await;
        assert!(
            matches!(&refused, Err(StoreError::Line { line: 4, reason }) if reason.contains("past the text")),
            "{refused:?}"
        );
        assert_eq!(store.card(KindStore::Datasets, "web").await.unwrap(), old);
        assert_eq!(files(&store, KindStore::Datasets), ["web.jsonl"]);
    }

    #[tokio::test]
    async fn test_put_refuses_empty_overlong_and_misnamed() {
        let (_root, store) = store();
        assert!(matches!(
            store.put(KindStore::Evals, "e", "\n \n".as_bytes()).await,
            Err(StoreError::Empty)
        ));
        let long = format!("{GOOD}{}\n", "x".repeat(2000));
        assert!(matches!(
            store.put(KindStore::Evals, "e", long.as_bytes()).await,
            Err(StoreError::Long { line: 4, max: 1024 })
        ));
        for name in ["", "../up", ".hidden", "a/b", "a b", &"n".repeat(65)] {
            assert!(
                matches!(
                    store.put(KindStore::Evals, name, GOOD.as_bytes()).await,
                    Err(StoreError::Name(_))
                ),
                "{name}"
            );
        }
        assert!(store.put(KindStore::Evals, "v1.2_x-y", GOOD.as_bytes()).await.is_ok());
        assert_eq!(files(&store, KindStore::Evals), ["v1.2_x-y.jsonl"]);
    }

    #[tokio::test]
    async fn test_list_and_read() {
        let (_root, store) = store();
        assert!(store.list(KindStore::Evals).await.unwrap().is_empty());
        store.put(KindStore::Evals, "b", GOOD.as_bytes()).await.unwrap();
        store.put(KindStore::Evals, "a", GOOD.as_bytes()).await.unwrap();
        store.put(KindStore::Datasets, "c", GOOD.as_bytes()).await.unwrap();
        let names: Vec<String> = store
            .list(KindStore::Evals)
            .await
            .unwrap()
            .into_iter()
            .map(|card| card.name)
            .collect();
        assert_eq!(names, ["a", "b"]);
        let lines: Vec<LineDataset> = store.read(KindStore::Evals, "a", |line| line).await.unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].entities[0].label, "person");
        let prints: HashSet<[u8; 32]> = store
            .read(KindStore::Datasets, "c", |line| Example::fingerprint(&line.text))
            .await
            .unwrap();
        assert!(prints.contains(&Example::fingerprint("ana vive en málaga")));
        assert!(matches!(
            store.card(KindStore::Datasets, "a").await,
            Err(StoreError::Missing { kind: "datasets", .. })
        ));
        assert!(matches!(
            store
                .read::<LineDataset, Vec<_>>(KindStore::Evals, "zz", |line| line)
                .await,
            Err(StoreError::Missing { .. })
        ));
    }

    #[tokio::test]
    async fn test_read_reports_a_corrupted_file() {
        let (_root, store) = store();
        store.put(KindStore::Evals, "a", GOOD.as_bytes()).await.unwrap();
        std::fs::write(store.path(KindStore::Evals, "a").unwrap(), "{broken\n").unwrap();
        assert!(matches!(
            store
                .read::<LineDataset, Vec<_>>(KindStore::Evals, "a", |line| line)
                .await,
            Err(StoreError::Line { line: 1, .. })
        ));
    }
}
