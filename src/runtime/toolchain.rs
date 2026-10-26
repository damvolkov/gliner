use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::Stdio;

use futures_util::future::BoxFuture;
use futures_util::{Stream, StreamExt, stream};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;

use crate::config::train::TrainConfig;
use crate::schema::train::ParamsTrain;

/// One call of a Python tool: fine-tune `base` on a dataset into a checkpoint dir, export a checkpoint into an
/// artifact dir (its manifest recording `name`), or write `PyTorch`'s predictions on texts as a parity fixture.
#[derive(Debug, Clone, PartialEq)]
pub enum CallToolchain {
    Train {
        base: String,
        dataset: PathBuf,
        out: PathBuf,
        params: ParamsTrain,
    },
    Export {
        checkpoint: PathBuf,
        name: String,
        out: PathBuf,
    },
    Reference {
        checkpoint: PathBuf,
        texts: PathBuf,
        labels: Vec<String>,
        threshold: f32,
        out: PathBuf,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ToolchainError {
    #[error("cannot start {program}: {source}")]
    Spawn { program: String, source: io::Error },
    #[error("{tool} failed ({status}): {last}")]
    Exit {
        tool: &'static str,
        status: String,
        last: String,
    },
    #[error("{tool} output unreadable: {reason}")]
    Output { tool: &'static str, reason: String },
}

/// The port the train job drives the Python tools through. A call runs to completion, each line it prints handed
/// to `log`; dropping the future stops the tool and everything it started.
pub trait Toolchain: Send + Sync + std::fmt::Debug {
    fn run<'a>(
        &'a self,
        call: &'a CallToolchain,
        log: &'a (dyn Fn(String) + Send + Sync),
    ) -> BoxFuture<'a, Result<(), ToolchainError>>;
}

/// The tools as subprocesses: the configured command line, the call's arguments appended; stdout and stderr read
/// line by line; each run in its own process group, killed whole when the run is dropped or ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessToolchain {
    trainer: Vec<String>,
    exporter: Vec<String>,
    reference: Vec<String>,
}

impl ProcessToolchain {
    // ##### PRIVATE #####

    fn run_argv<'a>(&'a self, call: &CallToolchain) -> (&'static str, &'a [String], Vec<OsString>) {
        let path = |path: &PathBuf| path.as_os_str().to_owned();
        let text = |text: &str| OsString::from(text);
        match call {
            CallToolchain::Train {
                base,
                dataset,
                out,
                params,
            } => (
                "trainer",
                &self.trainer,
                vec![
                    text("--base"),
                    text(base),
                    text("--dataset"),
                    path(dataset),
                    text("--out"),
                    path(out),
                    text("--epochs"),
                    text(&params.epochs.to_string()),
                    text("--lr"),
                    text(&params.lr.to_string()),
                    text("--batch"),
                    text(&params.batch.to_string()),
                ],
            ),
            CallToolchain::Export { checkpoint, name, out } => (
                "exporter",
                &self.exporter,
                vec![path(checkpoint), text("--out"), path(out), text("--name"), text(name)],
            ),
            CallToolchain::Reference {
                checkpoint,
                texts,
                labels,
                threshold,
                out,
            } => (
                "reference",
                &self.reference,
                vec![
                    path(checkpoint),
                    text("--texts"),
                    path(texts),
                    text("--out"),
                    path(out),
                    text("--labels"),
                    text(&serde_json::to_string(labels).unwrap_or_default()),
                    text("--threshold"),
                    text(&threshold.to_string()),
                ],
            ),
        }
    }

    /// The lines of one pipe, each tagged with whether it came from stderr.
    fn run_lines<R: AsyncRead + Unpin>(reader: R, error: bool) -> impl Stream<Item = (bool, io::Result<String>)> {
        stream::unfold(BufReader::new(reader).lines(), move |mut lines| async move {
            lines.next_line().await.transpose().map(|line| ((error, line), lines))
        })
    }

    async fn run_process(
        tool: &'static str,
        command: &[String],
        args: Vec<OsString>,
        log: &(dyn Fn(String) + Send + Sync),
    ) -> Result<(), ToolchainError> {
        let (program, prefix) = command.split_first().ok_or_else(|| ToolchainError::Spawn {
            program: tool.to_owned(),
            source: io::Error::from(io::ErrorKind::NotFound),
        })?;
        let mut child = Command::new(program)
            .args(prefix)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| ToolchainError::Spawn {
                program: program.clone(),
                source,
            })?;
        let _group = GroupToolchain(child.id().and_then(|id| i32::try_from(id).ok()).map(Pid::from_raw));
        let unreadable = |reason: String| ToolchainError::Output { tool, reason };
        let (stdout, stderr) = child
            .stdout
            .take()
            .zip(child.stderr.take())
            .ok_or_else(|| unreadable("no pipes".to_owned()))?;
        let mut lines = std::pin::pin!(stream::select(
            Self::run_lines(stdout, false),
            Self::run_lines(stderr, true)
        ));
        // ### The pipes interleave freely; a failing tool's last word (a traceback's last line) is on stderr.
        let (mut last, mut said) = (String::new(), String::new());
        while let Some((error, line)) = lines.next().await {
            let line = line.map_err(|error| unreadable(error.to_string()))?;
            // ### Progress bars redraw with \r: the last redraw is the line.
            let shown = line.rsplit('\r').next().unwrap_or_default().trim_end();
            if !shown.is_empty() {
                shown.clone_into(if error { &mut last } else { &mut said });
                log(shown.to_owned());
            }
        }
        let status = child.wait().await.map_err(|error| unreadable(error.to_string()))?;
        status.success().then_some(()).ok_or_else(|| ToolchainError::Exit {
            tool,
            status: status.to_string(),
            last: if last.is_empty() { said } else { last },
        })
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(config: &TrainConfig) -> Self {
        Self {
            trainer: config.trainer.clone(),
            exporter: config.exporter.clone(),
            reference: config.reference.clone(),
        }
    }
}

impl Toolchain for ProcessToolchain {
    fn run<'a>(
        &'a self,
        call: &'a CallToolchain,
        log: &'a (dyn Fn(String) + Send + Sync),
    ) -> BoxFuture<'a, Result<(), ToolchainError>> {
        let (tool, command, args) = self.run_argv(call);
        Box::pin(Self::run_process(tool, command, args, log))
    }
}

/// A tool's process group, killed whole when dropped: `uv run`, Python and its workers go together.
#[derive(Debug)]
struct GroupToolchain(Option<Pid>);

impl Drop for GroupToolchain {
    fn drop(&mut self) {
        // ### An already empty group answers ESRCH: nothing left to stop.
        let _ = self.0.map(|group| killpg(group, Signal::SIGKILL));
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::config::train::TrainConfig;
    use crate::runtime::toolchain::{CallToolchain, ProcessToolchain, Toolchain, ToolchainError};
    use crate::schema::train::ParamsTrain;

    /// `sh -c script zero args…`.
    fn toolchain(script: &str, zero: &str) -> ProcessToolchain {
        let command = ["sh", "-c", script, zero].map(str::to_owned).to_vec();
        ProcessToolchain::new(&TrainConfig {
            trainer: command.clone(),
            exporter: command.clone(),
            reference: command,
            ..TrainConfig::default()
        })
    }

    fn train() -> CallToolchain {
        CallToolchain::Train {
            base: "org/model".to_owned(),
            dataset: PathBuf::from("/d/web.jsonl"),
            out: PathBuf::from("/o/checkpoint"),
            params: ParamsTrain::default(),
        }
    }

    /// Alive and not a zombie.
    fn alive(pid: &str) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
            stat.rsplit(')')
                .next()
                .is_some_and(|rest| !rest.trim_start().starts_with('Z'))
        })
    }

    #[tokio::test]
    async fn test_run_streams_both_outputs_and_passes_the_arguments() {
        let toolchain = toolchain(r#"echo "args $*"; echo "progress 1\rprogress 2" >&2; echo done"#, "sh");
        let lines = Mutex::new(Vec::new());
        let log = |line: String| lines.lock().unwrap().push(line);
        toolchain.run(&train(), &log).await.unwrap();
        let mut lines = lines.into_inner().unwrap();
        lines.sort();
        assert_eq!(
            lines,
            [
                "args --base org/model --dataset /d/web.jsonl --out /o/checkpoint --epochs 3 --lr 0.000005 --batch 8",
                "done",
                "progress 2"
            ]
        );
    }

    #[tokio::test]
    async fn test_run_arguments_of_export_and_reference() {
        let toolchain = toolchain(r#"echo "$*""#, "sh");
        let calls = [
            (
                CallToolchain::Export {
                    checkpoint: PathBuf::from("/c"),
                    name: "org/model".to_owned(),
                    out: PathBuf::from("/o"),
                },
                "/c --out /o --name org/model",
            ),
            (
                CallToolchain::Reference {
                    checkpoint: PathBuf::from("/c"),
                    texts: PathBuf::from("/t.json"),
                    labels: vec!["person".to_owned(), "software tool".to_owned()],
                    threshold: 0.5,
                    out: PathBuf::from("/r.json"),
                },
                r#"/c --texts /t.json --out /r.json --labels ["person","software tool"] --threshold 0.5"#,
            ),
        ];
        for (call, expected) in calls {
            let lines = Mutex::new(Vec::new());
            toolchain
                .run(&call, &|line| lines.lock().unwrap().push(line))
                .await
                .unwrap();
            assert_eq!(lines.into_inner().unwrap(), [expected]);
        }
    }

    #[tokio::test]
    async fn test_run_reports_failures() {
        let failed = toolchain("echo working; echo 'CUDA out of memory' >&2; exit 3", "sh")
            .run(&train(), &|_| {})
            .await;
        assert!(
            matches!(&failed, Err(ToolchainError::Exit { tool: "trainer", last, .. }) if last == "CUDA out of memory"),
            "{failed:?}"
        );
        let missing = ProcessToolchain::new(&TrainConfig {
            trainer: vec!["/nonexistent/python".to_owned()],
            ..TrainConfig::default()
        })
        .run(&train(), &|_| {})
        .await;
        assert!(matches!(missing, Err(ToolchainError::Spawn { .. })), "{missing:?}");
    }

    #[tokio::test]
    async fn test_dropping_a_run_kills_its_process_tree() {
        let dir = tempfile::tempdir().unwrap();
        let pids = dir.path().join("pids");
        let toolchain = toolchain(
            r#"sleep 30 & echo $! > "$0"; echo $$ >> "$0"; echo started; wait"#,
            pids.to_str().unwrap(),
        );
        let started = tokio::sync::Notify::new();
        let log = |line: String| {
            if line == "started" {
                started.notify_one();
            }
        };
        let call = train();
        tokio::select! {
            result = toolchain.run(&call, &log) => panic!("the tool ended on its own: {result:?}"),
            () = started.notified() => {}
        }
        let pids: Vec<String> = std::fs::read_to_string(&pids)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(pids.len(), 2);
        let gone = async {
            while pids.iter().any(|pid| alive(pid)) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(5), gone).await.unwrap();
        assert!(Path::new(dir.path()).exists());
    }
}
