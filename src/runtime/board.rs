use std::collections::{BTreeMap, VecDeque};
use std::sync::{Mutex, MutexGuard, PoisonError};

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::schema::train::{JobTrain, StateTrain};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoardError {
    #[error("job {0} is running: one job at a time")]
    Busy(u64),
    #[error("job {0} does not exist")]
    Unknown(u64),
    #[error("job {0} is not running")]
    Finished(u64),
}

#[derive(Debug)]
struct EntryBoard {
    job: JobTrain,
    log: VecDeque<String>,
}

#[derive(Debug)]
struct RunningBoard {
    id: u64,
    cancel: CancellationToken,
    task: Option<JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct StateBoard {
    jobs: BTreeMap<u64, EntryBoard>,
    running: Option<RunningBoard>,
    last: u64,
}

/// The train jobs of this process, at most one running: each job's record and the tail of its log. Kept in
/// memory: a restart forgets jobs, never what they registered.
#[derive(Debug, Default)]
pub struct JobBoard {
    state: Mutex<StateBoard>,
}

impl JobBoard {
    // ##### PRIVATE #####

    fn common_lock(&self) -> MutexGuard<'_, StateBoard> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn common_snapshot(entry: &EntryBoard) -> JobTrain {
        JobTrain {
            log: entry.log.iter().cloned().collect(),
            ..entry.job.clone()
        }
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    pub const TAIL: usize = 200;

    /// Opens a job if none runs: `make` receives its id.
    ///
    /// # Errors
    /// Another job is running.
    pub fn claim(&self, make: impl FnOnce(u64) -> JobTrain) -> Result<(JobTrain, CancellationToken), BoardError> {
        let mut state = self.common_lock();
        if let Some(running) = &state.running {
            return Err(BoardError::Busy(running.id));
        }
        let id = state.last.saturating_add(1);
        let cancel = CancellationToken::new();
        let job = make(id);
        state.last = id;
        state.running = Some(RunningBoard {
            id,
            cancel: cancel.clone(),
            task: None,
        });
        state.jobs.insert(
            id,
            EntryBoard {
                job: job.clone(),
                log: VecDeque::new(),
            },
        );
        Ok((job, cancel))
    }

    /// Hands the running job its task, so a cancel can wait for it.
    pub fn attach(&self, id: u64, task: JoinHandle<()>) {
        let mut state = self.common_lock();
        if let Some(running) = state.running.as_mut().filter(|running| running.id == id) {
            running.task = Some(task);
        }
    }

    pub fn update(&self, id: u64, change: impl FnOnce(&mut JobTrain)) {
        if let Some(entry) = self.common_lock().jobs.get_mut(&id) {
            change(&mut entry.job);
        }
    }

    /// Appends to a job's log, keeping the last [`Self::TAIL`] lines.
    pub fn log(&self, id: u64, line: String) {
        if let Some(entry) = self.common_lock().jobs.get_mut(&id) {
            if entry.log.len() >= Self::TAIL {
                entry.log.pop_front();
            }
            entry.log.push_back(line);
        }
    }

    /// Ends the running job in `state`, freeing the board.
    pub fn finish(&self, id: u64, state: StateTrain, error: Option<String>, at: u64) {
        let mut board = self.common_lock();
        if board.running.as_ref().is_some_and(|running| running.id == id) {
            board.running = None;
        }
        if let Some(entry) = board.jobs.get_mut(&id) {
            entry.job.state = state;
            entry.job.error = error;
            entry.job.finished = Some(at);
        }
    }

    /// # Errors
    /// No such job.
    pub fn get(&self, id: u64) -> Result<JobTrain, BoardError> {
        self.common_lock()
            .jobs
            .get(&id)
            .map(Self::common_snapshot)
            .ok_or(BoardError::Unknown(id))
    }

    #[must_use]
    pub fn list(&self) -> Vec<JobTrain> {
        self.common_lock().jobs.values().map(Self::common_snapshot).collect()
    }

    /// Cancels the running job and waits for its task to end.
    ///
    /// # Errors
    /// No such job, or it is not running.
    pub async fn cancel(&self, id: u64) -> Result<JobTrain, BoardError> {
        let task = {
            let mut state = self.common_lock();
            state
                .jobs
                .contains_key(&id)
                .then_some(())
                .ok_or(BoardError::Unknown(id))?;
            let running = state
                .running
                .as_mut()
                .filter(|running| running.id == id)
                .ok_or(BoardError::Finished(id))?;
            running.cancel.cancel();
            running.task.take()
        };
        // ### A panicked task ended all the same: its record is what remains.
        if let Some(task) = task {
            let _ = task.await;
        }
        self.get(id)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::runtime::board::{BoardError, JobBoard};
    use crate::schema::dataset::CardDataset;
    use crate::schema::train::{JobTrain, ParamsTrain, StateTrain, StepTrain};

    fn job(id: u64) -> JobTrain {
        JobTrain {
            id,
            state: StateTrain::Running,
            step: StepTrain::Leakage,
            dataset: CardDataset {
                name: "web".to_owned(),
                lines: 1,
                sha256: String::new(),
            },
            base: 1,
            params: ParamsTrain::default(),
            auto_promote: false,
            version: None,
            promoted: false,
            error: None,
            report: None,
            log: Vec::new(),
            created: 7,
            finished: None,
        }
    }

    #[test]
    fn test_claim_one_at_a_time_and_numbered() {
        let board = JobBoard::default();
        let (first, _) = board.claim(job).unwrap();
        assert_eq!(first.id, 1);
        assert_eq!(board.claim(job).unwrap_err(), BoardError::Busy(1));
        board.finish(1, StateTrain::Failed, Some("boom".to_owned()), 9);
        let ended = board.get(1).unwrap();
        assert_eq!(
            (ended.state, ended.error.as_deref(), ended.finished),
            (StateTrain::Failed, Some("boom"), Some(9))
        );
        let (second, _) = board.claim(job).unwrap();
        assert_eq!(second.id, 2);
        assert_eq!(board.list().iter().map(|job| job.id).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(board.get(3).unwrap_err(), BoardError::Unknown(3));
    }

    #[test]
    fn test_update_and_log_tail() {
        let board = JobBoard::default();
        board.claim(job).unwrap();
        board.update(1, |job| job.step = StepTrain::Export);
        (0..JobBoard::TAIL + 5).for_each(|n| board.log(1, format!("line {n}")));
        board.log(9, "nobody".to_owned());
        let seen = board.get(1).unwrap();
        assert_eq!(seen.step, StepTrain::Export);
        assert_eq!(seen.log.len(), JobBoard::TAIL);
        assert_eq!(seen.log.first().map(String::as_str), Some("line 5"));
        assert_eq!(seen.log.last().map(String::as_str), Some("line 204"));
    }

    #[tokio::test]
    async fn test_cancel_waits_for_the_task() {
        let board = Arc::new(JobBoard::default());
        let (_, cancel) = board.claim(job).unwrap();
        let task = tokio::spawn({
            let board = Arc::clone(&board);
            async move {
                cancel.cancelled().await;
                board.finish(1, StateTrain::Cancelled, None, 8);
            }
        });
        board.attach(1, task);
        let cancelled = board.cancel(1).await.unwrap();
        assert_eq!((cancelled.state, cancelled.finished), (StateTrain::Cancelled, Some(8)));
        assert_eq!(board.cancel(1).await.unwrap_err(), BoardError::Finished(1));
        assert_eq!(board.cancel(4).await.unwrap_err(), BoardError::Unknown(4));
    }
}
