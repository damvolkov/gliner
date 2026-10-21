use std::convert::Infallible;
use std::io;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use bytes::BytesMut;
use futures_util::{Stream, StreamExt, TryStreamExt, future};
use tokio_util::codec::{Decoder, FramedRead, LinesCodec, LinesCodecError};
use tokio_util::io::StreamReader;
use tokio_util::sync::CancellationToken;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::core::query::Query;
use crate::runtime::executor::BoundedExecutor;
use crate::schema::batch::{LineBatch, ReplyBatch};

pub const NDJSON: &str = "application/x-ndjson";

/// One item to predict, its query resolved against the stream's defaults.
#[derive(Debug)]
pub struct JobBatch {
    pub id: String,
    pub text: String,
    pub query: Query,
}

impl JobBatch {
    pub async fn run(self, executor: BoundedExecutor) -> ReplyBatch {
        match executor.predict(self.text, self.query).await {
            Ok(entities) => ReplyBatch::Entities { id: self.id, entities },
            Err(error) => {
                tracing::warn!(id = %self.id, %error, "batch.failed");
                ReplyBatch::Failed {
                    id: self.id,
                    error: error.to_string(),
                }
            }
        }
    }
}

/// Lines up to a byte limit, an overlong one surfacing as `None` (and skipped) rather than as an error:
/// `FramedRead` ends the stream at the first decoder error, and one bad line must not end a batch.
#[derive(Debug)]
pub struct CodecBatch(LinesCodec);

impl Decoder for CodecBatch {
    type Item = Option<String>;
    type Error = LinesCodecError;

    fn decode(&mut self, source: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.0.decode(source) {
            Err(LinesCodecError::MaxLineLengthExceeded) => Ok(Some(None)),
            decoded => decoded.map(|line| line.map(Some)),
        }
    }

    fn decode_eof(&mut self, source: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.0.decode_eof(source) {
            Err(LinesCodecError::MaxLineLengthExceeded) => Ok(Some(None)),
            decoded => decoded.map(|line| line.map(Some)),
        }
    }
}

/// The NDJSON pipeline of one `/ner/batch` request: lines framed as they arrive, at most `concurrency` items
/// in flight (the body is read no faster than items complete), replies written as each finishes. Dropping the
/// stream (client gone) drops every pending item; shutdown ends it after the items in flight.
#[derive(Debug)]
pub struct StreamBatch {
    executor: BoundedExecutor,
    defaults: Query,
    concurrency: usize,
    line: usize,
    shutdown: CancellationToken,
}

impl StreamBatch {
    // ##### PRIVATE #####

    fn run_parse(
        state: &mut (Query, bool),
        number: usize,
        line: Result<Option<String>, LinesCodecError>,
        max: usize,
    ) -> Result<Option<JobBatch>, ReplyBatch> {
        let invalid = |error: String| ReplyBatch::Invalid { line: number, error };
        let line = line
            .map_err(|error| invalid(format!("cannot read the body: {error}")))?
            .ok_or_else(|| invalid(format!("line exceeds {max} bytes")))?;
        if line.trim().is_empty() {
            return Ok(None);
        }
        let parsed: LineBatch = serde_json::from_str(&line).map_err(|error| invalid(error.to_string()))?;
        let (defaults, started) = state;
        let header = !std::mem::replace(started, true);
        let labels = parsed.labels.map_or_else(|| Arc::clone(&defaults.labels), Arc::from);
        let threshold = parsed.threshold.unwrap_or(defaults.threshold);
        match (parsed.id, parsed.text, header) {
            (None, None, true) => {
                *defaults = Query { labels, threshold };
                Ok(None)
            }
            (None, None, false) => Err(invalid("a header (no id, no text) must precede every item".to_owned())),
            (None, Some(_), _) => Err(invalid("missing id".to_owned())),
            (Some(id), None, _) => Err(ReplyBatch::Failed {
                id,
                error: "missing text".to_owned(),
            }),
            (Some(id), Some(text), _) => match Query::new(labels, threshold) {
                Ok(query) => Ok(Some(JobBatch { id, text, query })),
                Err(error) => Err(ReplyBatch::Failed {
                    id,
                    error: error.to_string(),
                }),
            },
        }
    }

    fn run_encode(reply: &ReplyBatch) -> Bytes {
        let mut line =
            serde_json::to_vec(reply).unwrap_or_else(|error| format!(r#"{{"error":"{error}"}}"#).into_bytes());
        line.push(b'\n');
        Bytes::from(line)
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(executor: BoundedExecutor, state: &AppState) -> Self {
        Self {
            executor,
            defaults: state.defaults.clone(),
            concurrency: state.concurrency,
            line: state.line,
            shutdown: state.shutdown.clone(),
        }
    }

    /// Response lines for a request body.
    pub fn run(self, body: Body) -> impl Stream<Item = Bytes> + Send + 'static {
        let reader = StreamReader::new(body.into_data_stream().map_err(io::Error::other));
        let (max, executor) = (self.line, self.executor);
        FramedRead::new(reader, CodecBatch(LinesCodec::new_with_max_length(max)))
            .enumerate()
            .scan((self.defaults, false), move |state, (index, line)| {
                future::ready(Some(
                    Self::run_parse(state, index.saturating_add(1), line, max).transpose(),
                ))
            })
            .filter_map(future::ready)
            .map(move |parsed| {
                let executor = executor.clone();
                async move {
                    match parsed {
                        Ok(job) => job.run(executor).await,
                        Err(reply) => reply,
                    }
                }
            })
            .buffer_unordered(self.concurrency.max(1))
            .take_until(self.shutdown.cancelled_owned())
            .map(|reply| Self::run_encode(&reply))
    }
}

/// `POST /ner/batch`: NDJSON in, NDJSON out, streamed both ways.
///
/// # Errors
/// The model is not loaded yet (503).
pub async fn batch(State(state): State<AppState>, body: Body) -> Result<Response, ApiError> {
    let stream = StreamBatch::new(state.ready()?.clone(), &state).run(body);
    Ok((
        [(CONTENT_TYPE, NDJSON)],
        Body::from_stream(stream.map(Ok::<_, Infallible>)),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use bytes::BytesMut;
    use tokio_util::codec::{Decoder, LinesCodec};

    use crate::api::batch::CodecBatch;

    #[test]
    fn test_codec_skips_overlong_lines_and_keeps_framing() {
        let mut codec = CodecBatch(LinesCodec::new_with_max_length(8));
        let mut source = BytesMut::from("short\r\nmuch too long a line\nnext\nlast");
        let mut lines = std::iter::from_fn(|| codec.decode(&mut source).unwrap()).collect::<Vec<_>>();
        lines.extend(codec.decode_eof(&mut source).unwrap());
        assert_eq!(
            lines,
            [
                Some("short".to_owned()),
                None,
                Some("next".to_owned()),
                Some("last".to_owned())
            ]
        );
    }

    #[test]
    fn test_codec_reports_an_overlong_last_line() {
        let mut codec = CodecBatch(LinesCodec::new_with_max_length(4));
        let mut source = BytesMut::from("unterminated");
        assert_eq!(codec.decode(&mut source).unwrap(), Some(None));
        assert_eq!(codec.decode_eof(&mut source).unwrap(), None);
    }
}
