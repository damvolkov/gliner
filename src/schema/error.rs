/// Why a request cannot be served as asked.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RequestError {
    #[error("labels must not be empty")]
    Empty,
    #[error("label {0} is blank")]
    Blank(usize),
    #[error("threshold {0} must be in (0, 1)")]
    Threshold(f32),
}

/// Why a prediction failed once the request was valid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecognizerError {
    #[error("tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("onnx runtime failed: {0}")]
    Onnx(String),
    #[error("unexpected tensor shape: {0}")]
    Shape(String),
    #[error("inference worker stopped: {0}")]
    Worker(String),
}

/// Why a line of a dataset or an eval set is refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExampleError {
    #[error("not a {{text, entities}} object: {0}")]
    Json(String),
    #[error("entity {index}: {start}..{end} is empty or reversed")]
    Empty { index: usize, start: usize, end: usize },
    #[error("entity {index}: end {end} is past the text ({chars} characters)")]
    Past { index: usize, end: usize, chars: usize },
    #[error("entity {0}: the label is blank")]
    Blank(usize),
}
