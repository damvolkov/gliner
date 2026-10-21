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
