use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use crate::schema::error::RecognizerError;

/// The text encoder's inputs for one window of words (batch of one), as GLiNER's processor builds them:
/// words tokenized pre-split with special tokens, the first sub-token of each word marking it (1-based) in
/// `words_mask`, and every span of up to `width` words enumerated start-major with its validity in `span_mask`.
#[derive(Debug, Clone, PartialEq)]
pub struct TextTensor {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub words_mask: Array2<i64>,
    pub text_lengths: Array2<i64>,
    pub span_idx: Array3<i64>,
    pub span_mask: Array2<bool>,
}

impl TextTensor {
    // ##### PRIVATE #####

    fn encode_index(value: usize) -> Result<i64, RecognizerError> {
        i64::try_from(value).map_err(|error| RecognizerError::Shape(error.to_string()))
    }

    fn encode_row(values: Vec<i64>) -> Result<Array2<i64>, RecognizerError> {
        Array2::from_shape_vec((1, values.len()), values).map_err(|error| RecognizerError::Shape(error.to_string()))
    }

    fn encode_words(word_ids: &[Option<u32>]) -> Vec<i64> {
        word_ids
            .iter()
            .scan(None, |last, &word| {
                let first = word.filter(|_| *last != word);
                *last = word;
                Some(first.map_or(0, |index| i64::from(index).saturating_add(1)))
            })
            .collect()
    }

    fn encode_spans(words: usize, width: usize) -> Result<(Array3<i64>, Array2<bool>), RecognizerError> {
        let spans = words.saturating_mul(width);
        let (pairs, mask): (Vec<[usize; 2]>, Vec<bool>) = (0..words)
            .flat_map(|start| (0..width).map(move |offset| (start, start.saturating_add(offset))))
            .map(|(start, end)| {
                if end < words {
                    ([start, end], true)
                } else {
                    ([0, 0], false)
                }
            })
            .unzip();
        let flat = pairs
            .into_iter()
            .flatten()
            .map(Self::encode_index)
            .collect::<Result<Vec<_>, _>>()?;
        let shape = |error: ndarray::ShapeError| RecognizerError::Shape(error.to_string());
        Ok((
            Array3::from_shape_vec((1, spans, 2), flat).map_err(shape)?,
            Array2::from_shape_vec((1, spans), mask).map_err(shape)?,
        ))
    }

    // ----------------------------------------------------------

    // ##### PUBLIC #####

    /// # Errors
    /// The tokenizer failed, or a size does not fit the tensor types.
    pub fn encode(tokenizer: &Tokenizer, words: &[&str], width: usize) -> Result<Self, RecognizerError> {
        let encoding = tokenizer
            .encode(words, true)
            .map_err(|error| RecognizerError::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids().iter().map(|&id| i64::from(id)).collect::<Vec<_>>();
        let attention = vec![1; ids.len()];
        let (span_idx, span_mask) = Self::encode_spans(words.len(), width)?;
        Ok(Self {
            words_mask: Self::encode_row(Self::encode_words(encoding.get_word_ids()))?,
            input_ids: Self::encode_row(ids)?,
            attention_mask: Self::encode_row(attention)?,
            text_lengths: Self::encode_row(vec![Self::encode_index(words.len())?])?,
            span_idx,
            span_mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tokenizers::Tokenizer;

    use crate::core::tensor::TextTensor;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/tiny/tokenizer.json")).unwrap()
    }

    #[test]
    fn test_encode_marks_first_subtoken_of_each_word() {
        let tensor = TextTensor::encode(&tokenizer(), &["person", "software tool", "zzz"], 2).unwrap();
        assert_eq!(tensor.input_ids.row(0).to_vec(), [1, 4, 8, 7, 0, 2]);
        assert_eq!(tensor.attention_mask.row(0).to_vec(), [1; 6]);
        assert_eq!(tensor.words_mask.row(0).to_vec(), [0, 1, 2, 0, 3, 0]);
        assert_eq!(tensor.text_lengths.row(0).to_vec(), [3]);
    }

    #[test]
    fn test_encode_enumerates_spans_start_major() {
        let tensor = TextTensor::encode(&tokenizer(), &["a", "b", "c"], 2).unwrap();
        assert_eq!(tensor.span_idx.shape(), [1, 6, 2]);
        let pairs: Vec<(i64, i64)> = tensor
            .span_idx
            .outer_iter()
            .flat_map(|batch| batch.outer_iter().map(|pair| (pair[0], pair[1])).collect::<Vec<_>>())
            .collect();
        assert_eq!(pairs, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (0, 0)]);
        assert_eq!(tensor.span_mask.row(0).to_vec(), [true, true, true, true, true, false]);
    }
}
