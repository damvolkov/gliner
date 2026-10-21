use std::ops::Range;

/// A run of consecutive words the encoder sees at once: `start..end` over the text's words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: usize,
    pub end: usize,
}

impl Window {
    /// Windows of at most `size` words covering `words`, consecutive ones sharing `overlap` words.
    /// `overlap < size` is the caller's contract; a larger one degrades to a stride of one word.
    #[must_use]
    pub fn plan(words: usize, size: usize, overlap: usize) -> Vec<Self> {
        let stride = size.saturating_sub(overlap).max(1);
        let first = Self {
            start: 0,
            end: size.min(words),
        };
        std::iter::successors((words > 0).then_some(first), |last| {
            let start = last.start.saturating_add(stride);
            (last.end < words).then(|| Self {
                start,
                end: start.saturating_add(size).min(words),
            })
        })
        .collect()
    }

    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

#[cfg(test)]
mod tests {
    use crate::core::window::Window;

    fn plan(words: usize, size: usize, overlap: usize) -> Vec<(usize, usize)> {
        Window::plan(words, size, overlap)
            .into_iter()
            .map(|window| (window.start, window.end))
            .collect()
    }

    #[test]
    fn test_plan_short_text_is_one_window() {
        assert_eq!(plan(5, 8, 2), [(0, 5)]);
        assert_eq!(plan(8, 8, 2), [(0, 8)]);
    }

    #[test]
    fn test_plan_long_text_overlaps_and_ends_flush() {
        assert_eq!(plan(20, 8, 2), [(0, 8), (6, 14), (12, 20)]);
        assert_eq!(plan(21, 8, 2), [(0, 8), (6, 14), (12, 20), (18, 21)]);
        assert_eq!(plan(9, 8, 0), [(0, 8), (8, 9)]);
    }

    #[test]
    fn test_plan_degenerate_inputs() {
        assert!(plan(0, 8, 2).is_empty());
        assert_eq!(plan(3, 2, 5), [(0, 2), (1, 3)]);
    }
}
