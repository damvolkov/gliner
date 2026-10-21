use std::ops::Range;

/// One word of a text, located twice: `bytes` slices the `&str`, `chars` is what clients see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    pub bytes: Range<usize>,
    pub chars: Range<usize>,
}

impl Word {
    /// The word's text, empty if `text` is not the one it was split from.
    #[must_use]
    pub fn of<'t>(&self, text: &'t str) -> &'t str {
        text.get(self.bytes.clone()).unwrap_or_default()
    }
}
