use regex::Regex;

use crate::core::word::Word;

/// GLiNER's `whitespace` words splitter: words (with inner `-`/`_`) and single non-space symbols.
#[derive(Debug, Clone)]
pub struct WhitespaceSplitter {
    regex: Regex,
}

impl WhitespaceSplitter {
    pub const PATTERN: &str = r"\w+(?:[-_]\w+)*|\S";

    /// # Errors
    /// Never in practice: the pattern is a constant.
    pub fn new() -> Result<Self, regex::Error> {
        Ok(Self {
            regex: Regex::new(Self::PATTERN)?,
        })
    }

    /// Words in order, with byte and character ranges.
    #[must_use]
    pub fn split(&self, text: &str) -> Vec<Word> {
        // ### One pass: characters are counted only between and inside matches, never from the start again.
        self.regex
            .find_iter(text)
            .scan((0_usize, 0_usize), |(byte, char), found| {
                let gap = text.get(*byte..found.start()).map_or(0, |gap| gap.chars().count());
                let start = (*char).saturating_add(gap);
                let end = start.saturating_add(found.as_str().chars().count());
                (*byte, *char) = (found.end(), end);
                Some(Word {
                    bytes: found.range(),
                    chars: start..end,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::splitter::WhitespaceSplitter;

    fn split(text: &str) -> Vec<(String, usize, usize)> {
        WhitespaceSplitter::new()
            .unwrap()
            .split(text)
            .into_iter()
            .map(|word| (word.of(text).to_owned(), word.chars.start, word.chars.end))
            .collect()
    }

    #[test]
    fn test_split_ascii_words_and_symbols() {
        assert_eq!(
            split("Hi, gRPC-web_v2 works."),
            [
                ("Hi".to_owned(), 0, 2),
                (",".to_owned(), 2, 3),
                ("gRPC-web_v2".to_owned(), 4, 15),
                ("works".to_owned(), 16, 21),
                (".".to_owned(), 21, 22)
            ]
        );
    }

    #[test]
    fn test_split_counts_characters_not_bytes() {
        let text = "José 🚀 en Málaga, ¿sí?";
        let words = split(text);
        assert_eq!(
            words,
            [
                ("José".to_owned(), 0, 4),
                ("🚀".to_owned(), 5, 6),
                ("en".to_owned(), 7, 9),
                ("Málaga".to_owned(), 10, 16),
                (",".to_owned(), 16, 17),
                ("¿".to_owned(), 18, 19),
                ("sí".to_owned(), 19, 21),
                ("?".to_owned(), 21, 22)
            ]
        );
        let chars: Vec<char> = text.chars().collect();
        let (word, start, end) = &words[3];
        assert_eq!(&chars[*start..*end].iter().collect::<String>(), word);
    }

    #[test]
    fn test_split_empty_and_blank() {
        assert!(split("").is_empty());
        assert!(split(" \n\t ").is_empty());
    }
}
