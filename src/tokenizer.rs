use cached::proc_macro::cached;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use whichlang::Lang as DetectedLanguage;

/// Languages supported by the tokenizer.
#[allow(missing_docs)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    English,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

/// The language mode used by the tokenizer. This determines the algorithm used for stemming and
/// the dictionary of stopwords.
#[derive(Debug, Clone)]
pub enum LanguageMode {
    /// Automatically detect the language. Note that this adds a small performance overhead.
    Detect,
    /// Use a fixed language.
    Fixed(Language),
}

impl From<Language> for LanguageMode {
    fn from(language: Language) -> Self {
        LanguageMode::Fixed(language)
    }
}

impl TryFrom<DetectedLanguage> for Language {
    type Error = ();

    fn try_from(detected_language: DetectedLanguage) -> Result<Self, Self::Error> {
        match detected_language {
            DetectedLanguage::Ara => Ok(Language::Arabic),
            DetectedLanguage::Cmn => Err(()),
            DetectedLanguage::Deu => Ok(Language::German),
            DetectedLanguage::Eng => Ok(Language::English),
            DetectedLanguage::Fra => Ok(Language::French),
            DetectedLanguage::Hin => Err(()),
            DetectedLanguage::Ita => Ok(Language::Italian),
            DetectedLanguage::Jpn => Err(()),
            DetectedLanguage::Kor => Err(()),
            DetectedLanguage::Nld => Ok(Language::Dutch),
            DetectedLanguage::Por => Ok(Language::Portuguese),
            DetectedLanguage::Rus => Ok(Language::Russian),
            DetectedLanguage::Spa => Ok(Language::Spanish),
            DetectedLanguage::Swe => Ok(Language::Swedish),
            DetectedLanguage::Tur => Ok(Language::Turkish),
            DetectedLanguage::Vie => Err(()),
        }
    }
}

impl From<&Language> for Algorithm {
    fn from(language: &Language) -> Self {
        match language {
            Language::Arabic => Algorithm::Arabic,
            Language::Danish => Algorithm::Danish,
            Language::Dutch => Algorithm::Dutch,
            Language::English => Algorithm::English,
            Language::French => Algorithm::French,
            Language::German => Algorithm::German,
            Language::Greek => Algorithm::Greek,
            Language::Hungarian => Algorithm::Hungarian,
            Language::Italian => Algorithm::Italian,
            Language::Norwegian => Algorithm::Norwegian,
            Language::Portuguese => Algorithm::Portuguese,
            Language::Romanian => Algorithm::Romanian,
            Language::Russian => Algorithm::Russian,
            Language::Spanish => Algorithm::Spanish,
            Language::Swedish => Algorithm::Swedish,
            Language::Tamil => Algorithm::Tamil,
            Language::Turkish => Algorithm::Turkish,
        }
    }
}

impl TryFrom<&Language> for stop_words::LANGUAGE {
    type Error = ();

    fn try_from(language: &Language) -> Result<Self, Self::Error> {
        match language {
            Language::Arabic => Ok(stop_words::LANGUAGE::Arabic),
            Language::Danish => Ok(stop_words::LANGUAGE::Danish),
            Language::Dutch => Ok(stop_words::LANGUAGE::Dutch),
            Language::English => Ok(stop_words::LANGUAGE::English),
            Language::French => Ok(stop_words::LANGUAGE::French),
            Language::German => Ok(stop_words::LANGUAGE::German),
            Language::Greek => Ok(stop_words::LANGUAGE::Greek),
            Language::Hungarian => Ok(stop_words::LANGUAGE::Hungarian),
            Language::Italian => Ok(stop_words::LANGUAGE::Italian),
            Language::Norwegian => Ok(stop_words::LANGUAGE::Norwegian),
            Language::Portuguese => Ok(stop_words::LANGUAGE::Portuguese),
            Language::Romanian => Ok(stop_words::LANGUAGE::Romanian),
            Language::Russian => Ok(stop_words::LANGUAGE::Russian),
            Language::Spanish => Ok(stop_words::LANGUAGE::Spanish),
            Language::Swedish => Ok(stop_words::LANGUAGE::Swedish),
            Language::Tamil => Err(()),
            Language::Turkish => Ok(stop_words::LANGUAGE::Turkish),
        }
    }
}

#[cached(size = 16)]
fn get_stopwords(language: Language) -> HashSet<String> {
    match TryInto::<stop_words::LANGUAGE>::try_into(&language) {
        Err(_) => HashSet::new(),
        Ok(lang) => stop_words::get(lang).into_iter().collect(),
    }
}

pub(crate) struct Tokenizer {
    language_mode: LanguageMode,
    stemmer: Option<Stemmer>,
    stopwords: HashSet<String>,
}

impl Tokenizer {
    pub fn new(language_mode: &LanguageMode) -> Tokenizer {
        Tokenizer {
            language_mode: language_mode.clone(),
            stemmer: match language_mode {
                LanguageMode::Detect => None,
                LanguageMode::Fixed(lang) => Some(Stemmer::create(lang.into())),
            },
            stopwords: match language_mode {
                LanguageMode::Detect => HashSet::new(),
                LanguageMode::Fixed(lang) => get_stopwords(lang.clone()),
            },
        }
    }

    fn split_by_whitespace_and_punctuation(text: &str) -> impl Iterator<Item = &'_ str> {
        text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
    }

    fn stem<'a>(
        &self,
        stemmer: Option<&Stemmer>,
        words: impl Iterator<Item = &'a str> + 'a,
    ) -> Vec<String> {
        words
            .map(|word| match stemmer {
                Some(stemmer) => stemmer.stem(word).to_string(),
                None => word.to_string(),
            })
            .collect()
    }

    fn _tokenize(
        &self,
        input_text: &str,
        stemmer: Option<&Stemmer>,
        stopwords: &HashSet<String>,
    ) -> Vec<String> {
        let text = input_text.to_lowercase();
        let tokens = Tokenizer::split_by_whitespace_and_punctuation(&text);
        let tokens = tokens.filter(|token| !stopwords.contains(*token));
        self.stem(stemmer, tokens)
    }

    pub fn tokenize(&self, input_text: &str) -> Vec<String> {
        if input_text.is_empty() {
            return Vec::new();
        }
        match &self.language_mode {
            LanguageMode::Detect => {
                let detected_language =
                    Language::try_from(whichlang::detect_language(input_text)).ok();
                let stemmer = detected_language
                    .as_ref()
                    .map(|lang| Stemmer::create(lang.into()));
                let stopwords = match &detected_language {
                    Some(lang) => get_stopwords(lang.clone()),
                    None => HashSet::new(),
                };
                return self._tokenize(input_text, stemmer.as_ref(), &stopwords);
            }
            LanguageMode::Fixed(_) => {
                return self._tokenize(input_text, self.stemmer.as_ref(), &self.stopwords);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_data_loader::tests::{read_recipes, Recipe};

    use super::*;

    use insta::assert_debug_snapshot;

    #[test]
    fn it_can_tokenize_english() {
        let text = "space station";
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens, vec!["space", "station"]);
    }

    #[test]
    fn it_converts_to_lowercase() {
        let text = "SPACE STATION";
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens, vec!["space", "station"]);
    }

    #[test]
    fn it_removes_whitespace() {
        let text = "\tspace\r\nstation\n ";
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens, vec!["space", "station"]);
    }

    #[test]
    fn it_removes_stopwords() {
        let text = "i me my myself we our ours ourselves you you're you've you'll you'd";
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        let tokens = tokenizer.tokenize(text);

        assert!(tokens.is_empty());
    }

    #[test]
    fn it_keeps_numbers() {
        let text = "42 1337";
        let tokenizer = Tokenizer::new(&LanguageMode::Detect);

        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens, vec!["42", "1337"]);
    }

    #[test]
    fn it_removes_punctuation() {
        let test_cases = vec![
            ("space, station!", vec!["space", "station"]),
            ("space,station", vec!["space", "station"]),
            ("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", vec![]),
        ];
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        for (text, expected) in test_cases {
            let tokens = tokenizer.tokenize(text);
            assert_eq!(tokens, expected);
        }
    }

    #[test]
    fn it_stems_words() {
        let text = "connection connections connective connected connecting connect";
        let tokenizer = Tokenizer::new(&LanguageMode::Fixed(Language::English));

        let tokens = tokenizer.tokenize(text);

        assert_eq!(
            tokens,
            vec!["connect", "connect", "connect", "connect", "connect", "connect"]
        );
    }

    #[test]
    fn it_handles_empty_input() {
        let text = "";
        let tokenizer = Tokenizer::new(&LanguageMode::Detect);

        let tokens = tokenizer.tokenize(text);

        assert!(tokens.is_empty());
    }

    #[test]
    fn it_matches_snapshot() {
        let recipes = read_recipes("recipes_en.csv");

        let tokens: Vec<_> = recipes
            .iter()
            .map(|Recipe { recipe, .. }| {
                let tokenizer = Tokenizer::new(&LanguageMode::Detect);
                tokenizer.tokenize(recipe)
            })
            .collect();

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(tokens);
        });
    }
}
