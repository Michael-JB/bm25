use cached::proc_macro::cached;
use rust_stemmers::{Algorithm as StemmingAlgorithm, Stemmer};
use std::collections::HashSet;
use stop_words::LANGUAGE as StopWordLanguage;
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
#[derive(Debug, Clone, Default)]
pub enum LanguageMode {
    /// Automatically detect the language. Note that this adds a small performance overhead.
    #[default]
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

impl From<&Language> for StemmingAlgorithm {
    fn from(language: &Language) -> Self {
        match language {
            Language::Arabic => StemmingAlgorithm::Arabic,
            Language::Danish => StemmingAlgorithm::Danish,
            Language::Dutch => StemmingAlgorithm::Dutch,
            Language::English => StemmingAlgorithm::English,
            Language::French => StemmingAlgorithm::French,
            Language::German => StemmingAlgorithm::German,
            Language::Greek => StemmingAlgorithm::Greek,
            Language::Hungarian => StemmingAlgorithm::Hungarian,
            Language::Italian => StemmingAlgorithm::Italian,
            Language::Norwegian => StemmingAlgorithm::Norwegian,
            Language::Portuguese => StemmingAlgorithm::Portuguese,
            Language::Romanian => StemmingAlgorithm::Romanian,
            Language::Russian => StemmingAlgorithm::Russian,
            Language::Spanish => StemmingAlgorithm::Spanish,
            Language::Swedish => StemmingAlgorithm::Swedish,
            Language::Tamil => StemmingAlgorithm::Tamil,
            Language::Turkish => StemmingAlgorithm::Turkish,
        }
    }
}

impl TryFrom<&Language> for StopWordLanguage {
    type Error = ();

    fn try_from(language: &Language) -> Result<Self, Self::Error> {
        match language {
            Language::Arabic => Ok(StopWordLanguage::Arabic),
            Language::Danish => Ok(StopWordLanguage::Danish),
            Language::Dutch => Ok(StopWordLanguage::Dutch),
            Language::English => Ok(StopWordLanguage::English),
            Language::French => Ok(StopWordLanguage::French),
            Language::German => Ok(StopWordLanguage::German),
            Language::Greek => Ok(StopWordLanguage::Greek),
            Language::Hungarian => Ok(StopWordLanguage::Hungarian),
            Language::Italian => Ok(StopWordLanguage::Italian),
            Language::Norwegian => Ok(StopWordLanguage::Norwegian),
            Language::Portuguese => Ok(StopWordLanguage::Portuguese),
            Language::Romanian => Ok(StopWordLanguage::Romanian),
            Language::Russian => Ok(StopWordLanguage::Russian),
            Language::Spanish => Ok(StopWordLanguage::Spanish),
            Language::Swedish => Ok(StopWordLanguage::Swedish),
            Language::Tamil => Err(()),
            Language::Turkish => Ok(StopWordLanguage::Turkish),
        }
    }
}

#[cached(size = 16)]
fn get_stopwords(language: Language) -> HashSet<String> {
    match TryInto::<StopWordLanguage>::try_into(&language) {
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

    fn tokenize_recipes(recipe_file: &str, language_mode: LanguageMode) -> Vec<Vec<String>> {
        let recipes = read_recipes(recipe_file);

        recipes
            .iter()
            .map(|Recipe { recipe, .. }| {
                let tokenizer = Tokenizer::new(&language_mode);
                tokenizer.tokenize(recipe)
            })
            .collect()
    }

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
        let tokenizer = Tokenizer::new(&LanguageMode::default());

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
        let tokenizer = Tokenizer::new(&LanguageMode::default());

        let tokens = tokenizer.tokenize(text);

        assert!(tokens.is_empty());
    }

    #[test]
    fn it_detects_english() {
        let tokens_detected = tokenize_recipes("recipes_en.csv", LanguageMode::Detect);
        let tokens_en = tokenize_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        assert_eq!(tokens_detected, tokens_en);
    }

    #[test]
    fn it_detects_german() {
        let tokens_detected = tokenize_recipes("recipes_de.csv", LanguageMode::Detect);
        let token_de = tokenize_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        assert_eq!(tokens_detected, token_de);
    }

    #[test]
    fn it_matches_snapshot_en() {
        let tokens = tokenize_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(tokens);
        });
    }

    #[test]
    fn it_matches_snapshot_de() {
        let tokens = tokenize_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(tokens);
        });
    }
}
