# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2024-09-20

### Added
- You can now enable/disable stop word removal via feature flags. Disabling stop word removal will
  remove `cached` and `stop-words` from your dependency tree.
- You can now choose the stop word list. Options are `nltk_stopwords` and `iso_stopwords`.

### Changed
- The default stop words list is now NLTK. This change affects embeddings; if you are upgrading
  to this version and want to stay aligned with existing embeddings, disable default features and
  use the `iso_stopwords` feature.

## [0.2.1] - 2024-09-09

### Fixed
- Removed some (unreachable) unwraps in favour of panic-free alternatives.

## [0.2.0] - 2024-09-09

### Added
- Impl `Display` for `Embedder`

### Changed
- Moved language detection to the `language_detection` feature. If you were previously using
  `LanguageMode::Detect`, you'll now need to explicitly enable this feature with
  `cargo add bm25 --features language_detection`. If you were not using `LanguageMode::Detect`,
  you now have one dependency fewer.
- The `LanguageMode` enum is now non-exhaustive. This is to allow conditional compilation of the
  `Detect` variant.
- The default language mode has been changed to `LanguageMode::Fixed(Language::English)`. This is
  to avoid unexpected behaviour changes with feature unification.

## [0.1.1] - 2024-09-08

### Added
- Added `parallelism` feature. You can now fit and embed a corpus in parallel.
- Added `batch_embed` method to `Embedder`. 
- Implemented some common traits to improve interoperability.

## [0.1.0] - 2024-09-08

Initial release.
