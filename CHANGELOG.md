# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Introduces `TokenEmbedder::EmbeddingSpace` to decouple the output of `TokenEmbedder` from `Self`.
  This lets you customise the output of your `TokenEmbedder` without changing its type.

## [1.0.1] - 2024-11-10

### Fixed

- Correctly embed the README in the crate documentation. docs.rs should now display the README
  correctly.

## [1.0.0] - 2024-11-10

### Added

- `Scorer` that lets you score documents against a query. Previously, the crate did not expose
  this level of abstraction. This allows fine-grained access to the BM25 algorithm for those
  interested in the raw scores, e.g., if you're not using a vector database.
- `Tokenizer` trait that lets you use your own tokenizer with this crate. This now lets you use
  your own tokenizer with the `SearchEngine` as well as the `Embedder`.

### Changed

- Structure of an `Embedding`. Previously, the Embedding type had two fields; `indices` and
  `values`. This matched the (JSON/Python) formats commonly used in vector database APIs/SDKs.
  However, in Rust this format does not guarantee that the length/order of your indices match the
  values. While the crate does not emit invalid embeddings such as this, it is nicer to enforce
  this: the new structure pairs each index with its corresponding value, wrapping both in the new
  `TokenEmbedding` type. I've included `indices()` and `values()` convenience methods to get the
  data in a format compatible with the old one if you need it.
- The tokenizer provided by this crate is now available behind the `default_tokenizer` feature,
  which is enabled by default.
- Trait bounds moved from struct definitions to the requiring impl blocks. This makes the API
  less restrictive and should help prevent trait bound pollution.
- Renamed `EmbeddingDimension` trait to `TokenEmbedder`; this is more descriptive of what the
  trait does.
- `EmbedderBuilder` is now a consuming builder.

### Removed

- `Embedder::embed_tokens`. To use your own tokenizer, you can implement the
  `Tokenizer` trait and pass your type to the `Embedder`/`SearchEngine`.
- `iso_stopwords` feature. The default tokenizer is no longer configurable via feature flags.
- `nltk_stopwords` feature. The default tokenizer is no longer configurable via feature flags.
- Unnecessary `Display` impl for `Embedder`.
- `Embedder::batch_embed` function. Enabling the `parallelism` feature causes this function to
  return embeddings in a different order than the input, so I removed it to avoid confusion.

## [0.3.1] - 2024-10-04

### Added
- Added an `embed_tokens` function to `Embedder`. This lets you use your own tokenizer with this
  crate.

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
