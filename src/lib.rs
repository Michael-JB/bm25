#![warn(missing_docs)]
#![cfg_attr(feature = "language_detection", doc = include_str!("../README.md"))]
// Suppress missing docs warning for this file
//!

#[cfg(test)]
mod test_data_loader;

mod embedder;
mod scorer;
mod search;
mod tokenizer;

#[cfg(feature = "default_tokenizer")]
mod default_tokenizer;

#[cfg(feature = "default_tokenizer")]
pub use default_tokenizer::{Language, LanguageMode};

pub use embedder::{
    DefaultTokenizer, Embedder, EmbedderBuilder, Embedding, TokenEmbedder, TokenEmbedding,
};
pub use scorer::{ScoredDocument, Scorer};
pub use search::{Document, SearchEngine, SearchEngineBuilder, SearchResult};
pub use tokenizer::Tokenizer;
