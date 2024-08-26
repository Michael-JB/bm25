#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod embedder;
mod search;
mod tokenizer;

#[cfg(test)]
mod test_data_loader;

pub use embedder::{Embedder, EmbedderBuilder, Embedding, EmbeddingDimension};
pub use search::{Document, SearchEngine, SearchEngineBuilder, SearchResult};
pub use tokenizer::{Language, LanguageMode};
