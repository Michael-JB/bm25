pub use crate::tokenizer::LanguageMode;

use crate::tokenizer::Tokenizer;
use fxhash::{hash, hash32, hash64};
#[cfg(feature = "parallelism")]
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fmt::{self, Debug, Display},
    hash::Hash,
    marker::PhantomData,
};

pub type DefaultEmbeddingDimension = u32;

/// Represents a document embedded in a D-dimensional space.
/// The structure and naming of this struct matches the common format for BM25 embeddings.
#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub struct Embedding<D: EmbeddingDimension = DefaultEmbeddingDimension> {
    /// The index of each token in the embedding space, where indices\[i\] corresponds to the ith token.
    pub indices: Vec<D>,
    /// The value of each token in the embedding space, where values\[i\] corresponds to indices\[i\].
    pub values: Vec<f32>,
}

impl<D: EmbeddingDimension> Display for Embedding<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedding {{ indices: {:?}, values: {:?} }}",
            self.indices, self.values
        )
    }
}

/// Embeds text into a D-dimensional space using the BM25 algorithm.
pub struct Embedder<D: EmbeddingDimension = DefaultEmbeddingDimension> {
    tokenizer: Tokenizer,
    k1: f32,
    b: f32,
    avgdl: f32,
    embedding_dimension: PhantomData<D>,
}

impl<D> Display for Embedder<D>
where
    D: EmbeddingDimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedder {{ k1: {}, b: {}, avgdl: {} }}",
            self.k1, self.b, self.avgdl
        )
    }
}

/// A trait for embedding. Implement this to customise the embedding space and function.
pub trait EmbeddingDimension: Eq + Hash + Clone + Debug + Send + Sync {
    /// Embeds a token into the embedding space.
    fn embed(token: &str) -> Self;
}

impl EmbeddingDimension for u32 {
    fn embed(token: &str) -> u32 {
        hash32(token)
    }
}

impl EmbeddingDimension for u64 {
    fn embed(token: &str) -> u64 {
        hash64(token)
    }
}

impl EmbeddingDimension for usize {
    fn embed(token: &str) -> usize {
        hash(token)
    }
}

impl<D: EmbeddingDimension> Embedder<D> {
    const FALLBACK_AVGDL: f32 = 256.0;

    /// Returns the average document length used by the embedder.
    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    /// Embeds a batch of texts into the embedding space. Use the `parallelism` feature to speed
    /// this up for large batches.
    pub fn batch_embed(&self, texts: &[&str]) -> Vec<Embedding<D>> {
        #[cfg(not(feature = "parallelism"))]
        let text_iter = texts.iter();
        #[cfg(feature = "parallelism")]
        let text_iter = texts.par_iter();
        text_iter.map(|text| self.embed(text)).collect()
    }

    /// Embeds the given text into the embedding space.
    pub fn embed(&self, text: &str) -> Embedding<D> {
        let tokens = self.tokenizer.tokenize(text);

        let avgdl = if self.avgdl <= 0.0 {
            Self::FALLBACK_AVGDL
        } else {
            self.avgdl
        };
        let indices: Vec<D> = tokens.iter().map(|s| D::embed(s)).collect();
        let counts = indices.iter().fold(HashMap::new(), |mut acc, token| {
            let count = acc.entry(token).or_insert(0);
            *count += 1;
            acc
        });
        let values: Vec<f32> = indices
            .iter()
            .map(|i| {
                let term_frequency = *counts.get(i).unwrap_or(&0) as f32;
                let numerator = term_frequency * (self.k1 + 1.0);
                let denominator = term_frequency
                    + self.k1 * (1.0 - self.b + self.b * (tokens.len() as f32 / avgdl));
                numerator / denominator
            })
            .collect();

        Embedding { indices, values }
    }
}

/// A non-consuming builder for Embedder.
pub struct EmbedderBuilder<D: EmbeddingDimension = DefaultEmbeddingDimension> {
    k1: f32,
    b: f32,
    avgdl: f32,
    language_mode: LanguageMode,
    embedding_dimension: PhantomData<D>,
}

impl<D: EmbeddingDimension> EmbedderBuilder<D> {
    /// Constructs a new EmbedderBuilder with the given average document length. Use this if you
    /// know the average document length in advance. If you don't, but you have your full corpus
    /// ahead of time, use `with_fit_to_corpus` instead.
    ///
    /// If you have neither the full corpus nor a sample of it, you can configure the embedder to
    /// disregard document length by setting `b` to 0.0. In this case, it doesn't matter what
    /// value you pass to `with_avgdl`.
    ///
    /// The average document length is the average number of tokens in a document from your corpus;
    /// if you need access to this value, you can construct an Embedder and call `avgdl` on it.
    pub fn with_avgdl(avgdl: f32) -> EmbedderBuilder<D> {
        EmbedderBuilder {
            k1: 1.2,
            b: 0.75,
            avgdl,
            language_mode: LanguageMode::default(),
            embedding_dimension: PhantomData,
        }
    }

    /// Constructs a new EmbedderBuilder with its average document length fit to the given corpus.
    /// Use this if you have the full corpus (or a sample of it) available in advance.
    /// Use the `parallelism` feature to speed this up for large corpora. When you call `build`,
    /// the builder will set the language mode of the Embedder to `language_mode`.
    pub fn with_fit_to_corpus(
        language_mode: impl Into<LanguageMode>,
        corpus: &[&str],
    ) -> EmbedderBuilder<D> {
        let language_mode = language_mode.into();
        let tokenizer = Tokenizer::new(&language_mode);

        let avgdl = if corpus.is_empty() {
            Embedder::<D>::FALLBACK_AVGDL
        } else {
            #[cfg(not(feature = "parallelism"))]
            let corpus_iter = corpus.iter();
            #[cfg(feature = "parallelism")]
            let corpus_iter = corpus.par_iter();
            let total_len: u64 = corpus_iter
                .map(|doc| tokenizer.tokenize(doc).len() as u64)
                .sum();
            (total_len as f64 / corpus.len() as f64) as f32
        };

        EmbedderBuilder {
            k1: 1.2,
            b: 0.75,
            avgdl,
            language_mode,
            embedding_dimension: PhantomData,
        }
    }

    /// Sets the k1 parameter for the embedder. The default value is 1.2.
    pub fn k1(&mut self, k1: f32) -> &EmbedderBuilder<D> {
        self.k1 = k1;
        self
    }

    /// Sets the b parameter for the embedder. The default value is 0.75.
    pub fn b(&mut self, b: f32) -> &EmbedderBuilder<D> {
        self.b = b;
        self
    }

    /// Overrides the average document length for the embedder.
    pub fn avgdl(&mut self, avgdl: f32) -> &EmbedderBuilder<D> {
        self.avgdl = avgdl;
        self
    }

    /// Sets the language mode for the embedder. The default value is `LanguageMode::Detect`.
    pub fn language_mode(&mut self, language_mode: impl Into<LanguageMode>) -> &EmbedderBuilder<D> {
        self.language_mode = language_mode.into();
        self
    }

    /// Builds the Embedder.
    pub fn build(&self) -> Embedder<D> {
        Embedder {
            tokenizer: Tokenizer::new(&self.language_mode),
            k1: self.k1,
            b: self.b,
            avgdl: self.avgdl,
            embedding_dimension: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::{
        test_data_loader::tests::{read_recipes, Recipe},
        Language,
    };

    use super::*;

    fn embed_recipes(recipe_file: &str, language_mode: LanguageMode) -> Vec<Embedding> {
        let recipes = read_recipes(recipe_file);
        let embedder = EmbedderBuilder::with_fit_to_corpus(
            language_mode,
            &recipes
                .iter()
                .map(|Recipe { recipe, .. }| recipe.as_str())
                .collect::<Vec<_>>(),
        )
        .build();

        embedder.batch_embed(
            &recipes
                .iter()
                .map(|Recipe { recipe, .. }| recipe.as_str())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn it_weights_unique_words_equally() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(3.0)
            .language_mode(Language::English)
            .build();
        let embedding = embedder.embed("banana apple orange");

        assert!(embedding.indices.len() == 3);
        assert!(embedding.values.len() == 3);
        assert!(embedding.values.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn it_weights_repeated_words_unequally() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(3.0)
            .language_mode(Language::English)
            .build();
        let embedding = embedder.embed("space station station");

        assert!(embedding.indices == vec![866767497, 666609503, 666609503]);
        assert!(embedding.values == vec![1.0, 1.375, 1.375]);
    }

    #[test]
    fn it_constrains_avgdl() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(0.0)
            .language_mode(Language::English)
            .build();

        let embedding = embedder.embed("space station");

        assert!(!embedding.indices.is_empty());
        assert!(!embedding.values.is_empty());
        assert!(embedding.values.into_iter().all(|v| v > 0.0));
    }

    #[test]
    fn it_handles_empty_corpus() {
        let embedder = EmbedderBuilder::<u32>::with_fit_to_corpus(Language::English, &[]).build();

        let embedding = embedder.embed("space station");

        assert!(!embedding.indices.is_empty());
        assert!(!embedding.values.is_empty());
    }

    #[test]
    fn batch_embedding_is_consistent() {
        let corpus = ["The fire crackled, casting flickering shadows on the cabin walls."; 1000];
        let embedder = EmbedderBuilder::<u32>::with_avgdl(7.0)
            .language_mode(Language::English)
            .build();

        let embeddings = embedder.batch_embed(&corpus);

        assert!(embeddings.windows(2).all(|e| e[0] == e[1]));
    }

    #[test]
    fn it_handles_empty_input() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(1.0).build();

        let embedding = embedder.embed("");

        assert!(embedding.indices.is_empty());
        assert!(embedding.values.is_empty());
    }

    #[test]
    fn it_allows_customisation_of_embedder() {
        #[derive(Eq, PartialEq, Hash, Clone, Debug)]
        struct MyType(u32);

        impl EmbeddingDimension for MyType {
            fn embed(_: &str) -> Self {
                MyType(42)
            }
        }

        let bm25_embedder = EmbedderBuilder::<MyType>::with_avgdl(2.0).build();

        let embedding = bm25_embedder.embed("space station");

        assert_eq!(embedding.indices, vec![MyType(42), MyType(42)]);
    }

    #[test]
    fn it_matches_snapshot_en() {
        let embeddings = embed_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(embeddings);
        });
    }

    #[test]
    fn it_matches_snapshot_de() {
        let embeddings = embed_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(embeddings);
        });
    }
}
