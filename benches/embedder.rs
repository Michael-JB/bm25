use bm25::{EmbedderBuilder, Language, LanguageMode, TokenEmbedder};
use divan::Bencher;
use std::hash::Hash;

use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

const ENGLISH_TEXT: &str =
    "The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks in an encoder-decoder configuration. The best performing
models also connect the encoder and decoder through an attention mechanism. We
propose a new simple network architecture, the Transformer, based solely on attention
mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two
machine translation tasks show these models to be superior in quality while being more
parallelizable and requiring significantly less time to train.";

#[divan::bench(args = [
    #[cfg(feature = "language_detection")]
    LanguageMode::Detect,
    LanguageMode::Fixed(Language::English),
], sample_count = 10000)]
fn language_mode(bencher: Bencher, language_mode: &LanguageMode) {
    let embedder =
        EmbedderBuilder::<u32>::with_fit_to_corpus(language_mode.clone(), &[ENGLISH_TEXT]).build();
    // Run once beforehand to warm up the cache
    embedder.embed(ENGLISH_TEXT);

    bencher.bench(|| embedder.embed(ENGLISH_TEXT));
}

#[divan::bench(types = [usize, u32, u64], sample_count = 10000)]
fn dimension<T>(bencher: Bencher)
where
    T: TokenEmbedder + Hash + Eq,
{
    bencher
        .with_inputs(|| {
            let embedder =
                EmbedderBuilder::<T>::with_fit_to_corpus(Language::English, &[ENGLISH_TEXT])
                    .build();
            // Run once beforehand to warm up the cache
            embedder.embed(ENGLISH_TEXT);
            embedder
        })
        .bench_values(|embedder| embedder.embed(ENGLISH_TEXT));
}

#[divan::bench]
fn fit_to_corpus(bencher: Bencher) {
    let corpus = [ENGLISH_TEXT; 5000];
    // Run once beforehand to warm up the cache
    let _ = EmbedderBuilder::<u32>::with_fit_to_corpus(Language::English, &corpus).build();

    bencher
        .bench(|| EmbedderBuilder::<u32>::with_fit_to_corpus(Language::English, &corpus).build());
}
