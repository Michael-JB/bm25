# bm25

[![Docs](https://docs.rs/bm25/badge.svg)](https://docs.rs/bm25/)
[![Crates.io Version](https://img.shields.io/crates/v/bm25)](https://crates.io/crates/bm25)
[![Crates.io Total Downloads](https://img.shields.io/crates/d/bm25)](https://crates.io/crates/bm25)

A Rust crate for everything [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). This crate provides
utilities at three levels of abstraction:
1. **BM25 Embedder**: Embeds text into a sparse vector space for information retrieval. You can use
    these embeddings with vector databases, e.g., Qdrant, Pinecone and Milvus, etc.
2. **BM25 Scorer**: Efficiently scores the relevance of a query embedding to document embeddings.
3. **BM25 Search Engine**: A fast, light-weight, in-memory keyword search engine built on top of
    the embedder and scorer.

See [bm25-demo](https://michael-jb.github.io/bm25-demo) for a WebAssembly demo of the search engine.

## Features

- Multilingual tokenizer with stemming, stop word removal and unicode normalization
- Language detection
- Full access to BM25 parameters
- Parallelism for fast batch-fitting
- Modular and customisable
- Configurable via compile-time features

## The BM25 algorithm 

[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is an algorithm for scoring the relevance of a
query to documents in a corpus. You can make this scoring more efficient by pre-computing a
'sparse embedding' of each document. You can use these sparse embeddings directly, or upload them
to a vector database and query them from there.

BM25 assumes that you know the average (meaningful) token count of your documents ahead of time. This
crate provides utilities to compute this. If this assumption doesn't hold for your use-case, you
have two options: (1) make a sensible guess (e.g. based on a sample); or (2) configure the algorithm
to disregard document length. The former is recommended.

BM25 has three parameters: `b`, `k1` and `avgdl`. These terms match the formula given on
Wikipedia. `avgdl` ('average document length') is the aforementioned average meaningful token count;
you should always provide a value for this and the crate can fit this for you. `b` controls
document length normalization; `0` means no normalisation (length will not affect score) while `1`
means full normalisation. If you know `avgdl`, `0.75` is typically a good choice for `b`. If
you're guessing `avgdl`, you can use a slightly lower `b` to reduce the effect of document length
on score. If you have no idea what `avgdl` is, set `b` to `0`. `k1` controls how much weight is
given to recurring tokens. For almost all use-cases, a value of `1.2` is suitable.

## Getting started

Add `bm25` to your project with

```sh
cargo add bm25
```

Depending on your use-case, you may want to read more about the [Embedder](#embed), [Scorer](#score)
or [SearchEngine](#search).

### Embed

You can generate sparse embeddings from text using this crate. You can store and query these in
most vector databases, or use them directly. The best way to embed some text is to fit an embedder
to your corpus.

```rust
use bm25::{Embedder, EmbedderBuilder, Embedding, TokenEmbedding, Language};

let corpus = [
    "The sky blushed pink as the sun dipped below the horizon.",
    "Apples, oranges, papayas, and more papayas.",
    "She found a forgotten letter tucked inside an old book.",
    "A single drop of rain fell, followed by a thousand more.",
];

let embedder: Embedder = EmbedderBuilder::with_fit_to_corpus(Language::English, &corpus).build();

assert_eq!(embedder.avgdl(), 5.75);

let embedding = embedder.embed(corpus[1]);

assert_eq!(
    embedding,
    Embedding(vec![
        TokenEmbedding {
            index: 1777144781,
            value: 1.1422123,
        },
        TokenEmbedding {
            index: 3887370161,
            value: 1.1422123,
        },
        TokenEmbedding {
            index: 2177600299,
            value: 1.5037148,
        },
        TokenEmbedding {
            index: 2177600299,
            value: 1.5037148,
        },
    ])
)
```

#### BM25 parameters

For cases where you don't have the full corpus ahead of time, but have an approximate idea of the
average meaningful word count you expect, you can construct an embedder with your `avgdl` guess.

```rust
use bm25::{Embedder, EmbedderBuilder};

let embedder: Embedder = EmbedderBuilder::with_avgdl(7.0)
    .build();
```

If you want to disregard document length altogether, set `b` to 0.

```rust
use bm25::{Embedder, EmbedderBuilder};

let embedder: Embedder = EmbedderBuilder::with_avgdl(1.0)
    .b(0.0) // if b = 0, avgdl has no effect
    .build();
```

#### Language

By default, the embedder uses an English `DefaultTokenizer`. If you are working with a different
language, you can configure the embedder to tokenize accordingly.

```rust
use bm25::{Embedder, EmbedderBuilder, Language};

let embedder: Embedder = EmbedderBuilder::with_avgdl(256.0)
    .language_mode(Language::German)
    .build();
```

If your corpus is multilingual, or you don't know the language ahead of time, you can enable the
`language_detection` feature.

```sh
cargo add bm25 --features language_detection
```

This unlocks the `LanguageMode::Detect` enum value. In this mode, the tokenizer will try to detect
the language of each piece of input text before tokenizing. Note that there is a small performance
overhead when embedding in this mode.

```rust
use bm25::{Embedder, EmbedderBuilder, LanguageMode};

let embedder: Embedder = EmbedderBuilder::with_avgdl(64.0)
    .language_mode(LanguageMode::Detect)
    .build();
```

#### Tokenizer

The embedder uses a tokenizer to convert text into a sequence of tokens to embed. The default
tokenizer detects language, normalizes unicode, splits on unicode word boundaries, removes stop
words and stems the remaining words. You can customise its behaviour by using the builder.

```rust
use bm25::{DefaultTokenizer, Language, Tokenizer};

let tokenizer = DefaultTokenizer::builder()
    .language_mode(Language::English)
    .normalization(true) // Normalize unicode (e.g., 'é' -> 'e', '🍕' -> 'pizza', etc.)
    .stopwords(true) // Remove common words with little meaning (e.g., 'the', 'and', 'of', etc.)
    .stemming(true) // Reduce words to their root form (e.g., 'running' -> 'run')
    .build();

let text = "Slices of 🍕";

let tokens = tokenizer.tokenize(text);

// 'slices' is stemmed to 'slice', 'of' is a stopword, and '🍕' is normalized to 'pizza'
assert_eq!(tokens, vec!["slice", "pizza"]);
```

While this works well for most languages and use-cases, this crate makes it easy for you to provide
your own tokenizer. All you have to do is implement the `Tokenizer` trait.

```rust
use bm25::{EmbedderBuilder, Embedding, Tokenizer};

#[derive(Default)]
struct MyTokenizer;

// Tokenize on occurrences of "T"
impl Tokenizer for MyTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text
            .split("T")
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
}

let embedder = EmbedderBuilder::<u32, MyTokenizer>::with_avgdl(1.0).build();

let embedding = embedder.embed("CupTofTtea");

assert_eq!(
    embedding.indices().cloned().collect::<Vec<_>>(),
    vec![3568447556, 3221979461, 415655421]
);
```

If you're not using the `DefaultTokenizer` at all, you can disable the `default_tokenizer` feature
to remove some dependencies from your project.

```sh
cargo add bm25 --no-default-features
```

#### Embedding space

You can customise the dimensionality of your sparse vector via the generic parameter. Supported
values are `usize`, `u32` and `u64`. You can also use your own type (and inject your own embedding
function) by implementing the `TokenEmbedder` trait.

```rust
use bm25::{EmbedderBuilder, TokenEmbedder};

let text = "cup of tea";

// Embed into a u32-dimensional space
let embedder = EmbedderBuilder::<u32>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(
    embedding.indices().cloned().collect::<Vec<_>>(),
    [2070875659, 415655421]
);

// Embed into a u64-dimensional space
let embedder = EmbedderBuilder::<u64>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(
    embedding.indices().cloned().collect::<Vec<_>>(),
    [3288102823240002853, 7123809554392261272]
);

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct MyType(u32);
impl TokenEmbedder for MyType {
    type EmbeddingSpace = Self;
    fn embed(_token: &str) -> Self {
        MyType(42)
    }
}

// Embed into a MyType-dimensional space
let embedder = EmbedderBuilder::<MyType>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(
    embedding.indices().cloned().collect::<Vec<_>>(),
    [MyType(42), MyType(42)]
);
```

### Score

This crate provides a BM25 scorer that can efficiently score the relevance of a query embedding to
document embeddings. The scorer manages the complexity of maintaining token frequencies and indexes,
as well as the actual scoring. Use this if you need BM25 scoring, but want to manage the lifecycle
of raw documents yourself.

```rust
use bm25::{Embedder, EmbedderBuilder, Language, Scorer, ScoredDocument};

let corpus = [
    "The sky blushed pink as the sun dipped below the horizon.",
    "She found a forgotten letter tucked inside an old book.",
    "Apples, oranges, pink grapefruits, and more pink grapefruits.",
    "A single drop of rain fell, followed by a thousand more.",
];
let query = "pink";

let mut scorer = Scorer::<usize>::new();

let embedder: Embedder =
    EmbedderBuilder::with_fit_to_corpus(Language::English, &corpus).build();

for (i, document) in corpus.iter().enumerate() {
    let document_embedding = embedder.embed(document);
    scorer.upsert(&i, document_embedding);
}

let query_embedding = embedder.embed(query);

let score = scorer.score(&0, &query_embedding);
assert_eq!(score, Some(0.36260858));

let matches = scorer.matches(&query_embedding);
assert_eq!(
    matches,
    vec![
        ScoredDocument {
            id: 2,
            score: 0.4960082
        },
        ScoredDocument {
            id: 0,
            score: 0.36260858
        }
    ]
);
```

### Search

This crate includes a light-weight, in-memory keyword search engine built on top of the embedder
and scorer. See [bm25-demo](https://michael-jb.github.io/bm25-demo) for a WebAssembly demo.

```rust
use bm25::{Document, Language, SearchEngineBuilder, SearchResult};

let corpus = [
    "The rabbit munched the orange carrot.",
    "The snake hugged the green lizard.",
    "The hedgehog impaled the orange orange.",
    "The squirrel buried the brown nut.",
];

let search_engine = SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build();

let limit = 3;
let search_results = search_engine.search("orange", limit);

assert_eq!(
    search_results,
    vec![
        SearchResult {
            document: Document {
                id: 2,
                contents: String::from("The hedgehog impaled the orange orange."),
            },
            score: 0.4904281,
        },
        SearchResult {
            document: Document {
                id: 0,
                contents: String::from("The rabbit munched the orange carrot."),
            },
            score: 0.35667497,
        },
    ]
);
```

You can construct a search engine with documents (allowing you to customise the id type and
value), or with an average document length.

```rust
use bm25::{Document, Language, SearchEngineBuilder};

// Build a search engine from documents
let search_engine = SearchEngineBuilder::<&str>::with_documents(
    Language::English,
    [
        Document {
            id: "Guacamole",
            contents: String::from("avocado, lime juice, salt, onion, tomatoes, coriander."),
        },
        Document {
            id: "Hummus",
            contents: String::from("chickpeas, tahini, olive oil, garlic, lemon juice, salt."),
        },
    ],
)
.build();

// Build a search engine from avgdl
let search_engine = SearchEngineBuilder::<u32>::with_avgdl(128.0)
    .build();
```

You can upsert or remove documents from the search engine. Note that mutating the search corpus
by upserting or removing documents will change the true value of `avgdl`. The more `avgdl` drifts
from its true value, the less accurate the BM25 scores will be.

```rust
use bm25::{Document, SearchEngineBuilder};

let mut search_engine = SearchEngineBuilder::<u32>::with_avgdl(10.0)
    .build();

let document_id = 42;
let document = Document {
    id: document_id,
    contents: String::from(
        "A breeze carried the scent of blooming jasmine through the open window.",
    ),
};

search_engine.upsert(document.clone());
assert_eq!(search_engine.get(&document_id), Some(document));

search_engine.remove(&document_id);
assert_eq!(search_engine.get(&document_id), None);
```

### Working with a large corpus

If your corpus is large, fitting an embedder can be slow. Fortunately, you can trivially
parallelise this via the `parallelism` feature, which implements data parallelism using
[Rayon](https://crates.io/crates/rayon).

```sh
cargo add bm25 --features parallelism
```

## License

[MIT License](https://github.com/Michael-JB/bm25/blob/main/LICENSE)

