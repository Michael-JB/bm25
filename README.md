# bm25

A Rust crate that computes [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) embeddings for
information retrieval. You can use these embeddings in vector databases that support the storage
of sparse vectors, e.g. Qdrant, Pinecone, Milvus, etc.

This crate also contains a light-weight, in-memory full-text search engine built on top of the
embedder.

## The BM25 algorithm 

BM25 is an algorithm for scoring the relevance of a query to documents in a corpus. You can make
this scoring more efficient by pre-computing a 'sparse embedding' of each document. You can use
these sparse embeddings directly, or upload them to a vector database and query them from there.

BM25 assumes that you know the average (meaningful) word count of your documents ahead of time. This
crate provides utilities to compute this. If this assumption doesn't hold for your use-case, you
have two options: (1) make a sensible guess (e.g. based on a sample); or (2) configure the algorithm
to disregard document length. The former is recommended if most of your documents are around the
same size.

BM25 has three hyperparameters: `b`, `k1` and `avgdl`. These terms match the formula given on
Wikipedia. `avgdl` ('average document length') is the aforementioned average meaningful word count;
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

### Embed some text

The best way to embed some text is to fit an embedder to your corpus. 
```rust
use bm25::{EmbedderBuilder, Embedder, Embedding, Language};

let corpus = [
    "The sky blushed pink as the sun dipped below the horizon.",
    "Apples, oranges, papayas, and more papayas.",
    "She found a forgotten letter tucked inside an old book.",
    "A single drop of rain fell, followed by a thousand more.",
    "His laughter echoed through the empty streets, breaking the silence.",
];

let embedder: Embedder = EmbedderBuilder::with_fit_to_corpus(Language::English, &corpus)
    .build();

let embedding = embedder.embed(corpus[1]);

assert_eq!(
    embedding, 
    Embedding {
        indices: vec![1777144781, 3887370161, 2177600299, 2177600299],
        values:  vec![1.0563674 , 1.0563674 , 1.4273626 , 1.4273626 ],
    }
)
```

For cases where you don't have the full corpus ahead of time, but have an approximate idea of the
average meaningful word count you expect, you can construct an embedder with your `avgdl` guess.
By default, this crate will detect the language of your input text to embed it accordingly. You
can configure the embedder to embed with a specific language if you know it in advance.

```rust
use bm25::{EmbedderBuilder, Embedder, Language};

let embedder: Embedder = EmbedderBuilder::with_avgdl(7.0)
    .language_mode(Language::German) // `Detect` is the default language mode
    .build();
```

If you want to disregard document length altogether, set `b` to 0.

```rust
use bm25::{EmbedderBuilder, Embedder};

let embedder: Embedder = EmbedderBuilder::with_avgdl(1.0) // if b = 0, avgdl has no effect
    .b(0f32)
    .build();
```

You can customise the dimensionality of your sparse vector via the generic parameter.
Supported values are `usize`, `u32` and `u64`. You can also use your own type (and also
inject your own embedding function) by implementing the `EmbeddingDimension` trait.

```rust
use bm25::{EmbedderBuilder, EmbeddingDimension, Language};


let text = "cup of tea";

// Embed into a u32-dimensional space
let embedder = EmbedderBuilder::<u32>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(embedding.indices, [2070875659, 415655421]);

// Embed into a u64-dimensional space
let embedder = EmbedderBuilder::<u64>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(embedding.indices, [3288102823240002853, 7123809554392261272]);

// Embed into a usize-dimensional space
let embedder = EmbedderBuilder::<usize>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(embedding.indices, [3288102823240002853, 7123809554392261272]);

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct MyType(u32);
impl EmbeddingDimension for MyType {
    fn embed(_: &str) -> Self {
        MyType(42)
    }
}

// Embed into a MyType-dimensional space
let embedder = EmbedderBuilder::<MyType>::with_avgdl(2.0).build();
let embedding = embedder.embed(text);
assert_eq!(embedding.indices, [MyType(42), MyType(42)]);
```

### Search

This crate includes a light-weight, in-memory full-text search engine built on top of the embedder.

```rust
use bm25::{Language, SearchEngineBuilder, SearchResult, Document};

let corpus = [
    "The rabbit munched the orange carrot.",
    "The snake hugged the green lizard.",
    "The hedgehog impaled the orange orange.",
    "The squirrel buried the brown nut.",
];

let search_engine = SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus)
    .build();

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

You can also construct a search engine with documents (allowing you to customise the id type and
value), or with an average document length.

```rust
use bm25::{Language, SearchEngineBuilder, Document};

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
let search_engine = SearchEngineBuilder::<u32>::with_avgdl(256.0).build();
```

You can also upsert or remove documents from the search engine. Note that mutating the search corpus
by upserting or removing documents will change the true value of `avgdl`. The more `avgdl` drifts
from its true value, the less accurate the BM25 scores will be.

```rust
use bm25::{SearchEngineBuilder, Document};

let mut search_engine = SearchEngineBuilder::<u32>::with_avgdl(10.0).build();

let document_id = 42;
let document = Document {
    id: document_id,
    contents: String::from(
        "A breeze carried the scent of blooming jasmine through the open window.",
    ),
};

search_engine.upsert(document.clone());
assert_eq!(search_engine.get(&document_id).unwrap(), document);

search_engine.remove(&document_id);
assert_eq!(search_engine.get(&document_id), None);
```

## License

[MIT License](https://github.com/Michael-JB/bm25/blob/main/LICENSE)

