use crate::embedder::{
    DefaultEmbeddingDimension, Embedder, EmbedderBuilder, EmbeddingDimension, LanguageMode,
};
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Display},
    hash::Hash,
    marker::PhantomData,
};

use crate::embedder::Embedding;

/// A search engine that ranks documents with BM25.
/// K is the type of the document id and D is the type of the embedding dimension.
pub struct SearchEngine<K, D = DefaultEmbeddingDimension>
where
    K: Eq + Hash + Clone,
    D: EmbeddingDimension,
{
    /// The embedder used to convert documents into embeddings.
    embedder: Embedder<D>,
    /// A mapping from document ids to the document embeddings.
    document_embeddings: HashMap<K, Embedding<D>>,
    /// A mapping from document ids to document contents.
    documents: HashMap<K, String>,
    /// A mapping from token indices to the number of documents that contain that token.
    term_frequencies: HashMap<D, u32>,
    /// A mapping from token indices to the set of documents that contain that token.
    inverted_term_index: HashMap<D, HashSet<K>>,
}

impl<K, D> Display for SearchEngine<K, D>
where
    K: Eq + Hash + Clone + Debug,
    D: EmbeddingDimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SearchEngine {{ documents: {:?} }}", self.documents)
    }
}

/// A document that you can insert into a search engine. K is the type of the document, allowing
/// you to use any type as a document id. Note that it is more effient to use a numeric type.
#[derive(Eq, PartialEq, Debug, Clone, PartialOrd, Hash)]
pub struct Document<K>
where
    K: Eq + Hash + Clone + Debug,
{
    /// A unique identifier for the document.
    pub id: K,
    /// The contents of the document.
    pub contents: String,
}

impl<K> Display for Document<K>
where
    K: Eq + Hash + Clone + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.contents)
    }
}

impl<K> Document<K>
where
    K: Eq + Hash + Clone + Debug,
{
    /// Creates a new document with the given id and contents.
    pub fn new(id: K, contents: impl Into<String>) -> Document<K> {
        Document {
            id,
            contents: contents.into(),
        }
    }
}

/// A search result, containing a document and its BM25 score.
#[derive(PartialEq, Debug)]
pub struct SearchResult<K>
where
    K: Eq + Hash + Clone + Debug,
{
    /// The document that was found.
    pub document: Document<K>,
    /// The BM25 score of the document. A higher score means the document is more relevant to the
    /// query.
    pub score: f32,
}

impl<K, D> SearchEngine<K, D>
where
    K: Eq + Hash + Clone + Debug,
    D: EmbeddingDimension,
{
    /// Upserts a document into the search engine. If a document with the same id already exists,
    /// it will be replaced. Note that upserting a document will change the true value of `avgdl`.
    /// The more `avgdl` drifts from its true value, the less accurate the BM25 scores will be.
    pub fn upsert(&mut self, document: impl Into<Document<K>>) {
        let document = document.into();
        if self.documents.contains_key(&document.id) {
            self.remove(&document.id);
        }
        let Embedding { indices, values } = self.embedder.embed(document.contents.as_str());
        for token_index in indices.iter() {
            let term_frequency = self
                .term_frequencies
                .entry(token_index.clone())
                .or_insert(0);
            *term_frequency += 1;
            let document_set = self
                .inverted_term_index
                .entry(token_index.clone())
                .or_default();
            document_set.insert(document.id.clone());
        }
        self.document_embeddings
            .insert(document.id.clone(), Embedding { indices, values });
        self.documents
            .insert(document.id.clone(), document.contents);
    }

    /// Removes a document from the search engine if it exists.
    pub fn remove(&mut self, document_id: &K) {
        if let Some(document) = self.documents.remove(document_id) {
            let Embedding { indices, .. } = self.embedder.embed(&document);
            for token_index in indices.iter() {
                if let Some(term_frequency) = self.term_frequencies.get_mut(token_index) {
                    *term_frequency -= 1;
                }
                if let Some(document_set) = self.inverted_term_index.get_mut(token_index) {
                    document_set.remove(document_id);
                }
            }
            self.document_embeddings.remove(document_id);
        }
    }

    /// Gets the contents of a document by its id.
    pub fn get(&self, document_id: &K) -> Option<Document<K>> {
        self.documents.get(document_id).map(|contents| Document {
            id: document_id.clone(),
            contents: contents.clone(),
        })
    }

    fn idf(&self, token_index: &D) -> f32 {
        let term_frequency = *self.term_frequencies.get(token_index).unwrap_or(&0) as f32;
        let numerator = self.documents.len() as f32 - term_frequency + 0.5;
        let denominator = term_frequency + 0.5;
        (1f32 + (numerator / denominator)).ln()
    }

    /// Searches the documents for the given query and returns the top `limit` results.
    /// Only the document contents are searched, not the document ids.
    pub fn search(&self, query: &str, limit: impl Into<Option<usize>>) -> Vec<SearchResult<K>> {
        let Embedding { indices, .. } = self.embedder.embed(query);

        // Reduce search space by filtering out all documents whose score would be 0
        let relevent_embedding_it = indices
            .iter()
            .filter_map(|token_index| self.inverted_term_index.get(token_index))
            .flat_map(|document_set| document_set.iter())
            .collect::<HashSet<_>>()
            .into_iter()
            .filter_map(|document_id| {
                self.document_embeddings
                    .get(document_id)
                    .map(|embedding| (document_id, embedding))
            });

        // Compute the score of each candidate document
        let mut scores = HashMap::new();
        for (document_id, embedding) in relevent_embedding_it {
            let mut document_score = 0f32;
            for token_index in indices.iter() {
                let token_idf = self.idf(token_index);
                let token_index_value = embedding
                    .indices
                    .iter()
                    .zip(embedding.values.iter())
                    .find(|(i, _)| **i == *token_index)
                    .map(|(_, v)| *v)
                    .unwrap_or(0f32);
                let token_score = token_idf * token_index_value;
                document_score += token_score;
            }
            scores.insert(document_id, document_score);
        }

        // Sort and format results
        let mut results: Vec<_> = scores.iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        results
            .into_iter()
            .take(limit.into().unwrap_or(usize::MAX))
            .filter_map(|(id, score)| {
                self.get(id).map(|document| SearchResult {
                    document,
                    score: *score,
                })
            })
            .collect()
    }
}

/// A consuming builder for SearchEngine.
pub struct SearchEngineBuilder<K, D = DefaultEmbeddingDimension>
where
    K: Eq + Hash + Clone + Debug,
    D: EmbeddingDimension,
{
    embedder_builder: EmbedderBuilder<D>,
    documents: Vec<Document<K>>,
    document_id_type: PhantomData<K>,
    embedding_dimension_type: PhantomData<D>,
}

impl<K, D> SearchEngineBuilder<K, D>
where
    K: Eq + Hash + Clone + Debug,
    D: EmbeddingDimension,
{
    /// Constructs a new SearchEngineBuilder with the given average document length. Use this if you
    /// know the average document length in advance. If you don't, but you have your full corpus
    /// ahead of time, use `with_documents` or `with_corpus` instead.
    ///
    /// If you have neither the full corpus nor a sample of it, you can configure the embedder to
    /// disregard document length by setting `b` to 0.0. In this case, it doesn't matter what
    /// value you pass to `with_avgdl`.
    ///
    /// The average document length is the average number of tokens in a document from your corpus;
    /// if you need access to this value, you can construct an Embedder and call `avgdl` on it.
    pub fn with_avgdl(avgdl: f32) -> SearchEngineBuilder<K, D> {
        SearchEngineBuilder {
            embedder_builder: EmbedderBuilder::<D>::with_avgdl(avgdl),
            documents: Vec::new(),
            document_id_type: PhantomData,
            embedding_dimension_type: PhantomData,
        }
    }

    /// Constructs a new SearchEngineBuilder with the given documents. The search engine will fit
    /// to the given documents, using the given language mode. When you call `build`, the builder
    /// will pre-populate the search engine with the given documents, and set its language mode to
    /// `language_mode`.
    pub fn with_documents(
        language_mode: impl Into<LanguageMode>,
        documents: impl IntoIterator<Item = impl Into<Document<K>>>,
    ) -> SearchEngineBuilder<K, D> {
        let documents = documents.into_iter().map(|d| d.into()).collect::<Vec<_>>();
        SearchEngineBuilder {
            embedder_builder: EmbedderBuilder::<D>::with_fit_to_corpus(
                language_mode,
                &documents
                    .iter()
                    .map(|d| d.contents.as_str())
                    .collect::<Vec<_>>(),
            ),
            documents,
            document_id_type: PhantomData,
            embedding_dimension_type: PhantomData,
        }
    }

    /// Constructs a new SearchEngineBuilder with the corpus. The search engine will fit
    /// to the given corpus, using the given language mode. When you call `build`, the builder
    /// will pre-populate the search engine with the given corpus. This function will
    /// automatically generate u32 ids for each entry in your corpus.
    pub fn with_corpus(
        language_mode: impl Into<LanguageMode>,
        corpus: impl IntoIterator<Item = impl Into<String>>,
    ) -> SearchEngineBuilder<u32, D> {
        let documents = corpus
            .into_iter()
            .enumerate()
            .map(|(id, document)| Document::new(id as u32, document.into()))
            .collect::<Vec<_>>();
        SearchEngineBuilder::<u32, D>::with_documents(language_mode.into(), documents)
    }

    /// Sets the language mode of the embedder.
    pub fn language_mode(mut self, language_mode: LanguageMode) -> SearchEngineBuilder<K, D> {
        self.embedder_builder.language_mode(language_mode);
        self
    }

    /// Sets the k1 parameter of the embedder.
    pub fn k1(mut self, k1: f32) -> SearchEngineBuilder<K, D> {
        self.embedder_builder.k1(k1);
        self
    }

    /// Sets the b parameter of the embedder.
    pub fn b(mut self, b: f32) -> SearchEngineBuilder<K, D> {
        self.embedder_builder.b(b);
        self
    }

    /// Overrides the average document length of the embedder.
    pub fn avgdl(mut self, avgdl: f32) -> SearchEngineBuilder<K, D> {
        self.embedder_builder.avgdl(avgdl);
        self
    }

    /// Builds the search engine.
    pub fn build(self) -> SearchEngine<K, D> {
        let mut search_engine = SearchEngine::<K, D> {
            embedder: self.embedder_builder.build(),
            document_embeddings: HashMap::new(),
            inverted_term_index: HashMap::new(),
            term_frequencies: HashMap::new(),
            documents: HashMap::new(),
        };
        for document in self.documents {
            search_engine.upsert(document);
        }
        search_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_data_loader::tests::{read_recipes, Recipe},
        Language,
    };

    impl From<Recipe> for Document<String> {
        fn from(value: Recipe) -> Self {
            Document::new(value.title, value.recipe)
        }
    }

    fn create_recipe_search_engine(
        language_mode: impl Into<LanguageMode>,
    ) -> SearchEngine<String, u32> {
        let recipes = read_recipes("recipes_en.csv");

        SearchEngineBuilder::with_documents(language_mode, recipes).build()
    }

    #[test]
    fn search_returns_relevant_documents() {
        let corpus = vec!["space station", "bacon and avocado sandwich"];
        let search_engine =
            SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build();

        let results = search_engine.search("sandwich with bacon", 5);
        assert!(results.len() == 1);
        assert!(results[0].document.contents == "bacon and avocado sandwich");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn search_does_not_return_unrelated_documents() {
        let corpus = vec!["space station", "bacon and avocado sandwich"];
        let search_engine =
            SearchEngineBuilder::<u32>::with_corpus(Language::English, corpus).build();

        let results = search_engine.search("maths and computer science", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn it_can_insert_a_document() {
        let mut search_engine = SearchEngineBuilder::<&str>::with_avgdl(2.0).build();
        let document = Document::new("hello world", "bananas and apples");
        let document_id = document.id;

        search_engine.upsert(document.clone());
        let result = search_engine.get(&document_id);

        assert!(result.unwrap() == document);
    }

    #[test]
    fn it_can_remove_a_document() {
        let mut search_engine = SearchEngineBuilder::<usize>::with_avgdl(2.0).build();
        let document = Document::new(123, "bananas and apples");
        let document_id = document.id.clone();

        search_engine.upsert(document);
        search_engine.remove(&document_id);

        assert!(search_engine.get(&document_id).is_none());
    }

    #[test]
    fn it_can_update_a_document() {
        let document_id = "hello_world";
        let document = Document::new(document_id, "bananas and apples");
        let mut search_engine =
            SearchEngineBuilder::<&str>::with_documents(Language::English, vec![document]).build();
        let new_document = Document::new(document_id, "oranges and papayas");

        search_engine.upsert(new_document.clone());
        let result = search_engine.get(&document_id);

        assert!(result.unwrap() == new_document);
    }

    #[test]
    fn handles_empty_input() {
        let mut search_engine = SearchEngineBuilder::<u32>::with_avgdl(2.0).build();
        let document = Document::new(123, "");

        search_engine.upsert(document);

        let results = search_engine.search("bacon sandwich", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn handles_empty_search() {
        let mut search_engine = SearchEngineBuilder::<u32>::with_avgdl(2.0).build();
        let document = Document::new(123, "pencil and paper");

        search_engine.upsert(document);

        let results = search_engine.search("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn it_returns_exact_matches_with_highest_score() {
        let search_engine = create_recipe_search_engine(LanguageMode::Fixed(Language::English));

        let results = search_engine.search(
            "To make guacamole, start by mashing 2 ripe avocados in a bowl.",
            None,
        );

        assert!(!results.is_empty());
        assert_eq!(results[0].document.id, "Guacamole");
    }

    #[test]
    fn it_only_returns_results_containing_query() {
        let search_engine = create_recipe_search_engine(Language::English);

        let results = search_engine.search("vegetable", 5);

        // At least 5 recipes contain the word "vegetable"
        assert_eq!(results.len(), 5);
        assert!(results
            .iter()
            .all(|result| result.document.contents.contains("vegetable")));
    }

    #[test]
    fn it_returns_results_sorted_by_score() {
        let search_engine = create_recipe_search_engine(Language::English);

        let results = search_engine.search("chicken", 1000);

        assert!(!results.is_empty());
        assert!(results
            .windows(2)
            .all(|result_pair| { result_pair[0].score >= result_pair[1].score }));
    }

    #[test]
    fn it_ranks_shorter_documents_higher() {
        let documents = [
            Document {
                id: 0,
                contents: "Correct horse battery staple bacon bacon bacon".to_string(),
            },
            Document {
                id: 1,
                contents: "Correct horse battery staple".to_string(),
            },
        ];
        let search_engine =
            SearchEngineBuilder::<u32>::with_documents(Language::English, documents).build();

        let results = search_engine.search("staple", 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, 1);
        assert_eq!(results[1].document.id, 0);
        assert!(results[0].score > results[1].score);
    }
}
