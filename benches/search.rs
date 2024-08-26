use std::{fs::File, io::BufReader};

use bm25::{Document, EmbedderBuilder, Language, LanguageMode, SearchEngine, SearchEngineBuilder};
use csv::Reader;
use divan::Bencher;

use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

// We need to wrap the Recipe struct in a newtype to implement the From trait
#[derive(Clone)]
pub struct BenchmarkRecipe(Recipe);

impl From<BenchmarkRecipe> for Document<String> {
    fn from(value: BenchmarkRecipe) -> Self {
        Document::new(value.0.title, value.0.recipe)
    }
}

#[derive(Clone)]
pub struct Recipe {
    pub title: String,
    pub recipe: String,
}

pub fn read_recipes(recipe_file_name: &str) -> Vec<Recipe> {
    let file_path = format!("data/{}", recipe_file_name);
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let mut csv_reader = Reader::from_reader(reader);

    csv_reader
        .records()
        .map(|r| r.unwrap())
        .map(|r| {
            let title = r.get(0).unwrap().to_string();
            let recipe = r.get(1).unwrap().to_string();
            Recipe { title, recipe }
        })
        .collect()
}

fn create_recipe_search_engine(language_mode: LanguageMode) -> SearchEngine<String, u32> {
    let recipes = read_recipes("recipes_en.csv")
        .into_iter()
        .map(BenchmarkRecipe);

    SearchEngineBuilder::with_documents(language_mode, recipes).build()
}

#[divan::bench(args = [
    LanguageMode::Detect,
    LanguageMode::Fixed(Language::English),
])]
fn recipes_index_creation_language_mode(bencher: Bencher, language_mode: &LanguageMode) {
    let recipes = read_recipes("recipes_en.csv")
        .into_iter()
        .map(BenchmarkRecipe)
        .collect::<Vec<_>>();

    // We calculate the avgdl so that we can populate the search engine from empty
    let avgdl = EmbedderBuilder::<u32>::with_fit_to_corpus(
        language_mode.clone(),
        &recipes
            .iter()
            .map(|recipe| recipe.0.recipe.as_str())
            .collect::<Vec<_>>(),
    )
    .build()
    .avgdl();

    bencher
        .with_inputs(|| {
            (
                SearchEngineBuilder::<String>::with_avgdl(avgdl)
                    .language_mode(language_mode.clone())
                    .build(),
                recipes.clone(),
            )
        })
        .bench_values(|(mut search_engine, recipes)| {
            recipes.into_iter().for_each(|recipe| {
                search_engine.upsert(recipe);
            });
        });
}

#[divan::bench(args = [
    LanguageMode::Detect,
    LanguageMode::Fixed(Language::English),
])]
fn search_language_mode(bencher: Bencher, language_mode: &LanguageMode) {
    let search_engine = create_recipe_search_engine(language_mode.clone());

    bencher.bench(|| search_engine.search("bacon sandwich", 20));
}
