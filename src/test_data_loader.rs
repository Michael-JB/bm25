#[cfg(test)]
pub mod tests {
    use std::{fs::File, io::BufReader};

    use csv::Reader;

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

    #[test]
    #[should_panic(expected = "No such file or directory")]
    fn it_should_panic_if_the_file_does_not_exist() {
        read_recipes("non_existent_file.csv");
    }

    #[test]
    fn it_should_read_recipes_from_a_csv_file() {
        let recipes = read_recipes("recipes_en.csv");
        assert_eq!(recipes.len(), 50);
        assert_eq!(recipes[0].title, "French Toast");
    }
}
