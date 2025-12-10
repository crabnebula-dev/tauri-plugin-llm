use crate::Error;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, path::PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexFile {
    metadata: HashMap<String, String>,
    weight_map: HashMap<String, String>,
}

impl TryFrom<PathBuf> for IndexFile {
    type Error = Error;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        let mut file = File::open(value)?;

        let mut index_file: IndexFile = serde_json::from_reader(&mut file)?;

        let inverse = index_file
            .weight_map
            .into_iter()
            .map(|(k, v)| (v, k))
            .collect();

        index_file.weight_map = inverse;

        Ok(index_file)
    }
}

impl IndexFile {
    /// Invert the mapping to get all `*.safetensors` files
    pub fn invert(mut self) -> Self {
        self.weight_map = self.weight_map.into_iter().map(|(k, v)| (v, k)).collect();
        self
    }
}
