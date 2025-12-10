use crate::Error;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexFile {
    metadata: HashMap<String, serde_json::Value>,
    weight_map: HashMap<String, String>,
}

impl IndexFile {
    pub fn from_path<P>(value: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
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
    /// Returns the list of all model dependent sharded safetensor files
    pub fn files<P>(&mut self, dir: P) -> Vec<PathBuf>
    where
        P: AsRef<Path>,
    {
        let dir = dir.as_ref().to_path_buf();

        self.weight_map
            .keys()
            .map(|k| {
                let mut dir = dir.clone();
                dir.push(k);
                dir
            })
            .collect()
    }
}

