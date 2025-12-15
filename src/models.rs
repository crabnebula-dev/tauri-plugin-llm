use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum LlmMessage {
    Prompt {
        system: String,
        message: String,
        num_samples: usize,
    },
    Binary {
        system: String,
        data: Vec<u8>,
        num_samples: usize,
    },
    Response {
        error: Option<String>,
        message: String,
    },
    Exit,
    Status,
}
