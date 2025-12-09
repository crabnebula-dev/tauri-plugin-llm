use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum LlmMessage {
    Prompt {
        system: String,
        message: String,
        num_samples: usize,
    },
    Response {
        error: Option<String>,
        message: String,
    },
    Exit,
    Status,
}
