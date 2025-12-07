use candle_core::Device;
use tokenizers::Tokenizer;

use crate::runtime::{LLMRuntimeModel, LLM_Message};

pub struct Qwen3Model {
    pub(crate) streaming: bool,
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) temperature: f32,
    pub(crate) thinking: bool,
}

impl LLMRuntimeModel for Qwen3Model {
    fn execute(&self, message: &LLM_Message) {
        let _ = message;
        tracing::info!("Received Message: {:?}", message);
    }
}
