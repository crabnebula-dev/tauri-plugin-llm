use crate::runtime::LLMRuntimeModel;

pub struct Mock;

impl LLMRuntimeModel for Mock {
    fn execute(&mut self, message: crate::LlmMessage) -> Result<crate::LlmMessage, crate::Error> {
        Ok(message)
    }

    fn init(&mut self, config: &crate::llmconfig::LLMRuntimeConfig) -> Result<(), crate::Error> {
        Ok(())
    }

    fn apply_chat_template(&mut self, template: String) {}
}
