use crate::{runtime::LLMRuntimeModel, Query};
use std::sync::Arc;

pub struct Mock;

impl LLMRuntimeModel for Mock {
    fn init(&mut self, _: &crate::LLMRuntimeConfig) -> Result<(), crate::Error> {
        Ok(())
    }

    fn execute(
        &mut self,
        message: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> anyhow::Result<(), crate::Error> {
        tracing::debug!("Run Inference");
        // Run inference internally
        self.inference(message, response_tx.clone())?;

        // inference is done, so we have to indicate the end
        response_tx
            .send(crate::Query::End)
            .map_err(|e| crate::Error::StreamError(e.to_string()))?;

        tracing::debug!("Send Query End");

        Ok(())
    }

    fn inference(
        &mut self,
        q: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> Result<(), crate::Error> {
        if let Query::Prompt {
            messages,
            tools: _,
            config,
            chunk_size,
            timestamp: None,
        } = q
        {
            let samples = if let Some(config) = config {
                config.generate_num_samples
            } else {
                20000
            };
            let chunk_size = chunk_size.unwrap_or(100);
            let mut id = 0usize;

            for i in 0..samples {
                // do inference here ...

                // send a chunk of data every chunk size
                if i % chunk_size == 0 {
                    response_tx
                        .send(crate::Query::Chunk {
                            id,
                            data: messages.first().unwrap().content.as_bytes().to_vec(),
                            kind: crate::QueryChunkType::String,
                        })
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                    id = id.saturating_add(1);
                }
            }

            return Ok(());
        }

        Err(crate::Error::StreamError("_".to_string()))
    }
}
