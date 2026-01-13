use std::sync::Arc;

use crate::{runtime::LLMRuntimeModel, Query};

pub struct Mock;

impl LLMRuntimeModel for Mock {
    fn execute(&mut self, message: crate::Query) -> Result<crate::Query, crate::Error> {
        Ok(message)
    }

    fn init(&mut self, _: &crate::LLMRuntimeConfig) -> Result<(), crate::Error> {
        Ok(())
    }

    fn execute_streaming(
        &mut self,
        message: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::QueryStream>>,
    ) -> anyhow::Result<(), crate::Error> {
        // Run inference internally
        self.inference(message, response_tx.clone())?;

        // inference is done, so we have to indicate the end
        response_tx
            .send(crate::QueryStream::End)
            .map_err(|e| crate::Error::StreamError(e.to_string()))?;

        Ok(())
    }
}

impl Mock {
    fn inference(
        &mut self,
        q: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::QueryStream>>,
    ) -> Result<(), crate::Error> {
        if let Query::Prompt {
            messages,
            tools,
            config,
            chunk_size,
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
                        .send(crate::QueryStream::Chunk {
                            id,
                            data: "hello, world! ".as_bytes().to_vec(),
                            kind: crate::QueryStreamKind::String,
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
