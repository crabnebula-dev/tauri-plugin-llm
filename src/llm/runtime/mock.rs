use crate::{runtime::LLMRuntimeModel, Query, QueryMessage};
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
        tracing::debug!("Got `Query`: {q:?}");

        if let Query::Prompt {
            messages,
            tools: _,
            config: _,
            chunk_size,
            timestamp,
        } = q
        {
            let mut window_index = 0;

            let mock_message_bytes = match messages.as_slice() {
                [] => "No messages for the Mock runtime have been provided.".as_bytes(),
                [first] => first.content.as_bytes(),
                [_, ..] => {
                    if let Some(QueryMessage { content, .. }) = messages
                        .iter()
                        .find(|m| m.role.eq_ignore_ascii_case("user"))
                    {
                        content.as_bytes()
                    } else {
                        return Err(crate::Error::UnexpectedMessage);
                    }
                }
            };

            let chunk_size = chunk_size.unwrap_or(10);
            let mut id = 0usize;

            tracing::debug!("Simulate Inference");

            for i in 0..mock_message_bytes.len() {
                // send a chunk of data every chunk size
                if i % chunk_size == 0 {
                    tracing::debug!("Sending Chunk");
                    let end = mock_message_bytes.len().min(window_index + chunk_size);
                    response_tx
                        .send(crate::Query::Chunk {
                            id,
                            data: mock_message_bytes[window_index..end].to_vec(),
                            kind: crate::QueryChunkType::String,
                            timestamp,
                        })
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                    window_index += chunk_size;
                    id = id.saturating_add(1);
                }
            }

            tracing::debug!("End Inference");

            return Ok(());
        }

        tracing::warn!("Unknown `Query`: ({:?}), {q:?}", std::mem::discriminant(&q));

        Err(crate::Error::StreamError(
            "Unknown `Query` type".to_string(),
        ))
    }
}
