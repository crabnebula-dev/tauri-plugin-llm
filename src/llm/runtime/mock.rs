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
        let usage = self.inference(message, response_tx.clone())?;

        tracing::debug!("Inference ended. Got {usage:?}");

        // inference is done, so we have to indicate the end
        response_tx
            .send(crate::Query::End { usage })
            .map_err(|e| crate::Error::StreamError(e.to_string()))?;

        tracing::debug!("Send Query End");

        Ok(())
    }

    fn inference(
        &mut self,
        q: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> Result<Option<crate::TokenUsage>, crate::Error> {
        if let Query::Prompt {
            messages,
            tools: _,
            config: _,
            chunk_size,
            timestamp,
        } = q
        {
            let prompt_tokens = serde_json::to_vec(&messages).map(|v| v.len()).unwrap_or(0);

            let chunk_size = chunk_size.unwrap_or(self.default_chunksize());

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
            let mut chunks = Vec::with_capacity(chunk_size);
            let mut id = 0usize;

            for item in mock_message_bytes {
                chunks.push(*item);

                if chunks.len().eq(&chunk_size) {
                    tracing::debug!("Sending Chunk");

                    let chunk = crate::Query::Chunk {
                        id,
                        data: chunks.to_vec(),
                        kind: crate::QueryChunkType::String,
                        timestamp,
                    };

                    response_tx
                        .send(chunk)
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                    chunks.truncate(0);
                    id += 1;
                }
            }

            if !chunks.is_empty() {
                tracing::debug!("Sending Last Chunk");

                let chunk = crate::Query::Chunk {
                    id: id + 1,
                    data: chunks.to_vec(),
                    kind: crate::QueryChunkType::String,
                    timestamp,
                };

                response_tx
                    .send(chunk)
                    .map_err(|e| crate::Error::StreamError(e.to_string()))?;
            }

            let completion_tokens = mock_message_bytes.len();

            return Ok(Some(crate::TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }));
        }

        Err(crate::Error::StreamError(
            "Unknown `Query` type".to_string(),
        ))
    }
}
