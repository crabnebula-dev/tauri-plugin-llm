use anyhow::{Error as E, Result};
use candle_core::{quantized::gguf_file, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use std::fs::File;
use tokenizers::Tokenizer;

struct TokenStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev: usize,
    current: usize,
}
// modified from https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs
impl TokenStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: vec![],
            prev: 0,
            current: 0,
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(e) => Err(E::msg(format!("Cannot decode : {e}"))),
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev..self.current];
            self.decode(tokens)?
        };

        self.tokens.push(token);

        let text = self.decode(&self.tokens[self.prev..])?;

        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev = self.current;
            self.current = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }
}

#[test]
#[ignore = "This test relies on external dependencies that must exists prior to test execution. Check the code which files are necessary"]
fn test_load_and_execute_model() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;

    println!("Loading tokenzier");
    let tokenizer_file_path = "./models/Qwen3-4B-Instruct-2507-FP8/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_file_path)
        .map_err(|e| E::msg(format!("Could not load tokenizer {}", e)))?;

    let model_file_path = "./models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf";
    let mut model = {
        let mut model_file = File::open(model_file_path)?;
        let model = gguf_file::Content::read(&mut model_file)
            .map_err(|e| E::msg(format!("Could not read model file{}", e)))?;

        Qwen3::from_gguf(model, &mut model_file, &device)?
    };

    println!("Model build");

    let prompt = "Write 'Hello, World'";

    let mut tos = TokenStream::new(tokenizer);

    let tokens = tos
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| E::msg(format!("Error encoding prompt {}", e)))?;

    let tokens = tokens.get_ids();

    // number of tokens to generate
    let to_sample = 1000;
    // let repeat_penalty = 1.;

    let mut logits_processor = {
        let sampling = Sampling::TopK {
            k: 10,
            temperature: 0.9,
        };

        LogitsProcessor::from_sampling(0xBAD666, sampling)
    };

    let mut next_token = {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    };

    let mut all_tokens = vec![];

    all_tokens.push(next_token);

    let eos_token = *tos.tokenizer.get_vocab(true).get("<|im_end|>").unwrap();

    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;

        // skip applying repeat penalty for now

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Some(_) = tos.next_token(next_token)? {
            print!("processing ...\r");
        }

        if next_token == eos_token {
            break;
        }
    }

    println!("Result: {}", tos.decode_all()?);

    Ok(())
}
