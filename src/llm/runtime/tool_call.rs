use crate::ToolCall;

/// Parses tool calls from raw model output text.
///
/// Each model family has its own format for tool calls. This trait
/// decouples the parsing logic from model weights, allowing parsers
/// to be reused across models that share a format.
pub trait ToolCallParser: Send + Sync {
    /// Attempt to parse tool calls from the full decoded model output.
    /// Returns `None` if the output does not contain tool calls.
    fn parse(&self, output: &str) -> Option<Vec<ToolCall>>;
}

/// Llama 3.2 tool call parser.
///
/// The chat template instructs the model to respond with:
/// ```text
/// {"name": "function_name", "parameters": {"arg": "value"}}
/// ```
///
/// Note: Llama uses `"parameters"` (not `"arguments"`).
/// Only single tool calls are supported.
pub struct LlamaToolCallParser;

impl ToolCallParser for LlamaToolCallParser {
    fn parse(&self, output: &str) -> Option<Vec<ToolCall>> {
        let trimmed = output.trim();

        // Find the first JSON object that starts with {"name"
        let start = trimmed.find(r#"{"name""#)?;
        let json_str = &trimmed[start..];

        // Parse the JSON — serde will stop at the end of the first valid object
        // but we need to find the matching closing brace
        let parsed: serde_json::Value = find_first_json_object(json_str)?;

        let name = parsed.get("name")?.as_str()?.to_string();
        let arguments = parsed.get("parameters")?.clone();

        Some(vec![ToolCall::new("call_0".to_string(), name, arguments)])
    }
}

/// Qwen3 tool call parser.
///
/// The chat template instructs the model to respond with XML-wrapped tool calls:
/// ```text
/// <tool_call>
/// {"name": "function_name", "arguments": {"arg": "value"}}
/// </tool_call>
/// ```
///
/// Note: Qwen3 uses `"arguments"` (not `"parameters"`).
/// Multiple tool calls are supported.
pub struct Qwen3ToolCallParser;

impl ToolCallParser for Qwen3ToolCallParser {
    fn parse(&self, output: &str) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();

        for (idx, segment) in output.split("<tool_call>").skip(1).enumerate() {
            let json_str = match segment.split("</tool_call>").next() {
                Some(s) => s.trim(),
                None => continue,
            };

            let parsed: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let name = match parsed.get("name").and_then(|v| v.as_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            let arguments = match parsed.get("arguments") {
                Some(a) => a.clone(),
                None => continue,
            };

            calls.push(ToolCall::new(format!("call_{idx}"), name, arguments));
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }
}

/// Finds the first complete JSON object in a string.
/// Handles nested braces correctly.
fn find_first_json_object(input: &str) -> Option<serde_json::Value> {
    let start = input.find('{')?;
    let bytes = input.as_bytes();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match b {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let json_str = &input[start..start + i + 1];
                    return serde_json::from_str(json_str).ok();
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_parser_clean_json() {
        let parser = LlamaToolCallParser;
        let output = r#"{"name": "get_weather", "parameters": {"location": "Toronto"}}"#;
        let result = parser.parse(output);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_llama_parser_with_prefix() {
        let parser = LlamaToolCallParser;
        // Model output often has leading whitespace or template artifacts
        let output = r#"

{"name": "get_files", "parameters": {"path": "/home/user"}}"#;
        let result = parser.parse(output);
        assert!(result.is_some());
    }

    #[test]
    fn test_llama_parser_plain_text() {
        let parser = LlamaToolCallParser;
        let output = "Hello! How can I help you today?";
        let result = parser.parse(output);
        assert!(result.is_none());
    }

    #[test]
    fn test_qwen3_parser_single_call() {
        let parser = Qwen3ToolCallParser;
        let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Toronto"}}
</tool_call>"#;
        let result = parser.parse(output);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_qwen3_parser_multiple_calls() {
        let parser = Qwen3ToolCallParser;
        let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Toronto"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "EST"}}
</tool_call>"#;
        let result = parser.parse(output);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_qwen3_parser_plain_text() {
        let parser = Qwen3ToolCallParser;
        let output = "The weather in Toronto is sunny.";
        let result = parser.parse(output);
        assert!(result.is_none());
    }

    #[test]
    fn test_llama_parser_with_trailing_text() {
        let parser = LlamaToolCallParser;
        // Real-world output: model produces JSON then continues talking
        let output = r#"{"name": "get_files", "parameters": {"path": "/home"}}

This will list all files in the home directory."#;
        let result = parser.parse(output);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
    }
}
