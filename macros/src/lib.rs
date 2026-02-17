use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Ident, ItemFn, LitBool, LitStr, Token,
};

/// Parsed arguments for the `#[hf_test]` attribute.
///
/// Expected format:
///   `#[hf_test(model = "org/model", cleanup = false, ignore = "reason")]`
struct HfTestArgs {
    model: String,
    cleanup: bool,
    ignore: Option<String>,
}

impl Parse for HfTestArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut model = None;
        let mut cleanup = None;
        let mut ignore = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if key == "model" {
                let value: LitStr = input.parse()?;
                model = Some(value.value());
            } else if key == "cleanup" {
                let value: LitBool = input.parse()?;
                cleanup = Some(value.value());
            } else if key == "ignore" {
                let value: LitStr = input.parse()?;
                ignore = Some(value.value());
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "unknown argument `{key}`, expected `model`, `cleanup`, or `ignore`"
                    ),
                ));
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(HfTestArgs {
            model: model.ok_or_else(|| input.error("missing required argument `model`"))?,
            cleanup: cleanup.unwrap_or(false),
            ignore,
        })
    }
}

/// Attribute macro that provisions a HuggingFace model for a test.
///
/// # Arguments
///
/// - `model` — HuggingFace model ID (e.g. `"google/gemma-3-1b-it"`) (required)
/// - `cleanup` — whether to remove the model from disk after the test (optional, defaults to `false`)
/// - `ignore` — reason string to ignore this test (optional, generates `#[ignore = "reason"]`)
///
/// # Cache Directory Resolution
///
/// The cache directory is resolved in the following order:
/// 1. Try loading from `.env` file using dotenv (`HF_CACHE_DIR`)
/// 2. If not in `.env`, try reading from environment variable (`HF_CACHE_DIR`)
/// 3. If neither exists, use `hf_hub` defaults (typically `~/.cache/huggingface/hub`)
///
/// # Usage
///
/// ```ignore
/// // Basic usage - cache dir from .env or environment
/// #[hf_test(model = "google/gemma-3-1b-it")]
/// fn test_gemma3(config: LLMRuntimeConfig) {
///     let mut runtime = LLMRuntime::from_config(config)?;
///     runtime.run_stream()?;
///     // ...
///     Ok(())
/// }
///
/// // Ignore a test with a reason
/// #[hf_test(model = "google/gemma-3-1b-it", ignore = "Model too large for CI")]
/// fn test_large_model(config: LLMRuntimeConfig) {
///     // ...
///     Ok(())
/// }
///
/// // With cleanup enabled
/// #[hf_test(model = "google/gemma-3-1b-it", cleanup = true)]
/// fn test_with_cleanup(config: LLMRuntimeConfig) {
///     // ...
///     Ok(())
/// }
/// ```
///
/// # Environment Setup
///
/// Create a `.env` file in your project root:
/// ```env
/// HF_CACHE_DIR=/Volumes/MLM/huggingface
/// ```
///
/// Or set the environment variable:
/// ```bash
/// export HF_CACHE_DIR=/Volumes/MLM/huggingface
/// ```
///
/// The function parameter (e.g. `config`) receives the loaded `LLMRuntimeConfig`.
/// The generated test returns `Result<(), Box<dyn std::error::Error>>`.
#[proc_macro_attribute]
pub fn hf_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as HfTestArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let model_id = &args.model;
    let cleanup = args.cleanup;

    // Extract the config parameter name from the function signature.
    // The function must have exactly one parameter: the config binding.
    let config_ident = match input_fn.sig.inputs.first() {
        Some(syn::FnArg::Typed(pat_type)) => match pat_type.pat.as_ref() {
            syn::Pat::Ident(pat_ident) => pat_ident.ident.clone(),
            other => {
                return syn::Error::new_spanned(other, "expected a simple identifier parameter")
                    .to_compile_error()
                    .into();
            }
        },
        Some(syn::FnArg::Receiver(r)) => {
            return syn::Error::new_spanned(r, "hf_test functions cannot have a `self` parameter")
                .to_compile_error()
                .into();
        }
        None => {
            return syn::Error::new_spanned(
                &input_fn.sig,
                "hf_test function must have a config parameter, e.g. `fn test(config: LLMRuntimeConfig)`",
            )
            .to_compile_error()
            .into();
        }
    };

    let fn_name = &input_fn.sig.ident;
    let fn_body = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;

    // Generate #[ignore] attribute if requested
    let ignore_attr = if let Some(reason) = &args.ignore {
        quote! { #[ignore = #reason] }
    } else {
        quote! {}
    };

    let output = quote! {
        #[test]
        #ignore_attr
        #(#fn_attrs)*
        #fn_vis fn #fn_name() -> std::result::Result<(), Box<dyn std::error::Error>> {
            // Load .env file if it exists (using dotenv)
            let _ = dotenv::dotenv();

            // Try to get cache directory from environment (after loading .env)
            let __hf_cache_dir_opt = std::env::var("HF_CACHE_DIR")
                .ok()
                .map(std::path::PathBuf::from);

            // Only ensure download if we have an explicit cache directory
            if let Some(ref cache_dir) = __hf_cache_dir_opt {
                common::ensure_model_downloaded(#model_id, cache_dir)?;
            }

            let #config_ident = tauri_plugin_llm::LLMRuntimeConfig::from_hf_local_cache(
                #model_id,
                __hf_cache_dir_opt.as_ref(),
            )?;

            // Only set up cleanup guard if we have a cache_dir and cleanup is enabled
            let __hf_guard = if #cleanup {
                __hf_cache_dir_opt.map(|cache_dir| common::HfModelGuard::new(
                    #model_id,
                    cache_dir,
                    true,
                ))
            } else {
                std::option::Option::None
            };

            #fn_body
        }
    };

    output.into()
}
