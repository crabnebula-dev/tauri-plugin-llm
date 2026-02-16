use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Ident, ItemFn, LitBool, LitStr, Token,
};

/// Parsed arguments for the `#[hf_test]` attribute.
///
/// Expected format:
///   `#[hf_test(model = "org/model", cleanup = false, cache_dir = "/path/to/cache")]`
struct HfTestArgs {
    model: String,
    cleanup: bool,
    cache_dir: String,
}

impl Parse for HfTestArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut model = None;
        let mut cleanup = None;
        let mut cache_dir = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if key == "model" {
                let value: LitStr = input.parse()?;
                model = Some(value.value());
            } else if key == "cleanup" {
                let value: LitBool = input.parse()?;
                cleanup = Some(value.value());
            } else if key == "cache_dir" {

                
                let value: LitStr = input.parse()?;
                cache_dir = Some(value.value());
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "unknown argument `{key}`, expected `model`, `cleanup`, or `cache_dir`"
                    ),
                ));
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(HfTestArgs {
            model: model.ok_or_else(|| input.error("missing required argument `model`"))?,
            cleanup: cleanup.ok_or_else(|| input.error("missing required argument `cleanup`"))?,
            cache_dir: cache_dir
                .ok_or_else(|| input.error("missing required argument `cache_dir`"))?,
        })
    }
}

/// Attribute macro that provisions a HuggingFace model for a test.
///
/// # Arguments
///
/// - `model` — HuggingFace model ID (e.g. `"google/gemma-3-1b-it"`)
/// - `cleanup` — whether to remove the model from disk after the test
/// - `cache_dir` — path where the HF cache stores/downloads models
///
/// # Usage
///
/// ```ignore
/// #[hf_test(model = "google/gemma-3-1b-it", cleanup = false, cache_dir = "/Volumes/MLM/huggingface")]
/// fn test_gemma3(config: LLMRuntimeConfig) {
///     let mut runtime = LLMRuntime::from_config(config)?;
///     runtime.run_stream()?;
///     // ...
///     Ok(())
/// }
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
    let cache_dir = &args.cache_dir;

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

    let output = quote! {
        #[test]
        #(#fn_attrs)*
        #fn_vis fn #fn_name() -> std::result::Result<(), Box<dyn std::error::Error>> {
            common::enable_logging();

            let __hf_cache_dir = std::path::PathBuf::from(#cache_dir);
            common::ensure_model_downloaded(#model_id, &__hf_cache_dir)?;

            let #config_ident = tauri_plugin_llm::LLMRuntimeConfig::from_hf_local_cache(
                #model_id,
                std::option::Option::Some(&__hf_cache_dir),
            )?;

            let __hf_guard = common::HfModelGuard::new(
                #model_id,
                __hf_cache_dir,
                #cleanup,
            );

            #fn_body
        }
    };

    output.into()
}
