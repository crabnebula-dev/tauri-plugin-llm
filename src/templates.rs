use serde::Serialize;

use crate::Error;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

#[link(name = "gotproc")]
extern "C" {
    fn RenderTemplateString(template: *const c_char, input_json: *const c_char) -> *mut c_char;
    fn FreeString(input: *const c_char);
}

#[derive(Default)]
pub enum TemplateType {
    #[default]
    Jinja,
    Go,
    Unknown,
}

impl TemplateType {
    /// Tries to detect template type.
    ///
    /// For LLMs `jinja` seems to be the common choice for chat templates. However, some models
    /// are using Go Templates. This function accepts a source template and tries to build it using
    /// the provided template engines. If all detection methods fail, [`Self::Unknown`] is being returned.
    ///
    /// Use this function in case the template type is unknown, or requires active detection. Normally, you
    /// wouldn't use this function.
    pub fn detect(source: &str) -> Self {
        if let Ok(inner) = {
            let mut env = minijinja::Environment::new();
            env.add_template("jinja", source)
                .map_err(|e| Error::TemplateError(e.to_string()))
                .map(|_| Self::Jinja)
        } {
            return inner;
        } else if let Ok(inner) = {
            let input_json = serde_json::json!({}).to_string();
            render_template(source, &input_json).map(|_| Self::Go)
        } {
            return inner;
        }

        Self::Unknown
    }
}

#[derive(Default)]
pub struct TemplateProcessor {
    kind: TemplateType,
}

impl TemplateProcessor {
    pub fn new(kind: TemplateType) -> Self {
        Self { kind }
    }

    pub fn with_go_template() -> Self {
        Self {
            kind: TemplateType::Go,
        }
    }

    pub fn with_jinja_template() -> Self {
        Self {
            kind: TemplateType::Jinja,
        }
    }

    pub fn from_raw_template(input: String) -> Result<Self, Error> {
        let kind = TemplateType::detect(&input);

        Ok(Self { kind })
    }

    pub fn from_file<P>(source: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        Self::from_raw_template(std::fs::read_to_string(source)?)
    }

    pub fn render(&self, source: &str, input: &str) -> Result<String, Error> {
        match self.kind {
            TemplateType::Go => self.render_go_template(source, input),
            TemplateType::Jinja => self.render_jinja_template(source, input),
            TemplateType::Unknown => Err(Error::TemplateError("Unknown template type".to_owned())),
        }
    }

    fn render_go_template(&self, source: &str, input: &str) -> Result<String, Error> {
        render_template(source, input)
    }

    fn render_jinja_template<S>(&self, source: &str, input: S) -> Result<String, Error>
    where
        S: Serialize,
    {
        let mut env = minijinja::Environment::new();
        env.add_template("jinja", source)
            .map_err(|e| Error::TemplateError(e.to_string()))?;

        let templ = env
            .get_template("jinja")
            .map_err(|e| Error::TemplateError(e.to_string()))?;

        templ
            .render(input)
            .map_err(|e| Error::TemplateError(e.to_string()))
    }
}

/// Takes a Go template as &str, applies the json variables into it and returns the rendered template
fn render_template(template: &str, input_json: &str) -> Result<String, Error> {
    unsafe {
        let c_template = CString::new(template).map_err(|e| Error::Ffi(e.to_string()))?;
        let c_input_json = CString::new(input_json).map_err(|e| Error::Ffi(e.to_string()))?;
        let result_ptr = RenderTemplateString(c_template.as_ptr(), c_input_json.as_ptr());
        let result = CStr::from_ptr(result_ptr).to_string_lossy().into_owned();
        FreeString(result_ptr as *const c_char);

        // check for errors
        if result.starts_with("ERROR: ") {
            return Err(Error::Ffi(result));
        }

        Ok(result)
    }
}
