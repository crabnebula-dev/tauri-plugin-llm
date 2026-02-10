use crate::Error;
use minijinja::Environment;
use std::path::Path;

#[derive(Default)]
pub enum TemplateType {
    #[default]
    Jinja,
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
    pub fn detect_from_source(source: &str) -> Self {
        let mut env = Environment::new();
        if env.add_template("_detect", source).is_ok() {
            return Self::Jinja;
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

    pub fn with_jinja_template() -> Self {
        Self {
            kind: TemplateType::Jinja,
        }
    }

    pub fn from_raw_template(input: String) -> Result<Self, Error> {
        let kind = TemplateType::detect_from_source(&input);

        Ok(Self { kind })
    }

    pub fn from_file<P>(source: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        Self::from_raw_template(std::fs::read_to_string(source)?)
    }

    /// Renders the template's `source` and applies any `input` values.
    pub fn render(&self, source: &str, input: &str) -> Result<String, Error> {
        match self.kind {
            TemplateType::Jinja => self.render_jinja_template(source, input),
            TemplateType::Unknown => Err(Error::TemplateError("Unknown template type".to_owned())),
        }
    }

    fn render_jinja_template(&self, source: &str, input: &str) -> Result<String, Error> {
        let mut ctx: serde_json::Value =
            serde_json::from_str(input).map_err(|e| Error::TemplateError(e.to_string()))?;

        // CRITICAL: Set add_generation_prompt to true for instruct models
        // This adds the assistant header (e.g., <|start_header_id|>assistant<|end_header_id|>)
        // so the model knows it should generate a response
        // if let Some(obj) = ctx.as_object_mut() {
        //     obj.insert("add_generation_prompt".to_string(), serde_json::json!(true));
        // }

        let mut env = Environment::new();

        // extensions here
        self.set_extensions(&mut env);

        env.add_template("template", source)
            .map_err(|e| Error::TemplateError(e.to_string()))?;

        let tmpl = env
            .get_template("template")
            .map_err(|e| Error::TemplateError(e.to_string()))?;

        tmpl.render(ctx)
            .map_err(|e| Error::TemplateError(e.to_string()))
    }

    /// Sets extensions to minjinia
    fn set_extensions(&self, env: &mut Environment) {
        env.add_filter("tojson", minijinja::filters::tojson);
    }
}
