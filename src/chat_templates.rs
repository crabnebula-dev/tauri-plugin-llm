use crate::Error;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[link(name = "gotproc")]
extern "C" {
    fn RenderTemplateString(template: *const c_char, input_json: *const c_char) -> *mut c_char;
    fn FreeString(input: *const c_char);
}

/// Takes a Go template as &str, applies the json variables into it and returns the rendered template
pub fn render_template(template: &str, input_json: &str) -> Result<String, Error> {
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
