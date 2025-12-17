package main

//#include <stdlib.h>
import "C"
import (
	"bytes"
	"encoding/json"
	"text/template"
	"unsafe"

	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/exec"
)

//export FreeString
func FreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}

//export RenderTemplateString
func RenderTemplateString(templateStr *C.char, jsonData *C.char) *C.char {
	goTemplateStr := C.GoString(templateStr)
	goJsonData := C.GoString(jsonData)

	tmpl, err := template.New("template").Parse(goTemplateStr)
	if err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(goJsonData), &data); err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	return C.CString(buf.String())
}

//export RenderTemplateStringJinja
func RenderTemplateStringJinja(templateStr *C.char, jsonData *C.char) *C.char {
	goTemplateStr := C.GoString(templateStr)
	goJsonData := C.GoString(jsonData)

	template, err := gonja.FromString(goTemplateStr)
	if err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(goJsonData), &data); err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	result := exec.NewContext(data)

	var buf bytes.Buffer
	if err = template.Execute(&buf, result); err != nil {
		return C.CString("ERROR: " + err.Error())
	}

	return C.CString(buf.String())
}

func main() {}
