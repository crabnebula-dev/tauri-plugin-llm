use crate::Error;
use rustpython_vm as vm;
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
        // Use the RustPython-based Jinja2 validator which supports more features than Tera
        if validate_jinja_template(source).is_ok() {
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

    pub fn render(&self, source: &str, input: &str) -> Result<String, Error> {
        match self.kind {
            TemplateType::Jinja => self.render_jinja_template(source, input),
            TemplateType::Unknown => Err(Error::TemplateError("Unknown template type".to_owned())),
        }
    }

    fn render_jinja_template(&self, source: &str, input: &str) -> Result<String, Error> {
        render_template_jinja(source, input)
    }
}

/// Embedded Python code implementing a minimal Jinja2-compatible template engine.
/// This implementation handles: variable substitution, conditionals, loops, filters,
/// whitespace control, type checking, and arithmetic expressions.
const JINJA2_PYTHON_IMPL: &str = r#"
import re
import json

class MiniJinja:
    """Minimal Jinja2-compatible template engine for LLM chat templates."""

    def __init__(self):
        self.filters = {
            'trim': lambda x: str(x).strip(),
            'lower': lambda x: str(x).lower(),
            'upper': lambda x: str(x).upper(),
            'title': lambda x: str(x).title(),
            'length': lambda x: len(x),
            'default': lambda x, d='': x if x else d,
            'first': lambda x: x[0] if x else '',
            'last': lambda x: x[-1] if x else '',
            'join': lambda x, sep='': sep.join(str(i) for i in x),
            'safe': lambda x: str(x),
            'e': lambda x: str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'),
            'escape': lambda x: str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'),
            'string': lambda x: str(x),
            'int': lambda x: int(x),
            'float': lambda x: float(x),
            'list': lambda x: list(x),
            'tojson': lambda x: json.dumps(x),
        }

    def _eval_arithmetic(self, expr, context):
        """Evaluate simple arithmetic expressions like 'loop.index0 - 1'."""
        expr = expr.strip()
        # Handle subtraction
        if ' - ' in expr:
            parts = expr.split(' - ', 1)
            left = self._eval_expr(parts[0].strip(), context)
            right = self._eval_expr(parts[1].strip(), context)
            return left - right
        # Handle addition
        if ' + ' in expr:
            parts = expr.split(' + ', 1)
            left = self._eval_expr(parts[0].strip(), context)
            right = self._eval_expr(parts[1].strip(), context)
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        return None

    def _eval_expr(self, expr, context, _depth=0):
        """Evaluate a simple expression in the given context."""
        expr = expr.strip()
        if not expr:
            return ''

        # Check if expression is a string literal FIRST before any operator checks
        if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
            # It's a string literal - return the content with escape processing
            quote = expr[0]
            content = expr[1:-1]
            # Process escape sequences
            content = content.replace('\\n', '\n').replace('\\t', '\t')
            content = content.replace('\\"', '"').replace("\\'", "'")
            content = content.replace('\\\\', '\\')
            return content

        # Handle filters (e.g., "var | filter" or "var | filter(arg)")
        # But only split on | that's outside of string literals
        def split_on_pipe_outside_strings(s):
            """Split on | that's not inside quotes."""
            parts = []
            current = ''
            in_single = False
            in_double = False
            i = 0
            while i < len(s):
                c = s[i]
                # Skip escaped characters only when inside a string
                if c == '\\' and (in_single or in_double) and i + 1 < len(s):
                    current += c + s[i+1]
                    i += 2
                    continue
                if c == "'" and not in_double:
                    in_single = not in_single
                elif c == '"' and not in_single:
                    in_double = not in_double
                elif c == '|' and not in_single and not in_double:
                    parts.append(current)
                    current = ''
                    i += 1
                    continue
                current += c
                i += 1
            if current:
                parts.append(current)
            return parts

        pipe_parts = split_on_pipe_outside_strings(expr)
        if len(pipe_parts) > 1:
            value = self._eval_expr(pipe_parts[0], context)
            for filter_part in pipe_parts[1:]:
                filter_part = filter_part.strip()
                # Check for filter with arguments
                match = re.match(r'(\w+)\s*\((.*)\)', filter_part)
                if match:
                    filter_name = match.group(1)
                    args_str = match.group(2)
                    if filter_name in self.filters:
                        args = []
                        if args_str:
                            for arg in args_str.split(','):
                                arg = arg.strip().strip('"').strip("'")
                                args.append(arg)
                        value = self.filters[filter_name](value, *args)
                else:
                    if filter_part in self.filters:
                        value = self.filters[filter_part](value)
            return value

        # Handle comparisons and boolean operators BEFORE path resolution
        # But only if the operator is not inside a string literal
        def find_op_outside_strings(s, op):
            """Find operator position that's not inside quotes."""
            in_single = False
            in_double = False
            i = 0
            while i < len(s) - len(op) + 1:
                c = s[i]
                # Skip escaped characters only when inside a string
                if c == '\\' and (in_single or in_double) and i + 1 < len(s):
                    i += 2
                    continue
                if c == "'" and not in_double:
                    in_single = not in_single
                elif c == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if s[i:i+len(op)] == op:
                        return i
                i += 1
            return -1

        for op in ['==', '!=', '>=', '<=', '>', '<']:
            pos = find_op_outside_strings(expr, op)
            if pos != -1:
                left = expr[:pos]
                right = expr[pos+len(op):]
                left_val = self._eval_expr(left.strip(), context)
                right_val = self._eval_expr(right.strip(), context)
                if op == '==': return left_val == right_val
                if op == '!=': return left_val != right_val
                if op == '>=': return left_val >= right_val
                if op == '<=': return left_val <= right_val
                if op == '>': return left_val > right_val
                if op == '<': return left_val < right_val

        if ' and ' in expr:
            parts = expr.split(' and ', 1)
            return self._eval_expr(parts[0], context) and self._eval_expr(parts[1], context)

        if ' or ' in expr:
            parts = expr.split(' or ', 1)
            return self._eval_expr(parts[0], context) or self._eval_expr(parts[1], context)

        if ' in ' in expr:
            left, right = expr.split(' in ', 1)
            left_val = self._eval_expr(left.strip(), context)
            right_val = self._eval_expr(right.strip(), context)
            return left_val in right_val

        # Handle 'is' type checks (including 'is string', 'is defined', etc.)
        if ' is ' in expr:
            left, right = expr.split(' is ', 1)
            left_val = self._eval_expr(left.strip(), context)
            right = right.strip()
            if right == 'defined':
                # Check if the variable path exists in context
                return self._check_defined(left.strip(), context)
            if right == 'string':
                return isinstance(left_val, str)
            if right == 'number':
                return isinstance(left_val, (int, float))
            if right == 'integer':
                return isinstance(left_val, int)
            if right == 'float':
                return isinstance(left_val, float)
            if right == 'mapping' or right == 'dict':
                return isinstance(left_val, dict)
            if right == 'iterable':
                return hasattr(left_val, '__iter__')
            if right == 'sequence':
                return isinstance(left_val, (list, tuple))
            if right == 'none' or right == 'None':
                return left_val is None
            if right == 'true' or right == 'True':
                return left_val is True
            if right == 'false' or right == 'False':
                return left_val is False

        # Handle 'not' prefix
        if expr.startswith('not '):
            return not self._eval_expr(expr[4:], context)

        # Handle parentheses for grouping
        if expr.startswith('(') and expr.endswith(')'):
            return self._eval_expr(expr[1:-1], context)

        # Handle string concatenation with +
        if ' + ' in expr and not any(c.isdigit() for c in expr.split(' + ')[0].strip()[-1:]):
            arith = self._eval_arithmetic(expr, context)
            if arith is not None:
                return arith

        # Handle other literals (strings already handled at the top)
        if expr == 'true' or expr == 'True':
            return True
        if expr == 'false' or expr == 'False':
            return False
        if expr == 'none' or expr == 'None':
            return None
        try:
            return int(expr)
        except ValueError:
            pass
        try:
            return float(expr)
        except ValueError:
            pass

        # Handle list literals
        if expr.startswith('[') and expr.endswith(']'):
            inner = expr[1:-1].strip()
            if not inner:
                return []
            items = []
            depth = 0
            current = ''
            for char in inner:
                if char in '[{':
                    depth += 1
                elif char in ']}':
                    depth -= 1
                if char == ',' and depth == 0:
                    items.append(self._eval_expr(current.strip(), context))
                    current = ''
                else:
                    current += char
            if current.strip():
                items.append(self._eval_expr(current.strip(), context))
            return items

        # Handle attribute/item access (e.g., "obj.attr" or "obj['key']" or "arr[0]")
        if '.' in expr or '[' in expr:
            return self._resolve_path(expr, context)

        # Variable lookup
        return context.get(expr, '')

    def _check_defined(self, path, context):
        """Check if a variable path is defined in the context."""
        try:
            val = self._resolve_path(path, context)
            return val != '' or path in context
        except:
            return False

    def _resolve_path(self, path, context):
        """Resolve a dotted path or bracket notation in context."""
        # Handle expressions inside brackets like arr[loop.index0 - 1]
        bracket_match = re.search(r'\[([^\[\]]+)\]', path)
        while bracket_match:
            inner_expr = bracket_match.group(1)
            # Check if it's an arithmetic expression
            if ' - ' in inner_expr or ' + ' in inner_expr:
                idx = self._eval_arithmetic(inner_expr, context)
                if idx is not None:
                    path = path[:bracket_match.start()] + '.' + str(int(idx)) + path[bracket_match.end():]
            elif inner_expr.isdigit():
                path = path[:bracket_match.start()] + '.' + inner_expr + path[bracket_match.end():]
            elif inner_expr.startswith('"') or inner_expr.startswith("'"):
                key = inner_expr.strip('"').strip("'")
                path = path[:bracket_match.start()] + '.' + key + path[bracket_match.end():]
            else:
                # It's a variable reference
                idx = self._eval_expr(inner_expr, context)
                path = path[:bracket_match.start()] + '.' + str(idx) + path[bracket_match.end():]
            bracket_match = re.search(r'\[([^\[\]]+)\]', path)

        parts = path.split('.')
        value = context
        for part in parts:
            if not part:
                continue
            if isinstance(value, dict):
                value = value.get(part, '')
            elif isinstance(value, (list, tuple)):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError):
                    return ''
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return ''
        return value

    def _find_matching_end(self, template, start_pos, start_tag, end_tag, else_tags=None):
        """Find the matching end tag, handling nesting."""
        if else_tags is None:
            else_tags = []
        depth = 1
        pos = start_pos
        else_positions = []

        while depth > 0 and pos < len(template):
            next_start = template.find('{%', pos)
            if next_start == -1:
                break

            tag_end = template.find('%}', next_start)
            if tag_end == -1:
                break

            tag_content = template[next_start+2:tag_end].strip()
            # Strip whitespace control markers
            if tag_content.startswith('-'):
                tag_content = tag_content[1:].strip()
            if tag_content.endswith('-'):
                tag_content = tag_content[:-1].strip()
            tag_end += 2

            if tag_content.startswith(start_tag.split()[0]):
                depth += 1
            elif tag_content.startswith(end_tag):
                depth -= 1
                if depth == 0:
                    return tag_end, else_positions
            elif depth == 1:
                for else_tag in else_tags:
                    if tag_content.startswith(else_tag):
                        else_positions.append((next_start, tag_end, tag_content))

            pos = tag_end

        return -1, else_positions

    def render(self, template, context):
        """Render the template with the given context."""
        result = []
        pos = 0

        while pos < len(template):
            var_start = template.find('{{', pos)
            block_start = template.find('{%', pos)
            comment_start = template.find('{#', pos)

            next_pos = len(template)
            tag_type = None

            if var_start != -1 and var_start < next_pos:
                next_pos = var_start
                tag_type = 'var'
            if block_start != -1 and block_start < next_pos:
                next_pos = block_start
                tag_type = 'block'
            if comment_start != -1 and comment_start < next_pos:
                next_pos = comment_start
                tag_type = 'comment'

            # Add text before tag
            text_before = template[pos:next_pos]
            result.append(text_before)

            if tag_type is None:
                break

            if tag_type == 'var':
                end = template.find('}}', var_start)
                if end == -1:
                    result.append(template[var_start:])
                    break

                # Check for whitespace control
                raw_expr = template[var_start+2:end]
                trim_before = raw_expr.startswith('-')
                trim_after = raw_expr.endswith('-')
                expr = raw_expr.strip().strip('-').strip()

                if trim_before and result:
                    result[-1] = result[-1].rstrip()

                value = self._eval_expr(expr, context)
                if value is not None:
                    result.append(str(value))

                pos = end + 2
                if trim_after and pos < len(template):
                    # Find next non-whitespace position
                    while pos < len(template) and template[pos] in ' \t\n\r':
                        pos += 1

            elif tag_type == 'comment':
                end = template.find('#}', comment_start)
                if end == -1:
                    result.append(template[comment_start:])
                    break
                pos = end + 2

            elif tag_type == 'block':
                tag_end = template.find('%}', block_start)
                if tag_end == -1:
                    result.append(template[block_start:])
                    break

                raw_content = template[block_start+2:tag_end]
                trim_before = raw_content.startswith('-')
                trim_after = raw_content.endswith('-')
                tag_content = raw_content.strip().strip('-').strip()
                tag_end += 2

                if trim_before and result:
                    result[-1] = result[-1].rstrip()

                if tag_content.startswith('if '):
                    condition = tag_content[3:].strip()
                    end_pos, else_positions = self._find_matching_end(
                        template, tag_end, 'if', 'endif', ['elif', 'else'])

                    if end_pos == -1:
                        pos = tag_end
                        continue

                    branches = []
                    branch_start = tag_end

                    for else_start, else_end, else_content in else_positions:
                        branches.append((condition, template[branch_start:else_start]))
                        if else_content.startswith('elif '):
                            condition = else_content[5:].strip()
                        else:
                            condition = 'true'
                        branch_start = else_end

                    endif_start = template.rfind('{%', tag_end, end_pos)
                    branches.append((condition, template[branch_start:endif_start]))

                    for cond, content in branches:
                        if self._eval_expr(cond, context):
                            result.append(self.render(content, context))
                            break

                    pos = end_pos
                    if trim_after:
                        while pos < len(template) and template[pos] in ' \t\n\r':
                            pos += 1

                elif tag_content.startswith('for '):
                    match = re.match(r'for\s+(\w+)\s+in\s+(.+)', tag_content)
                    if match:
                        var_name = match.group(1)
                        iter_expr = match.group(2).strip()

                        end_pos, _ = self._find_matching_end(template, tag_end, 'for', 'endfor')
                        if end_pos == -1:
                            pos = tag_end
                            continue

                        endfor_start = template.rfind('{%', tag_end, end_pos)
                        loop_content = template[tag_end:endfor_start]

                        items = self._eval_expr(iter_expr, context)
                        if items:
                            loop_results = []
                            items_list = list(items)
                            for i, item in enumerate(items_list):
                                loop_context = context.copy()
                                loop_context[var_name] = item
                                loop_context['loop'] = {
                                    'index': i + 1,
                                    'index0': i,
                                    'first': i == 0,
                                    'last': i == len(items_list) - 1,
                                    'length': len(items_list),
                                }
                                loop_results.append(self.render(loop_content, loop_context))
                            result.append(''.join(loop_results))

                        pos = end_pos
                        if trim_after:
                            while pos < len(template) and template[pos] in ' \t\n\r':
                                pos += 1
                    else:
                        pos = tag_end

                elif tag_content.startswith('set '):
                    match = re.match(r'set\s+(\w+)\s*=\s*(.+)', tag_content)
                    if match:
                        var_name = match.group(1)
                        expr = match.group(2).strip()
                        context[var_name] = self._eval_expr(expr, context)
                    pos = tag_end
                    if trim_after:
                        while pos < len(template) and template[pos] in ' \t\n\r':
                            pos += 1

                elif tag_content.startswith('macro '):
                    end_pos, _ = self._find_matching_end(template, tag_end, 'macro', 'endmacro')
                    pos = end_pos if end_pos != -1 else tag_end

                elif tag_content.startswith('raw'):
                    end_pos = template.find('{% endraw %}', tag_end)
                    if end_pos == -1:
                        end_pos = template.find('{%- endraw -%}', tag_end)
                    if end_pos != -1:
                        result.append(template[tag_end:end_pos])
                        pos = template.find('%}', end_pos) + 2
                    else:
                        pos = tag_end

                else:
                    pos = tag_end
                    if trim_after:
                        while pos < len(template) and template[pos] in ' \t\n\r':
                            pos += 1

        return ''.join(result)


def render(template_str, json_str):
    """Render a Jinja2 template with the given JSON context."""
    context = json.loads(json_str)
    engine = MiniJinja()
    return engine.render(template_str, context)

result = render(TEMPLATE, INPUT_JSON)
"#;

/// Python code to validate a Jinja2 template without rendering.
/// This checks for balanced tags and basic syntax errors.
const JINJA2_VALIDATE_IMPL: &str = r#"
import re

def validate_template(template):
    """Validate a Jinja2 template for syntax errors.

    Returns True if valid, raises an exception if invalid.
    """
    # Track block depth for matching tags
    block_stack = []
    pos = 0

    while pos < len(template):
        # Find next tag
        var_start = template.find('{{', pos)
        block_start = template.find('{%', pos)
        comment_start = template.find('{#', pos)

        # Find earliest tag
        next_pos = len(template)
        tag_type = None

        if var_start != -1 and var_start < next_pos:
            next_pos = var_start
            tag_type = 'var'
        if block_start != -1 and block_start < next_pos:
            next_pos = block_start
            tag_type = 'block'
        if comment_start != -1 and comment_start < next_pos:
            next_pos = comment_start
            tag_type = 'comment'

        if tag_type is None:
            break

        if tag_type == 'var':
            end = template.find('}}', var_start)
            if end == -1:
                raise SyntaxError(f"Unclosed variable tag at position {var_start}")
            pos = end + 2

        elif tag_type == 'comment':
            end = template.find('#}', comment_start)
            if end == -1:
                raise SyntaxError(f"Unclosed comment tag at position {comment_start}")
            pos = end + 2

        elif tag_type == 'block':
            tag_end = template.find('%}', block_start)
            if tag_end == -1:
                raise SyntaxError(f"Unclosed block tag at position {block_start}")

            raw_content = template[block_start+2:tag_end]
            tag_content = raw_content.strip().strip('-').strip()
            pos = tag_end + 2

            # Check for block opening tags
            if tag_content.startswith('if ') or tag_content == 'if':
                block_stack.append('if')
            elif tag_content.startswith('for '):
                block_stack.append('for')
            elif tag_content.startswith('macro '):
                block_stack.append('macro')
            elif tag_content.startswith('block '):
                block_stack.append('block')
            elif tag_content == 'raw':
                block_stack.append('raw')
            # Check for block closing tags
            elif tag_content == 'endif':
                if not block_stack or block_stack[-1] != 'if':
                    raise SyntaxError(f"Unexpected endif at position {block_start}")
                block_stack.pop()
            elif tag_content == 'endfor':
                if not block_stack or block_stack[-1] != 'for':
                    raise SyntaxError(f"Unexpected endfor at position {block_start}")
                block_stack.pop()
            elif tag_content == 'endmacro':
                if not block_stack or block_stack[-1] != 'macro':
                    raise SyntaxError(f"Unexpected endmacro at position {block_start}")
                block_stack.pop()
            elif tag_content == 'endblock':
                if not block_stack or block_stack[-1] != 'block':
                    raise SyntaxError(f"Unexpected endblock at position {block_start}")
                block_stack.pop()
            elif tag_content == 'endraw':
                if not block_stack or block_stack[-1] != 'raw':
                    raise SyntaxError(f"Unexpected endraw at position {block_start}")
                block_stack.pop()
            # elif, else are intermediate tags (valid inside if blocks)
            elif tag_content.startswith('elif ') or tag_content == 'else':
                if not block_stack or block_stack[-1] != 'if':
                    raise SyntaxError(f"Unexpected {tag_content.split()[0]} outside if block at position {block_start}")
            # set, include, extends, etc. are self-closing

    if block_stack:
        raise SyntaxError(f"Unclosed blocks: {', '.join(block_stack)}")

    return True

result = validate_template(TEMPLATE)
"#;

/// Validate a Jinja2 template for syntax errors using RustPython.
/// Returns Ok(()) if valid, Err with details if invalid.
fn validate_jinja_template(template: &str) -> Result<(), Error> {
    rustpython::InterpreterConfig::new()
        .init_stdlib()
        .interpreter()
        .enter(|vm| {
            let scope = vm.new_scope_with_builtins();

            // Set the template as a Python variable
            scope
                .globals
                .set_item("TEMPLATE", vm.new_pyobj(template), vm)
                .map_err(|e| Error::TemplateError(format!("Failed to set TEMPLATE: {:?}", e)))?;

            // Compile and run the validation code
            let code = vm
                .compile(
                    JINJA2_VALIDATE_IMPL,
                    vm::compiler::Mode::Exec,
                    "<jinja2_validate>".to_owned(),
                )
                .map_err(|e| {
                    Error::TemplateError(format!("Failed to compile validation code: {:?}", e))
                })?;

            vm.run_code_obj(code, scope.clone())
                .map_err(|e| Error::TemplateError(format!("Template validation failed: {:?}", e)))?;

            Ok(())
        })
}

/// Render a Jinja2 template using RustPython with embedded Python implementation.
fn render_template_jinja(template: &str, input_json: &str) -> Result<String, Error> {
    rustpython::InterpreterConfig::new()
        .init_stdlib()
        .interpreter()
        .enter(|vm| {
            let scope = vm.new_scope_with_builtins();

            // Set the template and input as Python variables
            scope
                .globals
                .set_item("TEMPLATE", vm.new_pyobj(template), vm)
                .map_err(|e| Error::TemplateError(format!("Failed to set TEMPLATE: {:?}", e)))?;
            scope
                .globals
                .set_item("INPUT_JSON", vm.new_pyobj(input_json), vm)
                .map_err(|e| Error::TemplateError(format!("Failed to set INPUT_JSON: {:?}", e)))?;

            // Compile and run the Python code
            let code = vm
                .compile(
                    JINJA2_PYTHON_IMPL,
                    vm::compiler::Mode::Exec,
                    "<jinja2_impl>".to_owned(),
                )
                .map_err(|e| {
                    Error::TemplateError(format!("Failed to compile Python code: {:?}", e))
                })?;

            vm.run_code_obj(code, scope.clone())
                .map_err(|e| Error::TemplateError(format!("Failed to run Python code: {:?}", e)))?;

            // Get the result from the scope
            let result = scope
                .globals
                .get_item("result", vm)
                .map_err(|e| Error::TemplateError(format!("Failed to get result: {:?}", e)))?;

            // Convert Python string to Rust string
            let result_str: String = result
                .try_into_value(vm)
                .map_err(|e| Error::TemplateError(format!("Failed to convert result: {:?}", e)))?;

            Ok(result_str)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_variable_substitution() {
        let template = "Hello, {{ name }}!";
        let json = r#"{"name": "World"}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_nested_variable_access() {
        let template = "{{ user.name }} is {{ user.age }} years old.";
        let json = r#"{"user": {"name": "Alice", "age": 30}}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "Alice is 30 years old.");
    }

    #[test]
    fn test_for_loop() {
        let template = "{% for item in items %}{{ item }},{% endfor %}";
        let json = r#"{"items": ["a", "b", "c"]}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "a,b,c,");
    }

    #[test]
    fn test_if_condition() {
        let template = "{% if show %}visible{% endif %}";
        let json = r#"{"show": true}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_if_else() {
        let template = "{% if value %}yes{% else %}no{% endif %}";

        let json_true = r#"{"value": true}"#;
        let result_true = render_template_jinja(template, json_true).unwrap();
        assert_eq!(result_true, "yes");

        let json_false = r#"{"value": false}"#;
        let result_false = render_template_jinja(template, json_false).unwrap();
        assert_eq!(result_false, "no");
    }

    #[test]
    fn test_filter_upper() {
        let template = "{{ name | upper }}";
        let json = r#"{"name": "hello"}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "HELLO");
    }

    #[test]
    fn test_filter_trim() {
        let template = "{{ text | trim }}";
        let json = r#"{"text": "  hello  "}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_loop_index() {
        let template = "{% for item in items %}{{ loop.index }}: {{ item }}\n{% endfor %}";
        let json = r#"{"items": ["a", "b"]}"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "1: a\n2: b\n");
    }

    #[test]
    fn test_chat_template_style() {
        let template = r#"{% for message in messages %}{% if message.role == "user" %}User: {{ message.content }}
{% elif message.role == "assistant" %}Assistant: {{ message.content }}
{% endif %}{% endfor %}"#;
        let json = r#"{
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }"#;
        let result = render_template_jinja(template, json).unwrap();
        assert_eq!(result, "User: Hello\nAssistant: Hi there!\n");
    }

    #[test]
    fn test_qwen_chat_template_from_fixtures() {
        // Load the template from the fixture file
        let template_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("tests/fixtures/test_jinja_template.json")
                .expect("Failed to read template fixture"),
        )
        .expect("Failed to parse template JSON");

        let chat_template = template_json["chat_template"]
            .as_str()
            .expect("chat_template not found");

        // Load the input data from the fixture file
        let input_data = std::fs::read_to_string("tests/fixtures/test_jinja_input_data.json")
            .expect("Failed to read input data fixture");

        // Render the template
        let result = render_template_jinja(chat_template, &input_data);

        // The template should render without errors
        assert!(
            result.is_ok(),
            "Template rendering failed: {:?}",
            result.err()
        );

        let rendered = result.unwrap();

        // Verify the output contains expected elements
        assert!(
            rendered.contains("<|im_start|>"),
            "Output should contain im_start token"
        );
        assert!(
            rendered.contains("<|im_end|>"),
            "Output should contain im_end token"
        );
        assert!(rendered.contains("user"), "Output should contain user role");
        assert!(
            rendered.contains("<tools>"),
            "Output should contain tools section since tools are provided"
        );

        println!("Rendered template:\n{}", rendered);
    }

    #[test]
    fn test_validate_jinja_template_valid() {
        let template = "Hello, {{ name }}!";
        assert!(validate_jinja_template(template).is_ok());
    }

    #[test]
    fn test_validate_jinja_template_with_blocks() {
        let template = "{% if show %}visible{% endif %}";
        assert!(validate_jinja_template(template).is_ok());
    }

    #[test]
    fn test_validate_jinja_template_nested() {
        let template = r#"{% for item in items %}{% if item %}{{ item }}{% endif %}{% endfor %}"#;
        assert!(validate_jinja_template(template).is_ok());
    }

    #[test]
    fn test_validate_jinja_template_unclosed_variable() {
        let template = "Hello, {{ name";
        assert!(validate_jinja_template(template).is_err());
    }

    #[test]
    fn test_validate_jinja_template_unclosed_block() {
        let template = "{% if show %}visible";
        assert!(validate_jinja_template(template).is_err());
    }

    #[test]
    fn test_validate_qwen_chat_template() {
        // Load the template from the fixture file
        let template_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("tests/fixtures/test_jinja_template.json")
                .expect("Failed to read template fixture"),
        )
        .expect("Failed to parse template JSON");

        let chat_template = template_json["chat_template"]
            .as_str()
            .expect("chat_template not found");

        // The complex Qwen template should validate successfully
        let result = validate_jinja_template(chat_template);
        assert!(
            result.is_ok(),
            "Qwen template validation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_detect_template_type_jinja() {
        let template = "Hello, {{ name }}!";
        assert!(matches!(
            TemplateType::detect_from_source(template),
            TemplateType::Jinja
        ));
    }

    #[test]
    fn test_detect_template_type_qwen() {
        // Load the template from the fixture file
        let template_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("tests/fixtures/test_jinja_template.json")
                .expect("Failed to read template fixture"),
        )
        .expect("Failed to parse template JSON");

        let chat_template = template_json["chat_template"]
            .as_str()
            .expect("chat_template not found");

        // The complex Qwen template should be detected as Jinja
        let detected = TemplateType::detect_from_source(chat_template);
        assert!(
            matches!(detected, TemplateType::Jinja),
            "Qwen template should be detected as Jinja, got Unknown"
        );
    }
}
