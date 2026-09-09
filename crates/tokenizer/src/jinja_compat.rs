/// Wraps a conditional expression used as a KEYWORD ARGUMENT in parentheses,
/// so minijinja can parse a template Jinja2 accepts.
///
/// **THIS IS A COMPATIBILITY SHIM AND IT IS MEANT TO BE DELETED.** minijinja
/// 2.22.0 (the latest 2.x) rejects `f(k=a if c else d)` with
/// `syntax error: unexpected identifier, expected ","` -- after `k=a` its
/// parser wants a `,` or `)` and finds `if`. Jinja2's grammar for a keyword
/// argument's value is a full expression, conditional included, so real
/// templates use it: `mlx-community/Muse-Glimmer-30B-4bit`'s
/// `chat_template.jinja` has `namespace(name=tcid if tcid else '')`, and
/// without this the whole template fails to parse and `--messages-file`
/// cannot render a prompt at all.
///
/// **WHEN TO REMOVE IT.** If minijinja gains support (no stable release has
/// it as of 2.22.0; 3.0.0-alpha.0 is untested here and deliberately not
/// taken), delete this function and its call site and run
/// `tests/jinja_chat_template.rs` -- `the_shim_is_a_no_op_on_templates_that_
/// do_not_need_it` will still pass, and
/// `a_conditional_keyword_argument_parses` is the one that says whether the
/// engine now handles it directly. Nothing else depends on the rewrite.
///
/// **WHY THIS IS SAFE, and the two ways it could not have been.** The
/// transformation is `k=EXPR` -> `k=(EXPR)`, which is exactly Jinja2's own
/// precedence for a keyword argument value, so it changes no semantics by
/// construction -- it cannot reorder an expression, only make the existing
/// grouping explicit. The two real hazards are both handled structurally
/// rather than by pattern-matching:
///
/// 1. **It must not touch template TEXT.** A chat template is mostly prose
///    and markup, which is full of `=` and parentheses. So the scan only
///    enters `{{ ... }}` and `{% ... %}` blocks and steps over `{# ... #}`
///    comments; everything outside is copied verbatim.
/// 2. **It must not mistake a comparison for an assignment.** `==`, `!=`,
///    `<=`, `>=` are skipped, and a `=` only counts when the character
///    before it is part of an identifier and the character after it is not
///    another `=`.
///
/// String literals are tracked so a `,` or `)` inside `'...'` cannot end an
/// argument early, and nesting is tracked so a call inside a call is one
/// value rather than several.
///
/// Returns a borrowed `Cow` when nothing needed rewriting, which is the case
/// for every other template in this repo.
pub(crate) fn parenthesize_conditional_kwargs(source: &str) -> std::borrow::Cow<'_, str> {
    if !source.contains('=') {
        return std::borrow::Cow::Borrowed(source);
    }
    let bytes = source.as_bytes();
    let mut out = String::new();
    let mut rewrote = false;
    let mut i = 0usize;

    while i < bytes.len() {
        // Only Jinja blocks are scanned; everything else is template text.
        let block = block_at(bytes, i);
        let Some((open_len, close, is_comment)) = block else {
            // **COPY ONE UTF-8 CHARACTER, NOT ONE BYTE.** Templates are
            // mostly prose, and real prose is multibyte: Spark-X2.5's ships
            // a `{#- 0826版本 -#}` comment and fullwidth-bar markers, and
            // pushing `bytes[i] as char` here turned every one of those
            // bytes into a Latin-1 mojibake character in the rendered
            // template. `i` is always a char boundary here (it starts at 0
            // and only ever advances by whole characters or whole blocks),
            // so the slice below is safe.
            let ch_len = source[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            out.push_str(&source[i..i + ch_len]);
            i += ch_len;
            continue;
        };
        let start = i;
        let body_start = i + open_len;
        let Some(body_end) = find_close(source, body_start, close) else {
            // Unterminated block: copy the rest verbatim and let minijinja
            // report it. Guessing at a repair here would turn a clear
            // syntax error into a confusing one.
            out.push_str(&source[start..]);
            i = bytes.len();
            continue;
        };
        out.push_str(&source[start..body_start]);
        if is_comment {
            out.push_str(&source[body_start..body_end + close.len()]);
        } else {
            let (rewritten, changed) = rewrite_expression(&source[body_start..body_end]);
            rewrote |= changed;
            out.push_str(&rewritten);
            out.push_str(close);
        }
        i = body_end + close.len();
    }

    if rewrote {
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(source)
    }
}

/// `(opening delimiter length, closing delimiter, is a comment)` if a Jinja
/// block starts at `i`.
fn block_at(bytes: &[u8], i: usize) -> Option<(usize, &'static str, bool)> {
    if bytes[i] != b'{' || i + 1 >= bytes.len() {
        return None;
    }
    match bytes[i + 1] {
        b'{' => Some((2, "}}", false)),
        b'%' => Some((2, "%}", false)),
        b'#' => Some((2, "#}", true)),
        _ => None,
    }
}

/// Index of `close` at or after `from`, skipping string literals.
fn find_close(source: &str, from: usize, close: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = from;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => {
                if c == b'\'' || c == b'"' {
                    quote = Some(c);
                } else if (c & 0xC0) != 0x80 && source[i..].starts_with(close) {
                    return Some(i);
                }
                // The continuation-byte guard above is what keeps the slice
                // legal: a byte with the 0x80 pattern is mid-character, so
                // stepping onto one means the close delimiter is not here
                // (its first byte is ASCII), and `source[i..]` at a
                // non-boundary would panic. Spark-X2.5's template hit
                // exactly that, scanning for `#}` across its `版本` comment.
            }
        }
        i += 1;
    }
    None
}

/// Rewrites one Jinja expression/statement body.
fn rewrite_expression(body: &str) -> (String, bool) {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut changed = false;
    let mut i = 0usize;
    let mut quote: Option<u8> = None;
    let mut depth = 0usize;

    while i < bytes.len() {
        // **ANY NON-ASCII BYTE STARTS A CHARACTER THAT IS COPIED WHOLE.**
        // Every arm below pushes `c as char`, which turns one byte of a
        // multibyte character into one Latin-1 mojibake character; and the
        // keyword-argument logic itself is ASCII-only (`=`, quotes,
        // brackets), so a multibyte character can simply pass through.
        if bytes[i] >= 0x80 {
            let ch_len = body[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            out.push_str(&body[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        let c = bytes[i];
        if let Some(q) = quote {
            out.push(c as char);
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => {
                quote = Some(c);
                out.push(c as char);
                i += 1;
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                out.push(c as char);
                i += 1;
            }
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                out.push(c as char);
                i += 1;
            }
            b'=' if depth > 0 && is_kwarg_eq(bytes, i) => {
                out.push('=');
                let value_start = i + 1;
                let value_end = kwarg_value_end(bytes, value_start);
                let value = &body[value_start..value_end];
                if contains_top_level_conditional(value) {
                    out.push('(');
                    out.push_str(value.trim_end());
                    out.push(')');
                    // Preserve trailing whitespace outside the parens so the
                    // rewrite is byte-identical apart from the two brackets.
                    out.push_str(&value[value.trim_end().len()..]);
                    changed = true;
                } else {
                    out.push_str(value);
                }
                i = value_end;
            }
            _ => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    (out, changed)
}

/// True when the `=` at `i` is a keyword-argument assignment rather than a
/// comparison operator.
fn is_kwarg_eq(bytes: &[u8], i: usize) -> bool {
    if bytes.get(i + 1) == Some(&b'=') {
        return false; // ==
    }
    let Some(&prev) = bytes.get(i.wrapping_sub(1)) else {
        return false;
    };
    if matches!(prev, b'=' | b'!' | b'<' | b'>') {
        return false; // ==, !=, <=, >=
    }
    prev.is_ascii_alphanumeric() || prev == b'_'
}

/// End of a keyword argument's value: the next `,` or closing bracket at the
/// value's own nesting level, skipping string literals.
fn kwarg_value_end(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    let mut depth = 0usize;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    if depth == 0 {
                        return i;
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => return i,
                _ => {}
            },
        }
        i += 1;
    }
    bytes.len()
}

/// True when `value` holds an ` if ` at its own bracket level and outside
/// string literals -- i.e. a conditional expression rather than one nested in
/// a sub-call that already has its own parentheses.
fn contains_top_level_conditional(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut i = 0usize;
    let mut depth = 0usize;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = depth.saturating_sub(1),
                b'i' if depth == 0 && value[i..].starts_with("if ") => {
                    let before = i.checked_sub(1).map(|p| bytes[p]);
                    if matches!(before, Some(b' ') | Some(b'\t') | Some(b'\n')) {
                        return true;
                    }
                }
                _ => {}
            },
        }
        i += 1;
    }
    false
}
