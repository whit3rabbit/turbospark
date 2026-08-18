//! Tests the generic Jinja-templated chat rendering against the real,
//! vendored `chat_template.jinja` fixture (a genuine Qwen ChatML template,
//! not a stub), using `minijinja` as the Jinja engine.

use std::path::PathBuf;

use turbospark_tokenizer::{
    render_generic_chat_template, Message, MfTokenizer, ReasoningEffort, Role,
};

fn load() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

#[test]
fn renders_a_single_user_turn_with_generation_prompt() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "hi")];
    let rendered =
        render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off).unwrap();

    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
    assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn renders_system_then_user_turn() {
    let tok = load();
    let messages = vec![
        Message::new(Role::System, "be nice"),
        Message::new(Role::User, "hi"),
    ];
    let rendered =
        render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off).unwrap();

    assert!(rendered.starts_with("<|im_start|>system\nbe nice<|im_end|>\n"));
    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
}

#[test]
fn rejects_empty_messages_via_raise_exception() {
    let tok = load();
    let err = render_generic_chat_template(&tok, &[], &[], true, ReasoningEffort::Off).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("No messages provided"),
        "message = {message}"
    );
}

#[test]
fn encode_generic_tool_chat_tokenizes_the_rendered_text() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "hi")];
    let ids = tok
        .encode_generic_tool_chat(&messages, &[], ReasoningEffort::Off)
        .unwrap();
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids, false);
    assert!(decoded.contains("hi"));
}

#[test]
fn tools_branch_renders_the_tools_system_preamble() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "what's the weather")];
    let tools = vec![turbospark_tokenizer::FunctionDefinition {
        name: "get_weather".to_string(),
        description: "Get the weather".to_string(),
        parameters: turbospark_tokenizer::JsonValue::Object(std::collections::BTreeMap::new()),
    }];
    let rendered =
        render_generic_chat_template(&tok, &messages, &tools, true, ReasoningEffort::Off).unwrap();
    assert!(rendered.contains("# Tools"));
    assert!(rendered.contains("get_weather"));
}

// --- minijinja compatibility shim (conditional keyword arguments) ---
//
// These drive the shim through the REAL load path (a temp fixture directory
// carrying the template as `chat_template.jinja`) rather than by poking a
// private field, so what they assert is that such a template loads and
// renders the way a real install's would.

/// A tokenizer fixture whose `chat_template.jinja` is `template`.
fn tokenizer_with_template(name: &str, template: &str) -> MfTokenizer {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let dir = std::env::temp_dir().join(format!(
        "turbospark-jinja-shim-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("temp fixture dir");
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        let from = src.join(f);
        if from.exists() {
            std::fs::copy(&from, dir.join(f)).expect("copy fixture");
        }
    }
    std::fs::write(dir.join("chat_template.jinja"), template).expect("write template");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn render_with(name: &str, template: &str) -> String {
    let tok = tokenizer_with_template(name, template);
    let messages = vec![Message::new(Role::User, "hi")];
    render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// THE CASE THAT BLOCKED `muse_glimmer`. minijinja 2.22.0 rejects a
/// conditional expression as a keyword argument; Jinja2 accepts it, and the
/// real `Muse-Glimmer-30B-4bit` template uses it
/// (`namespace(name=tcid if tcid else '')`).
///
/// **THIS IS THE TEST TO RUN WHEN CONSIDERING A minijinja UPGRADE.** If a
/// future version parses it natively, `parenthesize_conditional_kwargs` and
/// its call site can be deleted and this must still pass.
#[test]
fn a_conditional_keyword_argument_parses() {
    let out = render_with(
        "cond",
        "{%- set ns = namespace(name=messages[0].content if messages else '') -%}[{{ ns.name }}]",
    );
    assert_eq!(out, "[hi]");
}

/// The conditional's SEMANTICS survive the rewrite -- the false branch is
/// still reachable and still chosen.
///
/// Wrapping in parentheses is Jinja2's own precedence for a keyword
/// argument's value, so this cannot be otherwise; asserted anyway because a
/// rewrite that silently took the true branch always would pass the test
/// above.
#[test]
fn the_rewritten_conditional_still_selects_both_branches() {
    assert_eq!(
        render_with(
            "branch",
            "{%- set ns = namespace(v='yes' if false else 'no') -%}[{{ ns.v }}]"
        ),
        "[no]"
    );
}

/// The shim must not touch template TEXT, which is full of `=` and brackets.
#[test]
fn the_shim_leaves_template_text_alone() {
    assert_eq!(
        render_with(
            "text",
            "a=(b if c else d) <tag k=\"v\"> {{ 'x' }} (n=1 if 2 else 3)"
        ),
        "a=(b if c else d) <tag k=\"v\"> x (n=1 if 2 else 3)"
    );
}

/// Comparisons are not keyword arguments.
#[test]
fn the_shim_does_not_touch_comparisons() {
    for (name, src, want) in [
        ("eq", "{{ 'eq' if (1 == 1) else 'ne' }}", "eq"),
        ("ge", "{{ 'ge' if (2 >= 1) else 'lt' }}", "ge"),
        ("ne", "{{ 'ne' if (1 != 2) else 'eq' }}", "ne"),
    ] {
        assert_eq!(render_with(name, src), want, "{src}");
    }
}

/// A `,` or `)` inside a STRING must not end the argument early.
#[test]
fn the_shim_respects_string_literals() {
    assert_eq!(
        render_with(
            "strlit",
            "{%- set ns = namespace(v='a, b)' if messages else 'z') -%}[{{ ns.v }}]"
        ),
        "[a, b)]"
    );
}

/// Several keyword arguments in one call, only one of them conditional.
#[test]
fn the_shim_handles_one_conditional_among_several_kwargs() {
    assert_eq!(
        render_with(
            "multi",
            "{%- set ns = namespace(a=1, b='y' if messages else 'n', c=[]) -%}[{{ ns.a }}{{ ns.b }}]"
        ),
        "[1y]"
    );
}

/// AND IT MUST BE A NO-OP ON EVERY TEMPLATE THAT DOES NOT NEED IT, which is
/// every other one in this repo. A rewrite that fired spuriously would be
/// changing four shipped families' prompts to fix a fifth.
#[test]
fn the_shim_is_a_no_op_on_templates_that_do_not_need_it() {
    assert_eq!(render_with("plain", "{{ 'ok' }}"), "ok");
    assert_eq!(
        render_with("comment", "{# k=v if x else y #}{{ 'ok' }}"),
        "ok"
    );
    assert_eq!(
        render_with("kwarg", "{%- set ns = namespace(a=1) -%}[{{ ns.a }}]"),
        "[1]"
    );
    assert_eq!(
        render_with(
            "loop",
            "{% for m in messages %}<{{ m.role }}:{{ m.content }}>{% endfor %}"
        ),
        "<user:hi>"
    );
}
