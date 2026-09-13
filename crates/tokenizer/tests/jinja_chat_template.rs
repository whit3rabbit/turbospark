//! Tests the generic Jinja-templated chat rendering against the real,
//! vendored `chat_template.jinja` fixture (a genuine Qwen ChatML template,
//! not a stub), using `minijinja` as the Jinja engine.

use std::path::PathBuf;

use turbospark_tokenizer::{
    render_generic_chat_template, ContentPart, Message, MfTokenizer, ReasoningEffort, Role,
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

/// An escape inside a quoted string consumes the complete following Unicode
/// scalar, rather than leaving the scanner on a UTF-8 continuation byte.
#[test]
fn the_shim_handles_escaped_unicode_in_string_literals() {
    assert_eq!(
        render_with("escaped-unicode", "x={{ '\\\u{7248}' }}"),
        "x=\\\u{7248}"
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

// ---------------------------------------------------------------------------
// Multimodal content parts (ROADMAP M-V6)
//
// The fixture's template is the real Qwen one and already carries the
// `render_content` macro, image branch and all -- so these exercise the
// checkpoint's own arm rather than a stub written to agree with this port.
// ---------------------------------------------------------------------------

#[test]
fn an_image_part_renders_the_checkpoints_marker_run() {
    let tok = load();
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::Image,
            ContentPart::Text("what is this?".to_string()),
        ],
    )];
    let rendered =
        render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off).unwrap();

    // ONE placeholder, not N. The expansion is a token-id pass
    // (`vision_io::splice_image_placeholders`), which is what
    // `docs/VISION_PHASE0.md` item 6 established.
    assert_eq!(rendered.matches("<|image_pad|>").count(), 1);
    assert!(
        rendered.contains("<|vision_start|><|image_pad|><|vision_end|>what is this?"),
        "rendered = {rendered}"
    );
}

/// The parts are ORDERED and the template emits the marker where the part
/// sits. Swapping them moves the image after the question, which changes
/// every mRoPE position past it -- fluently.
#[test]
fn the_part_order_decides_where_the_image_lands() {
    let tok = load();
    let render = |parts: Vec<ContentPart>| {
        render_generic_chat_template(
            &tok,
            &[Message::with_parts(Role::User, parts)],
            &[],
            true,
            ReasoningEffort::Off,
        )
        .unwrap()
    };
    let image_first = render(vec![
        ContentPart::Image,
        ContentPart::Text("caption".to_string()),
    ]);
    let text_first = render(vec![
        ContentPart::Text("caption".to_string()),
        ContentPart::Image,
    ]);

    assert!(image_first.contains("<|vision_end|>caption"));
    assert!(text_first.contains("caption<|vision_start|>"));
    assert_ne!(image_first, text_first);
}

/// Several images in one turn each get their own marker run, which is what
/// the splice then expands one at a time.
#[test]
fn two_images_in_one_turn_render_two_marker_runs() {
    let tok = load();
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::Image,
            ContentPart::Text(" and ".to_string()),
            ContentPart::Image,
        ],
    )];
    let rendered =
        render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off).unwrap();
    assert_eq!(rendered.matches("<|image_pad|>").count(), 2);
    assert!(rendered.contains("<|vision_end|> and <|vision_start|>"));
}

/// **THE INVARIANT THAT KEEPS EVERY FROZEN DIGEST WHERE IT IS.** A text-only
/// message takes the template's `content is string` branch exactly as it did
/// before content parts existed, so nothing about a text prompt moved.
#[test]
fn a_text_only_message_renders_identically_through_both_constructors() {
    let tok = load();
    let plain = render_generic_chat_template(
        &tok,
        &[Message::new(Role::User, "hi")],
        &[],
        true,
        ReasoningEffort::Off,
    )
    .unwrap();
    let parts = render_generic_chat_template(
        &tok,
        &[Message::with_parts(
            Role::User,
            vec![ContentPart::Text("hi".to_string())],
        )],
        &[],
        true,
        ReasoningEffort::Off,
    )
    .unwrap();
    assert_eq!(plain, parts);
}

/// The template's own guard, reached through the new path. Worth pinning
/// because it is the one place the port hands a structure the template can
/// REFUSE, and the refusal has to arrive as an error rather than as a panic.
#[test]
fn an_image_in_a_system_message_is_refused_by_the_template() {
    let tok = load();
    let messages = vec![
        Message::with_parts(Role::System, vec![ContentPart::Image]),
        Message::new(Role::User, "hi"),
    ];
    let err = render_generic_chat_template(&tok, &messages, &[], true, ReasoningEffort::Off)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("System message cannot contain images"),
        "{err}"
    );
}

/// The FALLBACK renderer has no vision markers, so it refuses rather than
/// rendering the text alone -- which would produce a prompt with no
/// placeholder for the splice to expand and a model answering about a picture
/// it never saw.
#[test]
fn the_fallback_renderer_refuses_an_image_rather_than_dropping_it() {
    let tok = load();
    let messages = vec![Message::with_parts(Role::User, vec![ContentPart::Image])];
    let err = tok
        .apply_dialect_chat_template(&messages)
        .unwrap_err()
        .to_string();
    assert!(err.contains("carries an image"), "{err}");
}

/// **A MULTIBYTE TEMPLATE SURVIVES THE SHIM'S SCAN.** Spark-X2.5's template
/// carries a `{#- 0826版本 -#}` comment and fullwidth-bar markers, and the
/// shim's byte-walking scan used to panic slicing `source[i..]` at a
/// continuation byte while hunting for the comment's `#}` close -- and, past
/// the panic, pushed text bytes as Latin-1 chars, which would have mojibaked
/// every multibyte character in the file. The scan must copy whole
/// characters, find the close delimiter across multibyte bytes, and leave
/// both intact in the rendered output.
#[test]
fn a_multibyte_template_survives_the_scan_intact() {
    let template =
        "{#- 0826版本 -#}<｜start▁of▁sentence｜>{{ 'ok' }}版本{%- if true %}判断{% endif %}";
    assert_eq!(
        render_with("multibyte", template),
        "<｜start▁of▁sentence｜>ok版本判断"
    );
}

/// The conditional-keyword-argument rewrite still fires when the template
/// ALSO carries multibyte text, which is the combination Spark-X2.5 ships.
#[test]
fn a_conditional_kwarg_rewrites_beside_multibyte_text() {
    assert_eq!(
        render_with(
            "multibyte-kwarg",
            "{#- 版本 -#}{%- set ns = namespace(a=1 if true else 2) -%}[{{ ns.a }}]版本"
        ),
        "[1]版本"
    );
}
