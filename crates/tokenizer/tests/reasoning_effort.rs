//! What `--reasoning` / `reasoning_effort` actually changes in a rendered
//! prompt, and what happens when the checkpoint cannot express it.
//!
//! The fixture under test carries the reasoning-effort gate of the real
//! `Qwen3.8-27B` template (see its own header), so these cases pin the
//! behaviour the flag was built against rather than a shape invented here.

use std::fs;
use std::path::PathBuf;

use turbospark_tokenizer::{Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role};

fn fixture(name: &str) -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn user() -> Vec<Message> {
    vec![Message::new(Role::User, "hi")]
}

fn fixture_with_template(template: &str) -> (MfTokenizer, PathBuf) {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ReasoningEffortTokenizer");
    let target = std::env::temp_dir().join(format!(
        "turbospark-reasoning-template-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&target);
    fs::create_dir(&target).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
        }
    }
    fs::write(target.join("chat_template.jinja"), template).unwrap();
    let tokenizer = MfTokenizer::load_from_dir(&target).unwrap();
    (tokenizer, target)
}

#[test]
fn untrusted_template_probe_has_work_and_output_limits() {
    for template in [
        "{{ 'x' * 1048577 }}",
        "{% for outer in range(1000) %}{% for inner in range(1000) %}x{% endfor %}{% endfor %}",
    ] {
        let (tok, dir) = fixture_with_template(template);
        let error = tok
            .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Low)
            .expect_err("an attacker-controlled template must hit a resource limit");
        assert!(
            error.to_string().contains("rendering") || error.to_string().contains("fuel"),
            "unexpected resource-limit error: {error}"
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

/// THE FIXTURE MUST DISCRIMINATE BEFORE ANYTHING BELOW MEANS ANYTHING.
///
/// A template that rendered identically at every level would pass every
/// other case in this file while proving nothing -- the same trap the
/// near-zero norm weights set for `muse_glimmer`'s parity fixture (AGENTS.md
/// Gotcha 50) and the tied-cut fixture set for `rank_top_k`. Assert the
/// three renders are mutually distinct first.
#[test]
fn the_fixture_renders_differently_at_every_level() {
    let tok = fixture("ReasoningEffortTokenizer");
    let off = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .unwrap();
    let low = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Low)
        .unwrap();
    let xhigh = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::XHigh)
        .unwrap();
    assert_ne!(off, low);
    assert_ne!(low, xhigh);
    assert_ne!(off, xhigh);
}

/// `Off` is the default and it renders the bytes this port rendered before
/// the knob existed: thinking disabled, no effort key, and a generation
/// prompt whose `<think>` block is already closed.
///
/// This is the case every frozen digest in `crates/bench` depends on.
#[test]
fn off_is_byte_identical_to_the_no_argument_render() {
    let tok = fixture("ReasoningEffortTokenizer");
    let implicit = tok.apply_chat_template(&user()).unwrap();
    let explicit = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .unwrap();
    assert_eq!(implicit, explicit);

    assert!(
        implicit.contains("<think>\n\n</think>"),
        "off must take the pre-closed branch: {implicit:?}"
    );
    assert!(
        !implicit.contains("Reasoning effort"),
        "off must not carry an effort instruction: {implicit:?}"
    );
}

/// A LEVEL REACHES THE PROMPT, and it reaches it through the gate rather
/// than around it: the instruction appears only because `enable_thinking`
/// went true with it. A caller that set the effort key while leaving
/// thinking off would render exactly the `Off` bytes above.
#[test]
fn a_level_reaches_the_system_preamble_and_opens_the_think_block() {
    let tok = fixture("ReasoningEffortTokenizer");
    for (level, expected) in [
        (ReasoningEffort::Low, "Reasoning effort is set to low."),
        (
            ReasoningEffort::Medium,
            "Reasoning effort is set to medium.",
        ),
        (ReasoningEffort::XHigh, "Reasoning effort is set to xhigh."),
    ] {
        let rendered = tok
            .apply_chat_template_with_reasoning(&user(), level)
            .unwrap();
        assert!(
            rendered.contains(expected),
            "{level:?} should render {expected:?}, got {rendered:?}"
        );
        assert!(
            !rendered.contains("<think>\n\n</think>"),
            "{level:?} must leave the think block OPEN: {rendered:?}"
        );
    }
}

/// **THE PORT DOES NOT TAKE THE TEMPLATE'S DEFAULT WHEN THINKING IS OFF**,
/// which is the whole reason `xhigh` was never in play here.
///
/// `xhigh` is what this template resolves to for a caller that passes
/// NEITHER key, which is what transformers and mlx-lm do. This port passes
/// `enable_thinking` always, so the gate is closed and the default is never
/// reached. Asserted explicitly because the model card says "xhigh by
/// default" and a reader is entitled to know why that was not what ran.
#[test]
fn the_templates_own_default_is_only_reachable_with_thinking_on() {
    let tok = fixture("ReasoningEffortTokenizer");
    let off = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .unwrap();
    let xhigh = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::XHigh)
        .unwrap();
    assert!(!off.contains("xhigh"));
    assert!(xhigh.contains("xhigh"));
}

/// A level the CHECKPOINT rejects is an error from the template, by name,
/// and is not silently downgraded here.
///
/// `high` is a real level on Harmony and Muse Glimmer and is not one on this
/// family, which is why `ReasoningEffort` accepts the union and lets the
/// template be the judge: a per-family allowlist in this crate would be a
/// second, staler copy of a set the checkpoint already states.
#[test]
fn a_level_this_checkpoint_rejects_surfaces_as_the_templates_own_error() {
    let tok = fixture("ReasoningEffortTokenizer");
    let err = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::High)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Unexpected reasoning effort high"),
        "the checkpoint's own message must reach the caller: {err}"
    );
    assert!(
        err.contains("xhigh (default), medium, and low"),
        "including the set it does accept: {err}"
    );
}

/// The three support shapes, read off the real fixtures rather than
/// asserted in the abstract. This is what lets a caller warn instead of
/// rendering a request it cannot serve.
#[test]
fn reasoning_support_distinguishes_the_three_template_shapes() {
    assert_eq!(
        fixture("ReasoningEffortTokenizer").reasoning_support(),
        ReasoningSupport::Level,
        "a template naming an effort key can express every level"
    );
    assert_eq!(
        fixture("ChatMLTokenizer").reasoning_support(),
        ReasoningSupport::ToggleOnly,
        "Qwen3.5-era templates gate on enable_thinking and have no effort key"
    );
    // No `chat_template.jinja` and no `chat_template` key: nothing to set.
    assert_eq!(
        fixture("DeepseekTokenizer").reasoning_support(),
        ReasoningSupport::None
    );
}

/// A TOGGLE-ONLY TEMPLATE STILL HONOURS THE HALF IT CAN. Thinking goes on;
/// the level goes nowhere. Refusing here would deny a real effect, which is
/// why this case is a caller-side warning rather than an error -- and why
/// [`ReasoningSupport`] has three values rather than a boolean.
#[test]
fn a_toggle_only_template_turns_thinking_on_and_drops_the_level() {
    let tok = fixture("ChatMLTokenizer");
    let off = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .unwrap();
    let low = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Low)
        .unwrap();
    let xhigh = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::XHigh)
        .unwrap();

    assert_ne!(off, low, "the toggle is a real effect");
    assert_eq!(
        low, xhigh,
        "and the LEVEL is not: two levels must render identically here"
    );
    assert!(off.contains("<think>\n\n</think>"));
    assert!(!low.contains("<think>\n\n</think>"));
}

/// A checkpoint with NO template refuses a level rather than dropping it.
///
/// The per-dialect fallback renderers are fixed strings with no knob in
/// them, so serving the request is impossible; the alternative to an error
/// is an answer that simply did not think, with nothing saying so.
#[test]
fn a_template_less_checkpoint_refuses_a_level_and_still_serves_off() {
    let tok = fixture("DeepseekTokenizer");
    assert!(tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .is_ok());

    let err = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Medium)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("ships no chat template"),
        "the refusal must say why: {err}"
    );
}

/// The parse/print round trip, and that `off` is the default value rather
/// than merely a spelling.
#[test]
fn levels_round_trip_and_off_is_the_default() {
    assert_eq!(ReasoningEffort::default(), ReasoningEffort::Off);
    for spelling in ["off", "low", "medium", "high", "xhigh"] {
        assert_eq!(ReasoningEffort::parse(spelling).unwrap().as_str(), spelling);
    }
    assert_eq!(ReasoningEffort::parse("HIGH"), None, "no case folding");
    assert_eq!(ReasoningEffort::parse(""), None);

    // Off inserts NO effort key at all, which is not the same as inserting
    // the string "off": a template resolves its key with `|default(...)`, so
    // an absent key takes the checkpoint's default and a present-but-unknown
    // one raises.
    assert_eq!(ReasoningEffort::Off.level(), None);
    assert_eq!(ReasoningEffort::Low.level(), Some("low"));
    assert!(!ReasoningEffort::Off.enable_thinking());
    assert!(ReasoningEffort::Low.enable_thinking());
}

/// `ALL` IS WHAT THE PROBE SWEEPS, SO A SPELLING MISSING FROM IT IS NEVER
/// OFFERED -- AND NO FIXTURE HERE CAN SEE THAT.
///
/// Found by mutation: dropping `High` from `ALL` left all 14 cases in this
/// file GREEN, because `ReasoningEffortTokenizer` is the only `Level`
/// fixture and it REJECTS `high`. The spelling only matters on Harmony and
/// Muse Glimmer, which accept it and are not fixtures here, so the failure
/// would have been a real checkpoint quietly losing a level it supports.
///
/// Tying the sweep to `parse` rather than restating five names is what makes
/// it hold: a sixth level added to the enum without being added to `ALL`
/// reddens this instead of shipping unprobed.
#[test]
fn the_probe_sweeps_every_spelling_the_parser_accepts() {
    for spelling in ["off", "low", "medium", "high", "xhigh"] {
        let level = ReasoningEffort::parse(spelling).unwrap();
        assert!(
            ReasoningEffort::ALL.contains(&level),
            "{spelling} parses but is never probed, so no template can offer it"
        );
    }
    assert_eq!(
        ReasoningEffort::ALL.len(),
        5,
        "a spelling in ALL that parse does not accept would be probed and \
         then unnameable by a caller"
    );
    assert_eq!(
        ReasoningEffort::ALL[0],
        ReasoningEffort::Off,
        "the sweep reports in this order, and a menu must open at off"
    );
}

/// THE OFFERABLE SET COMES OFF THE TEMPLATE, AND EXCLUDES WHAT IT REJECTS.
///
/// This is the case a GUI's menu is built from. The fixture carries the real
/// `Qwen3.8-27B` gate, so `high` raises here exactly as it does on the real
/// checkpoint (see
/// `a_level_this_checkpoint_rejects_surfaces_as_the_templates_own_error`
/// above) and must not be offered. `xhigh` is present for the same reason,
/// and a family table would have to know both facts.
#[test]
fn the_offered_levels_are_the_templates_own_set() {
    let tok = fixture("ReasoningEffortTokenizer");
    assert_eq!(tok.reasoning_support(), ReasoningSupport::Level);
    assert_eq!(
        tok.accepted_reasoning_levels(),
        vec![
            ReasoningEffort::Off,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::XHigh,
        ],
        "high raises on this template and xhigh does not, which is the whole \
         reason the set is read rather than assumed"
    );
}

/// A checkpoint with NO template can express nothing but `off`.
///
/// `apply_chat_template_with_reasoning` REFUSES a level here rather than
/// dropping it, so the probe simply gets four errors.
#[test]
fn a_template_less_checkpoint_offers_only_off() {
    let tok = fixture("DeepseekTokenizer");
    assert_eq!(tok.reasoning_support(), ReasoningSupport::None);
    assert_eq!(tok.accepted_reasoning_levels(), vec![ReasoningEffort::Off]);
}

/// **THE CASE THAT SAYS "DID IT RAISE" IS THE WRONG QUESTION.**
///
/// These two ship a template that names no reasoning key at all, so every
/// level renders happily and every level renders the SAME BYTES. A probe
/// that only caught errors would report five accepted levels of which four
/// change nothing -- the silent no-op `ReasoningSupport` exists to prevent,
/// re-introduced by the thing meant to refine it. The byte dedupe is what
/// collapses them, and the equality assertion below is the fact it rests on.
#[test]
fn a_template_that_reads_no_reasoning_key_offers_only_off() {
    for name in ["HarmonyTokenizer", "ZephyrTokenizer"] {
        let tok = fixture(name);
        assert_eq!(
            tok.reasoning_support(),
            ReasoningSupport::None,
            "{name} names no reasoning key"
        );

        // The premise: nothing raises, and nothing differs.
        let off = tok
            .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
            .unwrap_or_else(|e| panic!("{name} should render at off: {e}"));
        for level in [ReasoningEffort::Low, ReasoningEffort::High] {
            let rendered = tok
                .apply_chat_template_with_reasoning(&user(), level)
                .unwrap_or_else(|e| panic!("{name} should render at {level:?}: {e}"));
            assert_eq!(
                off, rendered,
                "{name} has no key to read, so {level:?} must be the same prompt"
            );
        }

        assert_eq!(
            tok.accepted_reasoning_levels(),
            vec![ReasoningEffort::Off],
            "{name} renders one distinct prompt, so it offers one level"
        );
    }
}

/// A TOGGLE-ONLY CHECKPOINT OFFERS `off` AND EXACTLY ONE ON-LEVEL.
///
/// Its template reads `enable_thinking` and no effort key, so thinking is a
/// real effect and the LEVEL is dropped: `low` and `xhigh` are two names for
/// one prompt. Offering four of them is offering one thing four times, which
/// is what reads as "Extra High" surviving a model switch it does not apply
/// to.
///
/// The reported spelling is `Low` BY POSITION and carries no meaning as a
/// label. A caller wanting to print something reads `reasoning_support` and
/// says "On".
#[test]
fn a_toggle_only_checkpoint_offers_off_and_exactly_one_on_level() {
    let tok = fixture("ChatMLTokenizer");
    assert_eq!(tok.reasoning_support(), ReasoningSupport::ToggleOnly);

    // The premise: the on-levels are indistinguishable, but off is NOT --
    // without that second half this fixture would prove the collapse and
    // hide a probe that had simply stopped discriminating at all.
    let low = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Low)
        .unwrap();
    let xhigh = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::XHigh)
        .unwrap();
    let off = tok
        .apply_chat_template_with_reasoning(&user(), ReasoningEffort::Off)
        .unwrap();
    assert_eq!(
        low, xhigh,
        "no effort key, so the level cannot change a byte"
    );
    assert_ne!(off, low, "enable_thinking IS read, so off must differ");

    assert_eq!(
        tok.accepted_reasoning_levels(),
        vec![ReasoningEffort::Off, ReasoningEffort::Low]
    );
}

/// The set is never empty and always starts at `off`, on every fixture here.
///
/// A caller builds a menu from this, and an empty one hides a control while
/// a set not starting at `off` offers no way back to not thinking.
#[test]
fn every_fixture_offers_at_least_off_and_offers_it_first() {
    for name in [
        "ChatMLTokenizer",
        "DeepseekTokenizer",
        "GemmaTokenizer",
        "HarmonyTokenizer",
        "Llama3Tokenizer",
        "MuseGlimmerTokenizer",
        "ReasoningEffortTokenizer",
        "ZephyrTokenizer",
    ] {
        let levels = fixture(name).accepted_reasoning_levels();
        assert_eq!(levels[0], ReasoningEffort::Off, "{name} must offer off");

        // Ascending, and no spelling twice.
        let expected: Vec<_> = ReasoningEffort::ALL
            .iter()
            .copied()
            .filter(|l| levels.contains(l))
            .collect();
        assert_eq!(levels, expected, "{name} must report in ALL's order");
    }
}
