//! The checkpoint's own chat template beats the dialect renderer.
//!
//! Two halves. The fixture half reproduces the bug this change fixes, with
//! no model install: `ZephyrTokenizer` carries TinyLlama-1.1B-Chat's exact
//! special-token table (`<unk>`, `<s>`, `</s>` and nothing else) and its own
//! Zephyr template in the pre-`chat_template.jinja` convention. It resolves
//! to [`ChatDialect::Mistral`] -- correctly, since that table is all the
//! dialect probe can see -- and must still render `<|user|>`, not `[INST]`.
//!
//! The real-install half is the DIGEST-SAFETY PROOF for the change. Routing
//! text chat through the installed template moves the prompt bytes of every
//! family that ships one, which would move every frozen quality-gate digest
//! in `crates/bench`. It does not, and this is why: for the four gated
//! families the two renders come out byte-identical. That is a property of
//! those specific templates rather than something the design guarantees, so
//! it is asserted rather than assumed. `#[ignore]`d and env-gated like every
//! other real-install test here; an unset var skips.

use std::path::{Path, PathBuf};

use turbospark_tokenizer::{
    ChatDialect, Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role,
};

/// THE EXACT BYTES the frozen digests are taken over, included from
/// `crates/bench`'s protocol rather than retyped, because retyping it is
/// what made the first version of this guard useless.
///
/// It cost a red gate. The first draft compared a tidy one-line string, on
/// which all four families agreed, and the change shipped -- then
/// `qwen3moe_quality_gate` came back with a changed digest. The real
/// protocol case is a multi-line file ending in `\n`, and the difference
/// only exists for content carrying whitespace: the dialect renderers call
/// `raw.trim()`, and a template trims only if it says so. Gemma's and Qwen
/// 3.6's templates do (`| trim`); Qwen3-30B-A3B's does not, so its user
/// turn gained back the trailing newline its own template never stripped.
/// A comparison fixture chosen for tidiness cannot see the property being
/// asserted.
const PROTOCOL_TURN: &str =
    include_str!("../../bench/prompts/real-generation-v1/short-explanation.txt");

fn fixture(name: &str) -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

/// The dialect probe sees the same three tokens TinyLlama and Mistral both
/// present, so it answers Mistral -- and that answer is not wrong, it is
/// simply not evidence about framing.
#[test]
fn a_zephyr_checkpoint_still_resolves_to_the_mistral_dialect() {
    assert_eq!(fixture("ZephyrTokenizer").dialect, ChatDialect::Mistral);
}

/// THE BUG, in one assertion. Before this change the render below was
/// ` [INST] hi [/INST]`, which TinyLlama echoes back instead of answering.
#[test]
fn a_zephyr_checkpoint_renders_its_own_framing_not_the_dialects() {
    let tok = fixture("ZephyrTokenizer");
    let rendered = tok
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .expect("template renders");
    assert!(rendered.contains("<|user|>"), "rendered = {rendered:?}");
    assert!(rendered.trim_end().ends_with("<|assistant|>"));
    assert!(!rendered.contains("[INST]"), "rendered = {rendered:?}");
}

/// The counterweight to the test above: the dialect renderer has NOT
/// changed and still produces `[INST]` for this tokenizer. So the fix is
/// the routing decision, not an edit to the Mistral renderer -- which
/// stays correct for the checkpoints that really are `[INST]`-framed.
#[test]
fn the_dialect_renderer_is_unchanged_and_still_disagrees() {
    let tok = fixture("ZephyrTokenizer");
    let rendered = tok
        .apply_dialect_chat_template(&[Message::new(Role::User, "hi")])
        .expect("dialect template renders");
    assert_eq!(rendered, " [INST] hi [/INST]");
}

/// A checkpoint shipping no template at all keeps the dialect render, which
/// is what leaves every synthetic install and DeepSeek untouched.
#[test]
fn a_checkpoint_without_a_template_falls_back_to_the_dialect() {
    let tok = fixture("DeepseekTokenizer");
    let messages = [Message::new(Role::User, "hi")];
    assert_eq!(
        tok.apply_chat_template(&messages).unwrap(),
        tok.apply_dialect_chat_template(&messages).unwrap()
    );
}

/// Every install with a frozen quality-gate row, and whether its own
/// template strips surrounding whitespace from message content the way the
/// dialect renderers unconditionally do.
///
/// `trims: true` is the digest-safe case: the two renders are identical and
/// the family's frozen row cannot move. `trims: false` means the template
/// keeps the content verbatim, which is upstream-correct (HF, vLLM and
/// llama.cpp all send what the template says) and DID move that family's
/// row. Recorded per family rather than assumed either way, so a change in
/// either direction reddens here -- cheaply, in seconds -- instead of two
/// minutes into a quality gate.
struct GatedInstall {
    var: &'static str,
    trims: bool,
}

const GATED_INSTALLS: &[GatedInstall] = &[
    GatedInstall {
        var: "TURBOSPARK_GEMMA4_INSTALL_DIR",
        trims: true,
    },
    GatedInstall {
        var: "TURBOSPARK_QWEN36_INSTALL_DIR",
        trims: true,
    },
    // Qwen3-30B-A3B's template is the one that does not. Its frozen row was
    // re-measured for exactly this reason; see the note on the row itself.
    GatedInstall {
        var: "TURBOSPARK_QWEN3MOE_INSTALL_DIR",
        trims: false,
    },
    GatedInstall {
        var: "TURBOSPARK_GEMMA4_IQ_INSTALL_DIR",
        trims: true,
    },
    // Ornith-1.5, all three installs. Their templates carry `| trim`, so they
    // land on Qwen 3.6's side of this table rather than Qwen3-30B-A3B's --
    // read off the files, not inferred from the family, because those two
    // spellings live one architecture apart. Worth having all three rows and
    // not one: the 35B's sidecars come from a DIFFERENT repository per
    // install (the GGUF one's from the BF16 repo, the INT4 one's from the MLX
    // conversion), and that they render identically is an assertion rather
    // than an assumption.
    GatedInstall {
        var: "TURBOSPARK_ORNITH9B_INSTALL_DIR",
        trims: true,
    },
    GatedInstall {
        var: "TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR",
        trims: true,
    },
    GatedInstall {
        var: "TURBOSPARK_ORNITH35B_INSTALL_DIR",
        trims: true,
    },
];

/// Every real install this machine can see, for the reasoning half below.
/// A superset of `GATED_INSTALLS`: that table is about the trim and needs a
/// `trims` answer per family, this one only needs the path.
const REASONING_INSTALLS: &[&str] = &[
    "TURBOSPARK_GEMMA4_INSTALL_DIR",
    "TURBOSPARK_QWEN36_INSTALL_DIR",
    "TURBOSPARK_QWEN3MOE_INSTALL_DIR",
    "TURBOSPARK_QWEN38_INSTALL_DIR",
    "TURBOSPARK_TERNARY_INSTALL_DIR",
    "TURBOSPARK_MUSEGLIMMER_INSTALL_DIR",
    "TURBOSPARK_GPTOSS_INSTALL_DIR",
    "TURBOSPARK_GEMMA4_IQ_INSTALL_DIR",
    "TURBOSPARK_ORNITH9B_INSTALL_DIR",
    "TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR",
    "TURBOSPARK_ORNITH35B_INSTALL_DIR",
];

/// THE DIGEST-SAFETY PROOF FOR `--reasoning`, and the coverage the fixture
/// half structurally cannot give.
///
/// Two claims, per real install, and each is worth a different thing.
///
/// **`Off` renders the frozen bytes.** Every quality-gate digest in
/// `crates/bench` was taken through `apply_chat_template`, which now
/// delegates; if the delegation moved a byte, every row moves. Asserted
/// against the same protocol prose the digests were taken over, not a tidy
/// string -- the mistake that cost a red gate the last time this file was
/// written (see `PROTOCOL_TURN`).
///
/// **A level does something, or says it cannot.** This is the half no
/// fixture reaches: `muse_glimmer` spells the key `reasoning_strength`
/// where Qwen 3.8 and Harmony spell it `reasoning_effort`, and the render
/// path sets BOTH precisely so one table does not have to know which family
/// is which. On a `Level` install the render MUST move; on `ToggleOnly` it
/// must move for the toggle and not for the level; there is no third
/// outcome that is not a silent no-op.
#[test]
#[ignore = "needs a real install; set the TURBOSPARK_*_INSTALL_DIR vars"]
fn a_reasoning_level_reaches_every_installed_template_that_can_express_one() {
    let mut checked = 0usize;
    for var in REASONING_INSTALLS {
        let Ok(dir) = std::env::var(var) else {
            println!("{var}: unset, skipping");
            continue;
        };
        let tok = MfTokenizer::load_from_dir(Path::new(&dir)).expect("install tokenizer loads");
        let turn = [Message::new(Role::User, PROTOCOL_TURN)];

        let off = tok
            .apply_chat_template_with_reasoning(&turn, ReasoningEffort::Off)
            .expect("off must render on every install");
        assert_eq!(
            off,
            tok.apply_chat_template(&turn).unwrap(),
            "{var}: the default render moved, which moves this family's frozen digest"
        );

        let support = tok.reasoning_support();
        // `high` is rejected by Qwen 3.8 and accepted by Harmony and Muse
        // Glimmer; `low` is the one spelling all three take, so it is what
        // a cross-family probe has to use.
        let low = tok.apply_chat_template_with_reasoning(&turn, ReasoningEffort::Low);
        let medium = tok.apply_chat_template_with_reasoning(&turn, ReasoningEffort::Medium);

        match support {
            ReasoningSupport::Level => {
                let low = low.expect("a Level install must render a level");
                let medium = medium.expect("a Level install must render a level");
                assert_ne!(off, low, "{var}: --reasoning low changed nothing");
                assert_ne!(
                    low, medium,
                    "{var}: two DIFFERENT levels rendered the same bytes, so the level \
                     is being dropped even though the template names a key for it"
                );
            }
            ReasoningSupport::ToggleOnly => {
                let low = low.expect("a ToggleOnly install still renders the toggle");
                let medium = medium.expect("a ToggleOnly install still renders the toggle");
                assert_ne!(off, low, "{var}: the thinking toggle changed nothing");
                assert_eq!(
                    low, medium,
                    "{var}: reported ToggleOnly but the level moved the render, so the \
                     support probe is reading the template wrong"
                );
            }
            // A real install always ships a template, so this is a
            // malformed one rather than a family shape.
            ReasoningSupport::None => panic!("{var}: a real install ships a chat template"),
        }
        println!("{var}: dialect {:?}, support {support:?}", tok.dialect);
        checked += 1;
    }
    println!("checked {checked} install(s)");
}

#[test]
#[ignore = "needs a real install; set the TURBOSPARK_*_INSTALL_DIR vars"]
fn the_installed_template_differs_from_the_dialect_by_nothing_but_the_trim() {
    let mut checked = 0usize;
    for install in GATED_INSTALLS {
        let var = install.var;
        let Ok(dir) = std::env::var(var) else {
            println!("{var}: unset, skipping");
            continue;
        };
        let tok = MfTokenizer::load_from_dir(Path::new(&dir)).expect("install tokenizer loads");
        let verbatim = [Message::new(Role::User, PROTOCOL_TURN)];
        let pre_trimmed = [Message::new(Role::User, PROTOCOL_TURN.trim())];

        // The uniform statement, true of every family: hand the installed
        // template content the dialect renderer would have trimmed anyway,
        // and the two agree exactly. That is what says the framing itself
        // is reproduced and the trim is the ONLY axis in play.
        assert_eq!(
            tok.apply_chat_template(&pre_trimmed).unwrap(),
            tok.apply_dialect_chat_template(&verbatim).unwrap(),
            "{var}: the installed template and the dialect renderer disagree \
             about more than surrounding whitespace, so this family's frozen \
             quality-gate row WILL move for a reason that needs explaining \
             before it is re-frozen"
        );

        let agrees_verbatim = tok.apply_chat_template(&verbatim).unwrap()
            == tok.apply_dialect_chat_template(&verbatim).unwrap();
        println!(
            "{var}: dialect {:?}, template trims content: {agrees_verbatim}",
            tok.dialect
        );
        assert_eq!(
            agrees_verbatim, install.trims,
            "{var}: whether this checkpoint's template trims message content \
             has changed, which moves the exact prompt bytes the frozen \
             digests were taken over"
        );
        checked += 1;
    }
    println!("compared {checked} install(s)");
}
