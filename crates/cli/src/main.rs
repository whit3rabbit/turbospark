//! `turbospark-check`: the process entry point for the deterministic front
//! half of the port (Phases 1-3). Reads `argv`, hands the tokens to
//! `turbospark-invocation`, applies its pure exit-status/stream-routing
//! decisions, and — for a validated invocation — prints the resolved
//! request. Matches the ROADMAP's Milestone M1: "a turbospark-check binary
//! that parses real CLI invocations, validates them, and prints the
//! resolved request — no model needed."
//!
//! Resolves the `AGENTS.md`-documented "process surface ownership" question
//! left open for the reserved `turbospark-entrypoint` name: this crate (named
//! `cli`, matching the ROADMAP's own Phase 7 crate list) is that owner. It
//! is intentionally thin — every decision it applies (exit code, which
//! stream gets which text) is computed by `invocation`; this binary only
//! performs the I/O `invocation` is barred from doing itself.
//!
//! Running the actual generation loop (`turbospark-runtime`) needs a loaded
//! model's tokenizer and forward-pass weights. `RealForwardRunner` (macOS/
//! GPU only) now exists, so this binary attempts real generation in all
//! three modes — `--prompt` (raw text, no templating), `--messages-file`
//! (a JSON conversation rendered through the tokenizer's chat template),
//! and `--chat` (the interactive REPL in `chat.rs`) — when `--model`
//! points at a `.gturbo` install `RealForwardRunner` supports with a
//! tokenizer bundled in the same directory; see `generate.rs`. Any
//! unsupported model shape, or a missing tokenizer, prints a clear note
//! and falls back to the resolved-request printout only; on non-macOS this
//! generation attempt is skipped entirely.

use std::io::Write;
use std::process::ExitCode;

use invocation::{exit_status, parse, stream_routing, ExitStatus, InvocationRequest, ParseOutcome};

#[cfg(target_os = "macos")]
mod chat;
#[cfg(target_os = "macos")]
mod generate;

fn main() -> ExitCode {
    let tokens: Vec<String> = std::env::args().skip(1).collect();
    let outcome = parse(&tokens);

    let routing = stream_routing(&outcome);
    if let Some(primary) = &routing.primary {
        println!("{primary}");
    }
    if let Some(diagnostic) = &routing.diagnostic {
        eprintln!("{diagnostic}");
    }
    if let ParseOutcome::Success(request) = &outcome {
        print_resolved_request(request);
        #[cfg(target_os = "macos")]
        generate::try_generate(request);
    }

    match exit_status(&outcome) {
        ExitStatus::Success => ExitCode::from(ExitStatus::Success.code() as u8),
        ExitStatus::InvalidInvocation => ExitCode::from(ExitStatus::InvalidInvocation.code() as u8),
    }
}

fn print_resolved_request(request: &InvocationRequest) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "resolved invocation:");
    let _ = writeln!(out, "  model: {}", request.model);
    let _ = writeln!(out, "  mode: {:?}", request.mode);
    let _ = writeln!(out, "  max_new: {}", request.max_new);
    let _ = writeln!(out, "  max_context: {:?}", request.max_context);
    let _ = writeln!(out, "  temperature: {}", request.temperature);
    let _ = writeln!(out, "  top_k: {}", request.top_k);
    let _ = writeln!(out, "  top_p: {}", request.top_p);
    let _ = writeln!(out, "  repetition_penalty: {}", request.repetition_penalty);
    let _ = writeln!(out, "  seed: {:?}", request.seed);
    let _ = writeln!(out, "  stop: {:?}", request.stop);
    let _ = writeln!(out, "  rdadvise: {:?}", request.rdadvise);
    let _ = writeln!(
        out,
        "  expert_cache_slots: {:?}",
        request.expert_cache_slots
    );
    let _ = writeln!(out, "  prefill_chunk: {:?}", request.prefill_chunk);
    let _ = writeln!(out, "  power_profile: {:?}", request.power_profile);
    let _ = writeln!(
        out,
        "  max_tokens_per_sec: {:?}",
        request.max_tokens_per_sec
    );
    let _ = writeln!(out, "  reasoning: {:?}", request.reasoning);
    let _ = writeln!(out, "  kv_bits: {:?}", request.kv_bits);
    let _ = writeln!(out, "  quiet: {}", request.quiet);
    let _ = writeln!(
        out,
        "note: on macOS, real generation is attempted next against the .gturbo install \
         at --model (see DEVIATIONS.md for scope); on other platforms this only \
         validates and prints the resolved request above."
    );
}
