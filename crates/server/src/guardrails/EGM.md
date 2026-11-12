# guardrails

## Overview
Helper submodule of `../guardrails.rs` holding extra tool-call rescue strategies for dialects the `forge_guardrails` crate does not parse (GLM arg_key/arg_value, MiniMax invoke, Kimi K2). [guardrails/extra_formats.rs:rescue] also tries Qwen XML and Gemma parsers after those, which the file header does not list. They are tried before forge's own strategies. Parent logic: [guardrails.rs:run_guarded], [guardrails.rs:inspect].

## Interface
- [guardrails/extra_formats.rs] declared as `mod extra_formats;` in [guardrails.rs:extra_formats]; crate-internal only, no public symbols.
- Parent entry points: [guardrails.rs:GuardrailConfig] (public, re-exported from lib.rs), [guardrails.rs:Verdict], [guardrails.rs:tool_specs], [guardrails.rs:inspect], [guardrails.rs:run_guarded].

## Invariants
- The retry re-renders the prompt through the chat template; it never appends tokens ([guardrails.rs:run_guarded]).
- Rescue runs before validation, so a rescued call is schema-checked like a parsed one ([guardrails.rs:inspect]).
- The retry budget is one ([guardrails.rs:GuardrailConfig]); [guardrails.rs:GuardrailConfig::OFF] disables everything.
- Longcat rescue is deliberately not implemented here because forge's JSON scan already covers it ([guardrails/extra_formats.rs] module comment).

## Side Effects
- None (pure text parsing in this directory). The parent may trigger a second generation ([guardrails.rs:run_guarded]).

## Verification
- Run: `cargo test -p turbospark-server`
