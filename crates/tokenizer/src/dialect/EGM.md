# tokenizer dialect

## Overview
Tokenizer wrapper and chat-dialect resolution. The dialect decides special-token IDs and the stop set, not chat framing (that comes from the checkpoint's Jinja template).

## Interface
- [mod.rs:ChatDialect] (Gemma, ChatMl, Deepseek, Mistral, Harmony, MuseGlimmer, and more; `tool_call_support`, `tool_call_unsupported_reason`), [mod.rs:ToolCallSupport], [mod.rs:NO_SUCH_TOKEN_ID].
- [mod.rs:MfTokenizer]: `load_from_dir`, `encode`, `decode`, `token_to_id`, `id_to_token`. Consumers also read its public fields: `dialect`, `bos_id`, `eos_id`, `pad_id`, `stop_token_ids`, `vocab_size`, and the optional special-token ids such as `think_end_id`.
- Crate-private: [config.rs:TokenizerConfig], [config.rs:TokenRef] (config and embedded template loading); [resolve.rs:special_token_id], [resolve.rs:required_id], [resolve.rs:Resolved]; per-dialect resolvers [resolvers.rs:resolve_gemma], [resolvers.rs:resolve_harmony], [resolvers.rs:resolve_mistral]; [minimax.rs] (MiniMax uses checkpoint framing, EOS also serves as padding).

## Invariants
- A dialect that lacks a special token resolves its id to `NO_SUCH_TOKEN_ID` rather than failing, so markup flows as text [mod.rs:NO_SUCH_TOKEN_ID].
- Required tokens missing from a vocabulary raise `MissingSpecialToken` [resolve.rs:required_id].
- generation_config EOS lists are capped at 256 ids and filtered to the vocabulary range before joining `stop_token_ids` [config.rs:MAX_EOS_TOKEN_IDS], [mod.rs:MfTokenizer].
- Dialect resolution decides ids and stops only; framing is the checkpoint template's job [mod.rs] (module doc).

## Side Effects
- `load_from_dir` reads `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and the chat-template sidecars (precedence `chat_template.jinja`, then `chat_template.json`, then the config key) from disk [mod.rs:MfTokenizer], [config.rs:TokenizerConfig].
- Sidecar reads are size-capped: `tokenizer_config.json` and the chat-template sidecars at 16 MiB, `generation_config.json` at 1 MiB. An over-cap file is treated as absent, so the template falls through to the next source and possibly the dialect fallback [mod.rs:read_capped], [mod.rs:MAX_SIDECAR_BYTES], [mod.rs:MAX_GENERATION_CONFIG_BYTES].

## Verification
- Run: `cargo test -p turbospark-tokenizer`
