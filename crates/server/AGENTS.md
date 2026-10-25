# turbospark-server

OpenAI, Anthropic, Ollama-compatible, and scripted server surfaces.

## Read first

- [Detailed module guide](../../.claude/docs/modules/server.md)
- [Tool calling](../../docs/TOOL_CALLING.md)
- [Streaming](../../docs/STREAMING.md)
- [CLI reference](../../docs/CLI.md)
- [Audio API](../../docs/AUDIO_API.md) and [OpenAPI specifications](../../docs/openapi/README.md)

## Rules

- The real server uses one runner per process (more only under `--pool-size`,
  which opens N runners of one install behind a single public id) and serves
  the documented OpenAI, Anthropic, and Ollama-compatible routes. Preserve
  each wire format.
- API keys protect configured routes. The server provides no TLS; state that
  boundary in operational docs.
- KV prefix reuse is a process-level opt-in (`--prefix-reuse on`) and is off
  by default: the reused cache is shared by every client and is not
  partitioned by API key or client, so a prefix match against another
  client's prompt is observable in response timing. `--session-slots` above 1
  requires it. The in-process FFI host reaches the same posture through
  `ServerOptions.prefixReuse`.
- Keep scripted and real backends separate. Scripted tests do not establish
  real model behavior.
- Tool guardrails buffer tool-bearing requests when a whole-turn verdict is
  required. Requests without tools must keep streaming behavior.
- Preserve stop reasons, finish events, cancellation, and error responses at
  the protocol boundary.
- Audio routes exist only with an attached `AudioProvider`; keep them out of
  the FFI host's route surface. HTTP model loading takes catalog aliases, never
  paths.
- Validate server model aliases through the same catalog resolution contract as
  the CLI.

## Checks

```sh
cargo test -p turbospark-server
```

Use the real endpoint smoke commands in the detailed guide for route changes.
