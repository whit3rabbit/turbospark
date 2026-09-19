# turbospark-server

OpenAI, Anthropic, Ollama-compatible, and scripted server surfaces.

## Read first

- [Detailed module guide](../../.claude/docs/modules/server.md)
- [Tool calling](../../docs/TOOL_CALLING.md)
- [Streaming](../../docs/STREAMING.md)
- [CLI reference](../../docs/CLI.md)

## Rules

- The real server uses one runner per process and serves the documented OpenAI,
  Anthropic, and Ollama-compatible routes. Preserve each wire format.
- API keys protect configured routes. The server provides no TLS; state that
  boundary in operational docs.
- Keep scripted and real backends separate. Scripted tests do not establish
  real model behavior.
- Tool guardrails buffer tool-bearing requests when a whole-turn verdict is
  required. Requests without tools must keep streaming behavior.
- Preserve stop reasons, finish events, cancellation, and error responses at
  the protocol boundary.
- Validate server model aliases through the same catalog resolution contract as
  the CLI.

## Checks

```sh
cargo test -p turbospark-server
```

Use the real endpoint smoke commands in the detailed guide for route changes.
