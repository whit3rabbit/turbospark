# ACP adapter

`turbospark-acp` lets an editor that speaks the Agent Client Protocol drive a
local TurboSpark session over stdio. Design and decisions:
[OC-12](designs/opencode2-parity/OC-12-acp-adapter.md).

## Status

The protocol core, session registry, file tools and permission gate are
implemented and tested against a fake engine. `turbospark acp --model M` wires
a real local engine (text-only, streaming, cancellable) behind that core. It is
tested with a fake backend and a real child process, but **the real-model path
has not been exercised end to end**: the `#[ignore]`d smoke test
(`crates/cli/tests/acp_agent.rs`) was written and not run, because the host was
not quiet. No editor (Zed or otherwise) has been run against it. The
`turbospark-acp` binary on its own is still the protocol-only build and answers
every prompt with a "no local model" error.

## Transport and safety

* stdio only. The adapter opens no socket, takes no pairing token, performs
  no login and makes no network request. The editor launches it as a child
  process.
* stdout carries protocol messages only; diagnostics go to stderr and never
  contain prompts, file contents or message bodies.
* One JSON message per line, at most 1 MiB. A longer line is discarded with a
  JSON-RPC error and the stream stays aligned.
* Closing stdin (the editor exiting) cancels running turns and exits.

## Pointing an editor at it

```sh
turbospark acp --model qwen3.6:35b      # a tag, alias or install path
```

`turbospark acp` runs the peer binary `turbospark-acp-agent` (built from
`crates/cli`, shipped beside `turbospark`). Register that command as an
external agent in the editor (for Zed, an `agent_servers` entry whose `command`
is `turbospark` with args `["acp", "--model", "..."]`). These instructions are
untested against a real editor.

Flags: `--model M` (required), `--allow-root DIR` (repeatable; session
workspaces must lie inside), `--state-dir DIR`, `--no-persist`.

## The local engine

* The model loads on the first prompt, not at startup, so `initialize` and
  `session/new` answer immediately. A model that cannot load (missing, busy,
  too big) is that prompt's error and is retried on the next prompt.
* One worker thread owns the model; turns from any session queue on it. The
  session is opened through the same path as `turbospark run`
  (`generate::open_session`): catalog alias or path resolution, context and
  memory planning, then the checkpoint's own chat template for every turn.
  Nothing is ever sent as a raw prompt.
* Sampling is the CLI defaults (`turbospark run` without sampling flags). The
  runtime's sampler owns normalization. Speculative decoding is forced off.
* Answer text streams as `agent_message_chunk` per decoded piece. Reasoning
  text is dropped, not shown. History that overflows the context window is
  trimmed oldest first (the same window fit as `--chat`).
* Cancel and editor disconnect set a flag the runtime's decode loop polls
  between tokens and between prefill chunks, so a cancel lands within one token
  or one prefill chunk. stdin EOF drops the engine, which joins the worker and
  unmaps the model.
* **Text only: tools are not wired.** The model is never told about
  `read_file`, `write_file` and the rest, so it never calls them and the
  permission gate is never reached from a real model yet. The runtime has a
  tool-call splitter (`TurnSplitter`, used by the server), but rendering the
  tool schema into the template and feeding tool results back is separate
  work. A tool message in the history is dropped.
* No `--max-context`, `--max-new` or sampling flags yet.

### When another process holds the model

The policy is to refuse, never to wait or share:

* Another `turbospark acp` on the same model directory holds an advisory
  `flock` on `<root>/run/acp-<hash>.lock` (`<root>` is `$TURBOSPARK_HOME`, else
  `~/.turbospark`). The second one's first prompt fails naming the model and
  the lock. A crash releases it.
* A running `turbospark serve` daemon whose `--model` resolves to the same
  directory (read from `run/server.meta`, pid checked live) also refuses.
* **Not covered:** the macOS app takes no lock and leaves no record the CLI
  can read, so the app and an ACP process can both load the same model. That
  needs the app to take the same lock.

## What it speaks (ACP protocol version 1)

`initialize`, `session/new`, `session/prompt`, `session/cancel`,
`session/update` (text chunks and tool-call events), and, towards the editor,
`session/request_permission`, `fs/read_text_file`, `fs/write_text_file`.
Not offered: `authenticate`, `session/load`, terminals, MCP servers, images,
audio, embedded context.

## Tools and permissions

The v1 tool set is `read_file`, `list_dir`, `write_file` and `edit_file`.
There is no shell.

* Reads and listings run without a prompt, inside the workspace root.
* Writes and edits ask the editor (`session/request_permission`) with four
  options: allow once, allow always (this session), reject, reject always
  (this session). Always choices apply to the whole session.
* **Default deny.** Silence for 60 seconds, a cancelled outcome, an
  unrecognized option, an error reply (a client without the method), a
  cancelled turn or a closed connection all refuse the call. The model is told
  `permission denied: 'edit' ... Nothing was executed.` There is no flag that
  approves automatically.
* The workspace root comes from `session/new`. It must be an existing absolute
  directory, not `/` and not the home directory itself, and inside
  `--allow-root` when given. Every tool path resolves against it after
  symlinks; a path that leaves it is refused before any prompt is shown.
* When the editor advertises `fs.readTextFile` / `fs.writeTextFile`, file
  reads and writes go through the editor so unsaved buffers are respected
  (a failure fails the tool; it does not fall back to disk). Otherwise they use
  the disk.

## Sessions are separate from the app's chats

Sessions live in the adapter's own records
(`$TURBOSPARK_STATE_DIR/acp-sessions`, else `$TURBOSPARK_HOME/acp-sessions`,
else `~/.turbospark/acp-sessions`), one JSON file per session, mode 0600 in a
0700 directory. The macOS app does not read them: an editor transcript does not
appear in the app. v1 writes records but cannot resume them.

## Limits

Not a sandbox: a check-then-use race on symlinks is not closed, and the
permission gate is the only control over writes. One engine per adapter
process; two editor windows are two processes, which means two model loads unless they
name the same model (then the second is refused, see above).
