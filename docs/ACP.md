# ACP adapter

`turbospark-acp` lets an editor that speaks the Agent Client Protocol drive a
local TurboSpark session over stdio. Design and decisions:
[OC-12](designs/opencode2-parity/OC-12-acp-adapter.md).

## Status

The protocol core, session registry, file tools and permission gate are
implemented and tested against a fake engine. **Real-model generation is not
wired yet**: the shipped binary answers every prompt with an error saying so.
No editor (Zed or otherwise) has been run against it. See the draft's
Implementation Record for the remaining steps.

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

Build `turbospark-acp` (`cargo build -p turbospark-acp`) and register the
binary as an external agent command in the editor. For Zed, an `agent_servers`
entry whose `command` is the binary path. The same shape applies to any ACP
client. These instructions are untested against a real editor.

Flags: `--allow-root DIR` (repeatable; session workspaces must lie inside),
`--state-dir DIR`, `--no-persist`.

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
process; two editor windows are two processes and, once a real model is
wired, two model loads.
