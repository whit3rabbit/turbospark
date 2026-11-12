# cli/src

## Overview
Library-side modules of `turbospark-cli`: the `turbospark-check` generation binary (`main.rs`), interactive chat, the managed server daemon, and agent launchers.

## Interface
- [main.rs] `turbospark-check` entry; declares `chat` and `generate` on macOS only.
- [chat.rs] interactive `--chat` loop (`chat::run`).
- [daemon.rs:start], [daemon.rs:stop], [daemon.rs:restart], [daemon.rs:get_running_pid], [daemon.rs:running_server], [daemon.rs:log_file], [daemon.rs:log_tail], [daemon.rs:find_server_binary], [daemon.rs:is_pid_alive].
- [agent.rs:is_agent], [agent.rs:http_health_ok]: launching external coding agents against the local server.
- Children: [bin](./bin/EGM.md), [front](./front/EGM.md), [generate](./generate/EGM.md), [ui](./ui/EGM.md).

## Invariants
- `start` and `stop` refuse the CWD-relative fallback root [daemon.rs:require_root].
- A daemon lock serializes start [daemon.rs:acquire_daemon_lock].
- Pid and meta files are mode 0600 and the run dir 0700 [daemon.rs:write_private_file], [daemon.rs:create_private_run_dir]. The log file is opened with plain create+append and is not made private [daemon.rs:start].
- `get_running_pid` deletes the pid and meta files when the recorded process is dead, so `status` and `ps` can mutate state [daemon.rs:get_running_pid].
- `start` strips `--api-key VALUE` from the server argv and passes it as `TURBOSPARK_API_KEY` in the child environment [daemon.rs:server_args_and_api_key].

## Side Effects
- Spawns `turbospark-server` and signals it [daemon.rs:start], [daemon.rs:stop].
- Writes pid and meta files under the run dir [daemon.rs:pid_file], [daemon.rs:meta_file]; the log is `logs/server.log`, a sibling of the run dir [daemon.rs:log_file].
- Reads `TURBOSPARK_API_KEY` [daemon.rs:API_KEY_ENV] and `TURBOSPARK_PREFIX_REUSE` (`quiet` silences the prefix-reuse note) [chat.rs:run]; HTTP health probe [agent.rs:http_health_ok].

## Verification
- Run: `cargo test -p turbospark-cli`
