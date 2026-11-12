# handler

## Overview
`/v1/chat/completions`, `/v1/models`, `/health` handlers plus the generation core shared with `/v1/messages` and `/v1/responses`: `plan` turns a request into prompt tokens and a generation config, `exec` runs it (full or streaming) on a blocking task.

## Interface
- [handler/mod.rs:chat_completions], [handler/mod.rs:models], [handler/mod.rs:model_detail], [handler/mod.rs:health].
- [handler/mod.rs:resolve_backend], [handler/mod.rs:resolve_embedding_backend], [handler/mod.rs:request_lease_layer], [handler/mod.rs:RequestLeaseSlot].
- [handler/mod.rs:error_response], [handler/mod.rs:status_for], [handler/mod.rs:queue_full_response], [handler/mod.rs:gen_error_response], [handler/mod.rs:SSE_KEEP_ALIVE].
- [handler/plan.rs:plan], [handler/plan.rs:Planned], [handler/plan.rs:AppState], [handler/plan.rs:build_shaping], [handler/plan.rs:tool_names], [handler/plan.rs:reasoning_effort], [handler/plan.rs:openai_request_warnings], [handler/plan.rs:merge_degradation], [handler/plan.rs:stop_strings], [handler/plan.rs:request_id], [handler/plan.rs:now_unix].
- [handler/exec.rs:run_full], [handler/exec.rs:stream_blocking], [handler/exec.rs:Generated], [handler/exec.rs:Piece], [handler/exec.rs:GenError], [handler/exec.rs:RunControl], [handler/exec.rs:cancelled_before_start].

## Invariants
- Cancellation and the queue permit travel together into detached blocking work ([handler/exec.rs:RunControl]).
- A request cancelled before generation starts reports zero tokens and `StopReason::Cancelled` ([handler/exec.rs:cancelled_before_start]).
- SSE streams send keep-alives every 15 s ([handler/mod.rs:SSE_KEEP_ALIVE]).

## Side Effects
- Runs synchronous CPU/GPU generation on blocking tasks ([handler/exec.rs:run_full], [handler/exec.rs:stream_blocking]).
- May load a model through the registry and hold a use lease for the request ([handler/mod.rs:resolve_backend], [handler/mod.rs:request_lease_layer]).

## Verification
- Run: `cargo test -p turbospark-server`
