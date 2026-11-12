# responses

## Overview
The OpenAI `POST /v1/responses` endpoint. Hand-written mapping folds a Responses request onto the shared `ChatCompletionRequest` and re-frames generations as Responses `output[]` items and typed SSE events. Scope is text and tool calling.

## Interface
- [responses/mod.rs:responses] route handler.
- [responses/map.rs:responses_to_chat_request], [responses/map.rs:chat_message], [responses/map.rs:item_to_message], [responses/map.rs:item_text], [responses/map.rs:flat_tool_to_chat_tool], [responses/map.rs:may_produce_reasoning], [responses/map.rs:responses_warnings].
- [responses/sse.rs:build_response], [responses/sse.rs:message_item], [responses/sse.rs:reasoning_item], [responses/sse.rs:function_call_item], [responses/sse.rs:response_status], and event builders [responses/sse.rs:created_event], [responses/sse.rs:completed_event], [responses/sse.rs:failed_event], [responses/sse.rs:text_delta_event], [responses/sse.rs:function_call_events].

## Invariants
- `previous_response_id` is refused with 400; the server is stateless ([responses/mod.rs] module comment).
- `store` is accepted but reported on the `x-anyllm-degradation` header, not honoured ([responses/map.rs:responses_warnings]).
- Reasoning is emitted as an output item only on the non-streaming path ([responses/sse.rs:build_response]); dropped and reported when streaming.

## Side Effects
- Triggers generation through `handler::exec` ([responses/mod.rs:responses]); holds no state of its own.

## Verification
- Run: `cargo test -p turbospark-server`
