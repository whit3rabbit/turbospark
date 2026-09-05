---
title: Session API (C ABI)
description: Session lifecycle, introspection, prompt and tokenizer helpers, and one-turn generation on the turbospark.h C ABI.
diataxisType: reference
---

<!-- generated: cpp lane, signal: crates/ffi/include/turbospark.h -->

Session half of `crates/ffi/include/turbospark.h`: lifecycle, introspection,
prompt/tokenizer helpers, and generation. Signatures are copied verbatim from
the header; prose under each is the header's own comment. See
[C ABI Contract and Constants](/reference/cpp-abi-contract) for the status
codes, the ownership and threading rules, and the string helpers.

## Session lifecycle

```c
int32_t ts_session_open(const char *model_dir, const char *options_json,
                        TsSession **out);
```

Opens a model and writes a handle to `*out`.

`model_dir` is a path to a `.gturbo` directory OR an installed alias; an
existing directory always wins, so a bare name cannot silently run a different
model than the one named.

`options_json` may be NULL or `"{}"`. Recognised keys, all optional:

- `maxContext`: number \| `"auto"` \| null (default auto)
- `expertCacheSlots`: number \| `"auto"` \| null (default auto; 8/16/24/32)
- `powerProfile`: `"performance"` \| `"balanced"` \| `"efficiency"` \| null.
  null ASKS THE OS, so Low Power Mode selects efficiency. Name one explicitly
  when measuring.
- `maxTokensPerSec`: number \| null
- `speculation`: `"off"` \| `"auto"` \| number \| `"<number>"` \| null
  (default auto). Resolved at OPEN, because that is where the drafter's state
  is allocated. A NAMED block that cannot be served fails this call; `"auto"`
  that cannot be served opens and reports why in `sessionInfo.speculation`.
- `speculativeDrafter`: `"auto"` \| `"mtp"` \| `"dflash"` \| null (default
  auto). `"auto"` ENABLES an MTP head and only REPORTS a DFlash2 one, which is
  measured rather than stylistic: DFlash2 reads 1.43-1.50x on code and math
  and 0.96x throughput at +17.4% J/token on PROSE, so it is opt-in.
- `loadGuard`: `"off"` \| `"relaxed"` \| `"balanced"` \| `"strict"` \| number
  \| null. How much of the machine may be committed. A NUMBER is an absolute
  ceiling in BYTES on what the engine ALLOCATES (slot cache + KV), not on the
  install's size, since a large install streaming from disk is what this
  engine is for. null means `"relaxed"`, which is what this ABI did before the
  key existed and what every published footprint row was measured under.
- `minAutoContext`: number \| null (default 0, meaning no floor). Refuse to
  open when maxContext is automatic and resolves below this many tokens. Says
  NOTHING about an explicit maxContext: a caller naming a number has decided
  how to spend their own machine.
- `steering`: string \| null (path to `.gguf` control vector)
- `steeringMode`: `"ablate"` \| `"add"` \| `"clamp"` \| `"renorm"` \| null
- `steeringScale`: number \| null (default 1.0)
- `steeringLayers`: `"START:END"` \| null (0-based inclusive layer range)
- `steeringTarget`: number \| null (for clamp mode, default 0.0)
- `steeringGate`: number \| null (activation threshold >= 0, default 0.0)

Opening is expensive: it maps gigabytes and compiles Metal pipelines. Open
once and keep the handle.

```c
void ts_session_close(TsSession *s);
```

Closes a session. NULL is a no-op. Must NOT be called while a generation is in
flight on another thread.

```c
void ts_session_cancel(const TsSession *s);
```

Asks the in-flight generation to stop. Safe from any thread; never blocks.

The flag is cleared at the start of every generation, so a Stop pressed
between turns does not cancel the next one. A cancelled run is NOT an error:
`ts_generate` returns TS_OK with stopReason `"cancelled"` and whatever text
had been produced, and the KV cache describes itself honestly, so the
conversation can continue from the partial turn.

## Introspection

```c
int32_t ts_session_info_json(const TsSession *s, char **out);
```

Everything resolved at open, as JSON:

```json
{ "modelPath", "family", "maxContext", "trainedContext",
  "pastTrainedContext", "expertCacheSlots", "vocabSize", "dialect",
  "reasoningSupport", "reasoningLevels",
  "steering": { "active", "mode", "scale", "summary" },
  "speculation": { "block", "drafter", "reason" },
  "vision": { "active", "imageTokenId", "reason" },
  "specialTokens": { "bosId", "eosId", "padId", "endOfTurnId",
                     "stopTokenIds", "thinkStartId", "thinkEndId" } }
```

- `maxContext` and `expertCacheSlots` are the RESOLVED values, never what was
  asked for: under `"auto"` the request carries no number, and the KV cache
  has already been allocated at the resolved one. Neither a throughput nor a
  footprint figure is readable without the slot count beside it.
- `reasoningSupport` is `"level"` \| `"toggleOnly"` \| `"none"` and says what
  KIND of control is meaningful: disable the picker on `"none"`, and on
  `"toggleOnly"` present it as an on/off switch, since asking for a level
  there turns thinking on and sets no level.
- `reasoningLevels` is WHAT TO PUT IN THAT CONTROL, ascending, always starting
  `"off"`. Build the menu from it and from nothing else. The set belongs to
  the checkpoint and cannot be derived from the family: Qwen 3.8 answers
  `["off","low","medium","xhigh"]` and RAISES on `"high"`, while gpt-oss and
  Muse Glimmer answer `["off","low","medium","high"]`. Offering a level absent
  from here fails the turn with the template's own error message. Levels that
  render the same prompt are already collapsed, so a `"toggleOnly"` checkpoint
  answers exactly two entries and a `"none"` one answers `["off"]`; that
  second entry is `"low"` by POSITION and is not a label to print.
- `steering.active` is true when a control vector is loaded on this session.
  `steering.summary` holds a human-readable one-line description of the edit.
- `speculation.block` is the resolved block size, or null when this session
  does not draft ahead; that null IS the "is it on" test, and `drafter`
  (`"mtp"` \| `"dflash"`) is non-null exactly when `block` is.
  `speculation.reason` says why it is off when a caller might have expected
  otherwise, and is null both when they asked for `"off"` and when it is on.
- `vision.active` MEANS "AN IMAGE WOULD BE SERVED", NOT "A TOWER EXISTS", and
  a host must gate its attach control on it rather than on the family name.
  An install can carry a tower and still refuse every image: the pixel budget
  is read from the checkpoint's own `preprocessor_config.json` and has no
  default worth falling back to, so an install streamed without that sidecar
  reports active false with the reason in `vision.reason`. `reason` is
  non-null exactly in that case, which is the only one a caller can act on.
- A NON-NULL `speculation.block` IS A STATEMENT ABOUT THE SESSION, NOT THE
  NEXT TURN. Acceptance is argmax(target) == proposal, exact only at
  temperature 0, so a sampled turn decodes sequentially whatever this says:
  silently, and by design, since this binding's own sampling default is 0.2,
  so a per-turn warning would fire on the normal case. Send temperature 0 to
  speculate.

```c
int32_t ts_session_phases_json(const TsSession *s, char **out);
```

The decode phase breakdown, as JSON (see `wire::PhaseReport`). Cumulative over
every forward pass the session has served, PREFILL INCLUDED, so a per-call
number is an average over the whole context range rather than a number at the
current context. The buckets also cover the inside of the forward pass only:
the sampler and the detokenizer run after it returns and appear in none of
them.

```c
uint64_t ts_peak_footprint_bytes(void);
```

This process's peak physical footprint in bytes, or 0 where unavailable.

The same mach counter every published memory figure for this engine is
measured with. What it counts differs by install shape: on a streamed MoE
model the mapped weights ARE counted, on a dense one they are not, so read it
beside the context window rather than comparing across families.

```c
int32_t ts_system_info_json(char **out);
```

Hardware and power telemetry for this machine, as JSON:
`{ "physicalMemoryBytes", "recommendedWorkingSetBytes", "chip",
"lowPowerMode", "thermalLevel", "memoryPressure" }`.

`"memoryPressure"` is `"normal"` \| `"warn"` \| `"critical"`, the kernel's own
verdict. POLLED HERE UNCONDITIONALLY, unlike the decode loop's own probe,
which follows the power profile: a session running the default
`"performance"` profile watches nothing, so the `"peakMemoryPressure"` on a
generation result is the ABSENCE of a reading rather than a report that memory
was fine. A status panel should read this call.

## Prompt, tokenizer and window helpers

All reasoning arguments below take `"off"|"low"|"medium"|"high"|"xhigh"`, or
NULL for `"off"`.

```c
int32_t ts_session_count_tokens(const TsSession *s, const char *messages_json,
                                const char *reasoning, uint32_t *out_count);
```

Evaluates the exact prompt token count of `messages_json` using the session's
chat template and tokenizer, without running generation. Writes token count to
`*out_count`.

```c
int32_t ts_session_render_prompt(const TsSession *s, const char *messages_json,
                                 const char *reasoning, char **out_prompt);
```

Formats a conversation transcript into raw prompt text using the session's
chat template and reasoning effort setting. `messages_json` is
`[{"role":"user","content":"..."}]`. Writes formatted prompt string to
`*out_prompt`.

```c
int32_t ts_session_tokenize_json(const TsSession *s, const char *text,
                                 bool add_special, char **out);
```

Tokenizes raw text into a JSON array of integer token IDs using the session
tokenizer. `add_special` indicates whether special tokens (such as BOS) should
be added. Writes JSON array of integers to `*out`.

```c
int32_t ts_session_detokenize_json(const TsSession *s, const char *tokens_json,
                                   bool skip_special, char **out);
```

Detokenizes a JSON array of integer token IDs into text using the session
tokenizer. `skip_special` indicates whether special tokens (BOS/EOS/turn
markers) are stripped. Writes decoded text string to `*out`.

```c
int32_t ts_session_count_text_tokens(const TsSession *s, const char *text,
                                     bool add_special, uint32_t *out_count);
```

Evaluates the token count of a raw text string using the session's tokenizer.
`add_special` indicates whether special tokens (such as BOS) should be added.
Writes token count to `*out_count`.

```c
int32_t ts_session_fit_window_json(const TsSession *s, const char *messages_json,
                                   const char *reasoning, uint32_t max_tokens,
                                   char **out);
```

Fits a conversation transcript into a context token budget by pruning older
turns (preserving optional leading system/developer instruction and newest
turn), using the checkpoint's chat template and tokenizer.

`max_tokens` is the context budget limit (0 means session's resolved
max_context). Writes JSON result to `*out`:

```json
{ "retained": [...], "measuredTokens": 120, "removedTurnCount": 1, "hasRoomForGeneration": true }
```

`measuredTokens` IS A FLOOR ON A CONVERSATION CARRYING IMAGES, NOT A COUNT.
This renders the template and encodes it, and the template emits ONE marker
per image whatever the picture's size; the expansion to that page's
merged-token count happens later, inside `ts_generate`'s splice, and needs the
preprocessed grid. So an image turn is undercounted by roughly a page's worth
of positions, and a conversation this call says fits can still be refused by
`ts_generate` with a context-overflow error naming both numbers. Loud rather
than silent, but worth knowing before trusting the fit.
`ts_session_count_tokens` has the same property for the same reason.

## Generation

```c
int32_t ts_generate(const TsSession *s, const char *messages_json,
                    const char *options_json, TsEventCallback cb,
                    void *userdata, char **result_json);
```

Generates one assistant turn. Blocks for the whole turn.

`messages_json` is `[{"role":"user","content":"..."}]`, rendered through the
checkpoint's own chat template. Roles: system, developer, user, assistant,
tool.

**IMAGES.** `"content"` also accepts an ORDERED array of parts, which is how a
caller sends a picture:

```json
{ "role": "user", "content": [
    { "type": "image", "path": "/abs/page.png" },
    { "type": "text",  "text": "Transcribe this." } ] }
```

An image part carries EXACTLY ONE of `"path"` (a file this process can read)
or `"base64"` (a bare payload, or a full `"data:<media>;base64,<data>"` URL).
Both or neither is an error rather than a precedence rule. A bare string
`"content"` behaves exactly as it always has, byte for byte.

ORDER IS LOAD-BEARING: the template renders one marker per image and the nth
image is injected at the nth marker, so a caller that reorders the parts pairs
each picture with the wrong span: fluently, with no error. PREPEND rather than
append to match the reference processor, which builds `[image, text]`;
appending moves every position past the image.

Refused BY NAME when `sessionInfo.vision.active` is false. Check that field
before offering the caller a way to attach one: an install can carry a tower
and still refuse every image.

`options_json` may be NULL or `"{}"`. Recognised keys, with the defaults the
CLI uses: `maxNewTokens` 512, `temperature` 0.2, `topK` 64, `topP` 0.95,
`repetitionPenalty` 1.0, `seed` null, `stop` [], `stopTokens` [],
`reasoning` "off".

`reasoning` is `"off"|"low"|"medium"|"high"|"xhigh"`. THE ACCEPTED SET IS THE
CHECKPOINT'S: a level its template rejects comes back as an error naming the
level (Qwen 3.8 rejects `"high"`; its top setting is `"xhigh"`).

`cb` may be NULL, in which case nothing streams and the whole turn arrives in
`*result_json`:

```json
{ "promptTokens", "newTokens", "prefillSeconds", "decodeSeconds",
  "stopReason", "tokensPerSecond", "content", "reasoning",
  "peakMemoryPressure" }
```

`stopReason` is `endOfTurn` | `toolCalls` | `eos` | `stopString` | `maxTokens`
| `cancelled`. `peakMemoryPressure` is `"normal"` | `"warn"` | `"critical"`,
the worst seen while this turn decoded, and `"normal"` when nothing was
watching, which is the default (the in-loop probe follows the power profile).
Read `ts_system_info_json` for the machine's current state. `tokensPerSecond`
is null when no decoding happened, so a caller cannot plot a rate that was
never measured.

(The header's comment block runs the `stopReason` enumeration and the
`peakMemoryPressure` paragraph together across a page break; the enumeration
above is reconstructed from the header's own listed values at those two
points.)
