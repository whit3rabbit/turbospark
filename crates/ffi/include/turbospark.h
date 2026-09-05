/*
 * turbospark.h -- C ABI over the inference engine, for a native GUI host.
 *
 * Hand-written rather than generated. The surface is small enough to read in
 * one sitting, and the SwiftPM target that compiles against it is a stronger
 * check that the two sides agree than a generator would be: a generator only
 * ever restates the Rust side to itself.
 *
 * THE CONTRACT, in four rules.
 *
 * 1. ERRORS. Every fallible call returns TS_OK (0) or a non-zero code.
 *    On a non-zero return, ts_last_error() on the SAME THREAD holds a
 *    message. Read it before making another call on that thread.
 *
 * 2. OWNERSHIP. A `const char *` argument is borrowed for the duration of
 *    the call and never retained. A `char **` out-parameter receives an
 *    allocation the caller must return through ts_string_free(). There is no
 *    third case, and nothing here hands back a pointer into its own state.
 *
 * 3. JSON. Options and results travel as JSON strings, so adding a knob is
 *    never an ABI break. Keys are camelCase, so a Swift Codable needs no
 *    CodingKeys. The per-token streaming path carries no JSON: it is a
 *    pointer and a length.
 *
 * 4. THREADING. A session is single-threaded: one generation at a time, and
 *    ts_generate() blocks for the whole turn, so call it from a background
 *    thread. The ONE exception is ts_session_cancel(), which is safe from
 *    any thread and never blocks -- that is what makes a Stop button work.
 *    ts_install()'s byte callback is ALSO called concurrently from worker
 *    threads; see below.
 *
 * PLATFORM. The engine is macOS-only. Everywhere else ts_session_open()
 * fails with TS_ERR_UNSUPPORTED and a sentence, while the catalog, probe and
 * install calls work normally -- the artifact is the same whether or not
 * this machine can run it.
 */

#ifndef TURBOSPARK_H
#define TURBOSPARK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ---- */

#define TS_OK 0
/* A null pointer, a non-UTF-8 string, or a value outside its allowed set. */
#define TS_ERR_INVALID_ARGUMENT 1
/* The install could not be opened: missing directory, unreadable manifest,
 * unsupported architecture, or a context window that does not fit memory. */
#define TS_ERR_OPEN 2
/* Generation or installation failed. The session remains usable. */
#define TS_ERR_GENERATE 3
/* A JSON argument did not parse, or a result could not be built. */
#define TS_ERR_JSON 4
/* The engine is not available on this platform. */
#define TS_ERR_UNSUPPORTED 5
/* A panic was caught at the boundary. The process is intact and the
 * operation did not happen. This is a bug in the library, not in the call. */
#define TS_ERR_PANIC 6

/* ---- streaming event kinds ---- */

/* Prefill progress. `a` is prompt tokens done, `b` the total. `text` empty. */
#define TS_EVENT_PREFILL 0
/* Visible reply text. Accumulate THIS as the assistant turn. */
#define TS_EVENT_CONTENT 1
/* The model's reasoning, already separated from the reply. Do NOT feed it
 * back as conversation history: Harmony's own convention drops prior-turn
 * analysis, and Qwen's template drops prior-turn <think> blocks, so sending
 * it back gives the model something it was never trained to read. */
#define TS_EVENT_REASONING 2

/* ---- install progress kinds ---- */

/* A human-readable stage line, on the calling thread. */
#define TS_INSTALL_STAGE 0
/* Byte progress. `done` of `total`. CALLED CONCURRENTLY, see ts_install. */
#define TS_INSTALL_BYTES 1

/* ---- handle ---- */

typedef struct TsSession TsSession;
typedef struct TsServer TsServer;

/*
 * One streamed generation event.
 *
 * `text` is UTF-8 of length `len`. It is NOT NUL-terminated and is valid
 * ONLY for the duration of this call: copy it before returning.
 */
typedef void (*TsEventCallback)(void *userdata, int32_t kind,
                                const char *text, size_t len,
                                uint32_t a, uint32_t b);

/*
 * One install-progress event. `text`/`len` as above; `done` and `total` are
 * bytes for TS_INSTALL_BYTES and zero otherwise.
 */
typedef void (*TsInstallCallback)(void *userdata, int32_t kind,
                                  const char *text, size_t len,
                                  uint64_t done, uint64_t total);

/* ---- errors and strings ---- */

/*
 * Copies this thread's last error into `buf` as a NUL-terminated string.
 *
 * Returns the message's own length in bytes, excluding the NUL -- which is
 * NOT necessarily the number of bytes written, so a caller given a value
 * >= cap can allocate that much and call again. Pass buf = NULL to ask for
 * the length alone.
 */
size_t ts_last_error(char *buf, size_t cap);

/* Frees a string returned through a `char **`. NULL is a no-op. */
void ts_string_free(char *s);

/* ---- session lifecycle ---- */

/*
 * Opens a model and writes a handle to `*out`.
 *
 * `model_dir` is a path to a .gturbo directory OR an installed alias; an
 * existing directory always wins, so a bare name cannot silently run a
 * different model than the one named.
 *
 * `options_json` may be NULL or "{}". Recognised keys, all optional:
 *   maxContext        number | "auto" | null   (default auto)
 *   expertCacheSlots  number | "auto" | null   (default auto; 8/16/24/32)
 *   powerProfile      "performance" | "balanced" | "efficiency" | null
 *                       null ASKS THE OS, so Low Power Mode selects
 *                       efficiency. Name one explicitly when measuring.
 *   maxTokensPerSec   number | null
 *   speculation       "off" | "auto" | number | "<number>" | null
 *                       (default auto). Resolved at OPEN, because that is
 *                       where the drafter's state is allocated. A NAMED
 *                       block that cannot be served fails this call; "auto"
 *                       that cannot be served opens and reports why in
 *                       sessionInfo.speculation.
 *   speculativeDrafter "auto" | "mtp" | "dflash" | null
 *                       (default auto). "auto" ENABLES an MTP head and only
 *                       REPORTS a DFlash2 one, which is measured rather
 *                       than stylistic: DFlash2 reads 1.43-1.50x on code
 *                       and math and 0.96x throughput at +17.4% J/token on
 *                       PROSE, so it is opt-in.
 *   loadGuard         "off" | "relaxed" | "balanced" | "strict" | number | null
 *                       How much of the machine may be committed. A NUMBER is
 *                       an absolute ceiling in BYTES on what the engine
 *                       ALLOCATES (slot cache + KV), not on the install's
 *                       size -- a large install streaming from disk is what
 *                       this engine is for. null means "relaxed", which is
 *                       what this ABI did before the key existed and what
 *                       every published footprint row was measured under.
 *   minAutoContext    number | null (default 0, meaning no floor)
 *                       Refuse to open when maxContext is automatic and
 *                       resolves below this many tokens. Says NOTHING about
 *                       an explicit maxContext: a caller naming a number has
 *                       decided how to spend their own machine.
 *   steering          string | null (path to .gguf control vector)
 *   steeringMode      "ablate" | "add" | "clamp" | "renorm" | null
 *   steeringScale     number | null (default 1.0)
 *   steeringLayers    "START:END" | null (0-based inclusive layer range)
 *   steeringTarget    number | null (for clamp mode, default 0.0)
 *   steeringGate      number | null (activation threshold >= 0, default 0.0)
 *
 * Opening is expensive: it maps gigabytes and compiles Metal pipelines.
 * Open once and keep the handle.
 */
int32_t ts_session_open(const char *model_dir, const char *options_json,
                        TsSession **out);

/*
 * Closes a session. NULL is a no-op.
 *
 * Must NOT be called while a generation is in flight on another thread.
 */
void ts_session_close(TsSession *s);

/*
 * Asks the in-flight generation to stop. Safe from any thread; never blocks.
 *
 * The flag is cleared at the start of every generation, so a Stop pressed
 * between turns does not cancel the next one. A cancelled run is NOT an
 * error: ts_generate returns TS_OK with stopReason "cancelled" and whatever
 * text had been produced, and the KV cache describes itself honestly, so the
 * conversation can continue from the partial turn.
 */
void ts_session_cancel(const TsSession *s);

/* ---- introspection ---- */

/*
 * Everything resolved at open, as JSON:
 *
 *   { "modelPath", "family", "maxContext", "trainedContext",
 *     "pastTrainedContext", "expertCacheSlots", "vocabSize", "dialect",
 *     "reasoningSupport", "reasoningLevels",
 *     "steering": { "active", "supported", "reason", "mode", "scale",
 *                   "summary" },
 *     "toolCalling": { "native", "reason" },
 *     "speculation": { "block", "drafter", "reason" },
 *     "vision": { "active", "imageTokenId", "reason" },
 *     "specialTokens": { "bosId", "eosId", "padId", "endOfTurnId",
 *                       "stopTokenIds", "thinkStartId", "thinkEndId" } }
 *
 * maxContext and expertCacheSlots are the RESOLVED values, never what was
 * asked for: under "auto" the request carries no number, and the KV cache
 * has already been allocated at the resolved one. Neither a throughput nor a
 * footprint figure is readable without the slot count beside it.
 *
 * reasoningSupport is "level" | "toggleOnly" | "none" and says what KIND of
 * control is meaningful: disable the picker on "none", and on "toggleOnly"
 * present it as an on/off switch, since asking for a level there turns
 * thinking on and sets no level.
 *
 * reasoningLevels is WHAT TO PUT IN THAT CONTROL, ascending, always starting
 * "off". Build the menu from it and from nothing else. The set belongs to the
 * checkpoint and cannot be derived from the family: Qwen 3.8 answers
 * ["off","low","medium","xhigh"] and RAISES on "high", while gpt-oss and Muse
 * Glimmer answer ["off","low","medium","high"]. Offering a level absent from
 * here fails the turn with the template's own error message. Levels that
 * render the same prompt are already collapsed, so a "toggleOnly" checkpoint
 * answers exactly two entries and a "none" one answers ["off"]; that second
 * entry is "low" by POSITION and is not a label to print.
 *
 * steering.active is true when a control vector is loaded on this session.
 * steering.summary holds a human-readable one-line description of the edit.
 *
 * steering.supported IS A DIFFERENT QUESTION FROM steering.active, and it is
 * the one to gate a control on. active says a vector is running; supported
 * says one COULD be. ts_session_open() REFUSES a control vector on a family
 * whose decode flow does not dispatch the edit, so a host that offers the
 * knob against such an install offers one whose only outcome is a failed
 * load. steering.reason carries the refusal's own wording when supported is
 * false, and is null when it is true. An unsteered session on a family that
 * steers answers { active: false, supported: true, reason: null }.
 *
 * toolCalling.native says whether THIS CHECKPOINT'S OWN MARKUP carries tool
 * calls the engine parses, i.e. whether a call can arrive already framed.
 *
 * IT IS NOT "TOOLS DO NOT WORK", AND HIDING A TOOL CONTROL ON IT IS THE WRONG
 * READING. A model on a dialect with no tool markup can still be prompted
 * into emitting a call as ordinary prose, and recovering exactly that is what
 * a guardrail rescue is for -- so native:false marks the case a rescue helps
 * MOST, not a case to refuse. Report it; do not gate on it. toolCalling.reason
 * names the dialect, and for one checkpoint family names markup that exists
 * and has no parser (calls are reported as reasoning there rather than handed
 * over).
 *
 * speculation.block is the resolved block size, or null when this session
 * does not draft ahead; that null IS the "is it on" test, and drafter
 * ("mtp" | "dflash") is non-null exactly when block is. speculation.reason
 * says why it is off when a caller might have expected otherwise, and is
 * null both when they asked for "off" and when it is on.
 *
 * vision.active MEANS "AN IMAGE WOULD BE SERVED", NOT "A TOWER EXISTS", and
 * a host must gate its attach control on it rather than on the family name.
 * An install can carry a tower and still refuse every image -- the pixel
 * budget is read from the checkpoint's own preprocessor_config.json and has
 * no default worth falling back to, so an install streamed without that
 * sidecar reports active false with the reason in vision.reason. reason is
 * non-null exactly in that case, which is the only one a caller can act on.
 *
 * A NON-NULL BLOCK IS A STATEMENT ABOUT THE SESSION, NOT THE NEXT TURN.
 * Acceptance is argmax(target) == proposal, exact only at temperature 0, so
 * a sampled turn decodes sequentially whatever this says -- silently, and
 * by design: this binding's own sampling default is 0.2, so a per-turn
 * warning would fire on the normal case. Send temperature 0 to speculate.
 */
int32_t ts_session_info_json(const TsSession *s, char **out);

/*
 * The decode phase breakdown, as JSON. See wire::PhaseReport.
 *
 * Cumulative over every forward pass the session has served, PREFILL
 * INCLUDED, so a per-call number is an average over the whole context range
 * rather than a number at the current context. The buckets also cover the
 * inside of the forward pass only: the sampler and the detokenizer run after
 * it returns and appear in none of them.
 */
int32_t ts_session_phases_json(const TsSession *s, char **out);

/*
 * This process's peak physical footprint in bytes, or 0 where unavailable.
 *
 * The same mach counter every published memory figure for this engine is
 * measured with. What it counts differs by install shape: on a streamed MoE
 * model the mapped weights ARE counted, on a dense one they are not, so read
 * it beside the context window rather than comparing across families.
 */
uint64_t ts_peak_footprint_bytes(void);

/*
 * Hardware and power telemetry for this machine, as JSON:
 *   { "physicalMemoryBytes", "recommendedWorkingSetBytes", "chip",
 *     "lowPowerMode", "thermalLevel", "memoryPressure" }
 *
 * "memoryPressure" is "normal" | "warn" | "critical", the kernel's own
 * verdict. POLLED HERE UNCONDITIONALLY, unlike the decode loop's own probe,
 * which follows the power profile: a session running the default
 * "performance" profile watches nothing, so the "peakMemoryPressure" on a
 * generation result is the ABSENCE of a reading rather than a report that
 * memory was fine. A status panel should read this call.
 */
int32_t ts_system_info_json(char **out);

/* ---- generation ---- */

/*
 * Evaluates the exact prompt token count of `messages_json` using the
 * session's chat template and tokenizer, without running generation.
 *
 * `reasoning` is "off"|"low"|"medium"|"high"|"xhigh" (or NULL for "off").
 * Writes token count to `*out_count`.
 */
int32_t ts_session_count_tokens(const TsSession *s, const char *messages_json,
                                const char *reasoning, uint32_t *out_count);

/*
 * Formats a conversation transcript into raw prompt text using the session's
 * chat template and reasoning effort setting.
 *
 * `messages_json` is [{"role":"user","content":"..."}].
 * `reasoning` is "off"|"low"|"medium"|"high"|"xhigh" (or NULL for "off").
 * Writes formatted prompt string to `*out_prompt`.
 */
int32_t ts_session_render_prompt(const TsSession *s, const char *messages_json,
                                 const char *reasoning, char **out_prompt);

/*
 * Tokenizes raw text into a JSON array of integer token IDs using the session tokenizer.
 * `add_special` indicates whether special tokens (such as BOS) should be added.
 * Writes JSON array of integers to `*out`.
 */
int32_t ts_session_tokenize_json(const TsSession *s, const char *text,
                                 bool add_special, char **out);

/*
 * Detokenizes a JSON array of integer token IDs into text using the session tokenizer.
 * `skip_special` indicates whether special tokens (BOS/EOS/turn markers) are stripped.
 * Writes decoded text string to `*out`.
 */
int32_t ts_session_detokenize_json(const TsSession *s, const char *tokens_json,
                                   bool skip_special, char **out);

/*
 * Evaluates the token count of a raw text string using the session's tokenizer.
 * `add_special` indicates whether special tokens (such as BOS) should be added.
 * Writes token count to `*out_count`.
 */
int32_t ts_session_count_text_tokens(const TsSession *s, const char *text,
                                     bool add_special, uint32_t *out_count);

/*
 * Fits a conversation transcript into a context token budget by pruning older
 * turns (preserving optional leading system/developer instruction and newest turn),
 * using the checkpoint's chat template and tokenizer.
 *
 * `messages_json` is [{"role":"user","content":"..."}].
 * `reasoning` is "off"|"low"|"medium"|"high"|"xhigh" (or NULL for "off").
 * `max_tokens` is the context budget limit (0 means session's resolved max_context).
 *
 * Writes JSON result to `*out`:
 *   { "retained": [...], "measuredTokens": 120, "removedTurnCount": 1, "hasRoomForGeneration": true }
 *
 * measuredTokens IS A FLOOR ON A CONVERSATION CARRYING IMAGES, NOT A COUNT.
 * This renders the template and encodes it, and the template emits ONE
 * marker per image whatever the picture's size; the expansion to that page's
 * merged-token count happens later, inside ts_generate's splice, and needs
 * the preprocessed grid. So an image turn is undercounted by roughly a
 * page's worth of positions, and a conversation this call says fits can
 * still be refused by ts_generate with a context-overflow error naming both
 * numbers. Loud rather than silent, but worth knowing before trusting the
 * fit. ts_session_count_tokens has the same property for the same reason.
 */
int32_t ts_session_fit_window_json(const TsSession *s, const char *messages_json,
                                   const char *reasoning, uint32_t max_tokens,
                                   char **out);

/*
 * Generates one assistant turn. Blocks for the whole turn.
 *
 * `messages_json` is [{"role":"user","content":"..."}], rendered through the
 * checkpoint's own chat template. Roles: system, developer, user, assistant,
 * tool.
 *
 * IMAGES. "content" also accepts an ORDERED array of parts, which is how a
 * caller sends a picture:
 *
 *   { "role": "user", "content": [
 *       { "type": "image", "path": "/abs/page.png" },
 *       { "type": "text",  "text": "Transcribe this." } ] }
 *
 * An image part carries EXACTLY ONE of "path" (a file this process can read)
 * or "base64" (a bare payload, or a full "data:<media>;base64,<data>" URL).
 * Both or neither is an error rather than a precedence rule. A bare string
 * "content" behaves exactly as it always has, byte for byte.
 *
 * ORDER IS LOAD-BEARING: the template renders one marker per image and the
 * nth image is injected at the nth marker, so a caller that reorders the
 * parts pairs each picture with the wrong span -- fluently, with no error.
 * PREPEND rather than append to match the reference processor, which builds
 * [image, text]; appending moves every position past the image.
 *
 * Refused BY NAME when sessionInfo.vision.active is false. Check that field
 * before offering the caller a way to attach one: an install can carry a
 * tower and still refuse every image (see the field's own note).
 *
 * `options_json` may be NULL or "{}". Recognised keys, with the defaults the
 * CLI uses:
 *   maxNewTokens 512, temperature 0.2, topK 64, topP 0.95,
 *   repetitionPenalty 1.0, seed null, stop [], stopTokens [], reasoning "off"
 *
 * `reasoning` is "off"|"low"|"medium"|"high"|"xhigh". THE ACCEPTED SET IS
 * THE CHECKPOINT'S: a level its template rejects comes back as an error
 * naming the level (Qwen 3.8 rejects "high"; its top setting is "xhigh").
 *
 * `cb` may be NULL, in which case nothing streams and the whole turn arrives
 * in `*result_json`:
 *
 *   { "promptTokens", "newTokens", "prefillSeconds", "decodeSeconds",
 *     "stopReason", "tokensPerSecond", "content", "reasoning",
 *     "peakMemoryPressure" }
 *
 * stopReason is endOfTurn | toolCalls | eos | stopString | maxTokens |
 * peakMemoryPressure is "normal" | "warn" | "critical", the worst seen while
 * this turn decoded -- and "normal" when nothing was watching, which is the
 * default (the in-loop probe follows the power profile). Read
 * ts_system_info_json for the machine's current state.
 *
 * cancelled. tokensPerSecond is null when no decoding happened, so a caller
 * cannot plot a rate that was never measured.
 */
int32_t ts_generate(const TsSession *s, const char *messages_json,
                    const char *options_json, TsEventCallback cb,
                    void *userdata, char **result_json);

/* ---- in-process HTTP server ---- */

/*
 * Starts an in-process HTTP server sharing ALREADY-OPEN models, and writes a
 * handle to `*out`. Serves the same routes turbospark-server does:
 * GET /health, POST /v1/chat/completions, POST /v1/completions,
 * POST /v1/responses, POST /v1/messages, POST /v1/messages/count_tokens,
 * GET /v1/models, GET /v1/models/{id}, and the Ollama-compatible
 * GET /api/tags, GET /api/version, POST /api/show, POST /api/chat,
 * POST /api/generate.
 *
 * `s` MAY BE NULL, meaning start with nothing attached. The server binds and
 * answers GET /health (reporting "state": "empty"); every generation route
 * returns 503 until ts_server_attach_session() adds a model. That is the
 * state a GUI starts a server in before its user has chosen what to load,
 * and passing a non-NULL `s` is exactly equivalent to starting NULL and
 * attaching immediately.
 *
 * `options_json` may be NULL or "{}". Recognised keys:
 *   port    number (default 0, meaning let the OS choose; read the port
 *             ACTUALLY bound back from ts_server_info_json)
 *   apiKey  string | null (default null, meaning NO AUTH AT ALL)
 *   guardrails
 *           "on" | "off" | null (default null, meaning the engine default,
 *             which is on). Any other string is an error rather than a
 *             silent fallback. PROCESS-LEVEL, matching
 *             `turbospark-server --guardrails`: a per-request field would let
 *             any client opt its own traffic out of the repair this
 *             deployment chose. Applies to every model attached to this
 *             server, including ones attached later.
 *
 * UNAUTHENTICATED DOES NOT MEAN PRIVATE TO THIS PROCESS. This comment used to
 * say the socket was "reachable only by the process embedding it", which is
 * not a property a TCP socket can have: a loopback bind keeps it off the
 * network and reachable by EVERY process on the machine, so any local program
 * that can read or guess the port can drive the model. apiKey is the only
 * access control there is.
 *
 * THE SERVER OUTLIVES EVERY SESSION ATTACHED TO IT. It holds its own
 * reference to each underlying engine, so calling ts_session_close(s) after
 * this call frees only the caller's own handle -- the model stays resident
 * and the server keeps serving it until ts_server_detach_model() removes
 * that one, or ts_server_stop() releases them all. Do one of those if a
 * model should actually be freed.
 *
 * This server serves images on both endpoints when the serving session's
 * install carries a usable tower (see sessionInfo.vision.active), encoding
 * and generating under one lock exactly as the standalone binary does. A
 * request it cannot serve is refused by name rather than silently dropped.
 *
 * Blocks until the socket is bound (or binding fails), not until the first
 * request is served.
 */
int32_t ts_server_start(const TsSession *s, const char *options_json,
                        TsServer **out);

/*
 * Adds an open session's model to a RUNNING server, and writes the id
 * clients address it by to `*out` (the install directory's own name, the
 * same string ts_server_info_json reports in "models"). Free it with
 * ts_string_free.
 *
 * Takes effect immediately: no rebind, and no interruption to a request
 * already in flight on another model.
 *
 * REFUSES A DUPLICATE ID rather than renaming it. The id is what a request's
 * "model" field names and what ts_server_detach_model keys on, so a silently
 * suffixed second copy would be addressable under a name the caller never
 * learned. Two sessions on one install directory is a caller mistake.
 *
 * WHICH MODEL SERVES A REQUEST: an exact id match wins. Failing that, if
 * exactly ONE model is attached it serves the request whatever name was
 * asked for -- which is what keeps a client sending its own default name
 * (Claude Code sends "claude-sonnet-4-6") working. With two or more
 * attached and no match, the request is a 404 naming what IS available.
 */
int32_t ts_server_attach_session(const TsServer *server, const TsSession *s,
                                 char **out);

/*
 * Removes a model from a running server by id, releasing the server's
 * reference to its engine.
 *
 * TS_OK when one was attached under `model_id`, TS_ERR_INVALID_ARGUMENT when
 * none was -- reported rather than silently succeeding, because at that
 * point the caller's own model list and the server's have gone out of step.
 */
int32_t ts_server_detach_model(const TsServer *server, const char *model_id);

/*
 * Signals the server to stop, blocks until its background thread has
 * actually exited, and frees the handle. NULL is a no-op.
 *
 * This is also what releases every attached model.
 */
void ts_server_stop(TsServer *server);

/*
 * { "port", "host", "modelId", "models", "authEnabled", "uptimeSeconds" }
 * as JSON.
 *
 * "port" is the port ACTUALLY bound, never the one requested: port 0 in
 * ts_server_start's options asks the OS to choose one, so this is the only
 * place that number is knowable.
 *
 * "host" is the IP ACTUALLY bound, from the same call, and is what a caller
 * should build a URL out of. It reads "127.0.0.1" today because that is what
 * this library binds; restating that literal instead of reading this field is
 * correct only for as long as that stays true, and cannot report the day it
 * does not.
 *
 * "models" is every attached id, in attachment order -- the same ids and the
 * same order GET /v1/models reports. "modelId" is the FIRST of them ("" when
 * none), kept for a reader written when a server could serve only one; on a
 * two-model server it is half the truth, so show "models".
 *
 * "uptimeSeconds" is from a monotonic clock and is unaffected by the wall
 * clock moving under a long-running host.
 */
int32_t ts_server_info_json(const TsServer *server, char **out);

/*
 * Takes up to `max` buffered request events and writes
 * { "events": [...], "dropped": N } to `*out`. Free it with ts_string_free.
 *
 * DRAINING, NOT PEEKING: an event is returned exactly once. Poll this on a
 * timer and append what comes back to your own log.
 *
 * Each event is an object tagged by "kind":
 *   requestStarted  { id, atMs, method, path }
 *   requestRouted   { id, requested, served, stream }
 *   generated       { id, model, promptTokens, newTokens, prefillSeconds,
 *                     decodeSeconds, stopReason }
 *   requestFinished { id, status, durationMs }
 *   modelAttached   { atMs, model }
 *   modelDetached   { atMs, model }
 *
 * "id" ties the events of one request together. "requested" and "served"
 * differ whenever the single-model fallback fired, which is the common case
 * rather than an edge one.
 *
 * THERE IS NO TIME-TO-FIRST-TOKEN FIELD, and its absence is the honest
 * answer: nothing inside a generation can measure one, because a caller
 * means "request in, first token out" and that includes the wait behind the
 * one-generation-at-a-time lock. Subtract "prefillSeconds" +
 * "decodeSeconds" from requestFinished's "durationMs" to get that wait.
 *
 * A request the tool-call guardrails re-asked emits TWO "generated" events,
 * which is the useful reading rather than a duplicate: the retry is real
 * work the machine did.
 *
 * `max` bounds ONE call rather than the buffer -- anything over it stays
 * queued for the next poll, so a burst arrives late rather than being lost.
 * `max == 0` means no bound, which is what a host draining before shutdown
 * wants.
 *
 * "dropped" counts events discarded since the PREVIOUS poll, oldest first,
 * and is nonzero only for a host that stopped draining long enough to
 * overrun ~2,000 events. Show it: a silently lossy log is indistinguishable
 * from an idle server.
 */
int32_t ts_server_poll_events_json(const TsServer *server, uint32_t max,
                                   char **out);

/* ---- model management (available on every platform) ---- */

/* The curated catalog as a JSON array, each row carrying "installed". */
int32_t ts_catalog_json(char **out);

/* What is installed in ~/.turbospark, as a JSON array. */
int32_t ts_installed_json(char **out);

/*
 * Deletes an installed model directory and forgets it from ~/.turbospark.
 * Returns TS_OK on success or TS_ERR_INVALID_ARGUMENT if not installed.
 */
int32_t ts_model_delete(const char *alias);

/*
 * Ranks curated models by hardware fit on this machine for `context_window`
 * tokens (e.g. 4096 or 8192, 0 means default 4096). Returns JSON array of
 * recommendations.
 *
 * `options_json` may be NULL, "" or "{}", all meaning every default. One key:
 *
 *   loadGuard  same spellings as ts_session_open's, and it MUST be the same
 *              value the host will OPEN with. This ranking and the loader's
 *              refusal share one memory budget by construction, which is what
 *              makes a recommendation trustworthy; ranking under "relaxed"
 *              while sessions open under "strict" promises a fit the loader
 *              then refuses, where the user cannot see the two disagree.
 */
int32_t ts_recommend_json(uint32_t context_window, const char *options_json,
                          char **out);

/*
 * Probes a Hugging Face repository by HEADER ALONE: kilobytes and seconds,
 * no download. `repo` is "owner/name" or "owner/name@revision". `file` and
 * `sidecar_repo` may be NULL.
 *
 * The result carries "runnable" and, when false, "refusedBecause". Read
 * "slotCacheBytes" before "downloadBytes": what decides whether a model runs
 * here is slots x layers x expert stride, not the model's size.
 */
int32_t ts_probe_json(const char *repo, const char *file,
                      const char *sidecar_repo, char **out);

/*
 * What a .gguf control vector declares, read from the file alone: no model,
 * no session, no network. A vector is around 1.3 MB, so this is milliseconds.
 *
 *   { "hidden", "coveredLayers", "minLayer", "maxLayer", "spannedLayers",
 *     "declaredMode", "declaredArch" }
 *
 * Call it BEFORE offering a vector against an install, so a host can say
 * "this file is 4096 wide and your model is 5120" instead of letting
 * ts_session_open() fail minutes into a load. It reads the same parser
 * ts_session_open() reads, so the two cannot disagree about what a file
 * means.
 *
 * A SHAPE MATCH IS NOT A SEMANTIC MATCH, and this reports the shape only. The
 * engine refuses a width or layer-count mismatch and refuses NOTHING else: a
 * vector extracted for a different checkpoint of the same hidden size opens,
 * steers, and changes behaviour in a direction nobody asked for, silently. A
 * host rendering these fields owes its user that sentence. declaredArch is
 * advisory -- nothing validates against it, which is why it is worth showing.
 *
 * minLayer is normally 1 rather than 0. Block 0 is not expressible in a file
 * this engine writes and llama.cpp never applies a direction there, so a
 * "31 of 32" reading is that convention working, not a gap.
 */
int32_t ts_control_vector_info_json(const char *path, char **out);

/*
 * What installing `alias` will cost, as {"downloadBytes","installBytes"}.
 * Call before ts_install to show a determinate bar and a space warning.
 */
int32_t ts_install_bytes_json(const char *alias, char **out);

/*
 * Installs the catalog row `alias`. Blocks for the whole walk, which is
 * minutes to tens of minutes.
 *
 * THE WALK CANNOT RESUME. It streams the checkpoint without writing it to
 * disk whole, and a failure restarts from the beginning. Tell the user that
 * BEFORE starting, not after failing; the first stage line says so.
 *
 * `cb` receives TS_INSTALL_STAGE lines on the calling thread and
 * TS_INSTALL_BYTES updates FROM WORKER THREADS, CONCURRENTLY and possibly
 * out of order -- ranged downloads are split across connections. A callback
 * that touches shared state must synchronise it.
 */
int32_t ts_install(const char *alias, TsInstallCallback cb, void *userdata,
                   char **result_json);

/*
 * Probes and installs an arbitrary Hugging Face repository `repo` under
 * local `alias`. `file` and `sidecar_repo` may be NULL.
 *
 * Blocks for the whole walk and cannot resume.
 */
int32_t ts_install_repo(const char *repo, const char *alias, const char *file,
                        const char *sidecar_repo, TsInstallCallback cb,
                        void *userdata, char **result_json);

#ifdef __cplusplus
}
#endif

#endif /* TURBOSPARK_H */
