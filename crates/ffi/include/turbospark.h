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
/* One parsed tool call, as a JSON object `{"id","name","arguments"}` in
 * `text` and the call's zero-based index within the turn in `a`. Fires only
 * when the caller offered the tool by name, which no ts_generate option
 * does yet: the kind is wired so hosts and the pipeline do not need a
 * second pass when the binding grows a way to offer tools. */
#define TS_EVENT_TOOL 3
/* Terminal event, fired ONCE per successful ts_generate just before it
 * returns, after every other event of the turn. `text` is the stop reason
 * spelled exactly as result_json's `stopReason` field spells it
 * ("endOfTurn", "toolCalls", "eos", "stopString", "maxTokens",
 * "cancelled"); `a` is the generated token count and `b` the prompt token
 * count. NOT fired when the run fails: a non-zero return plus
 * ts_last_error remains the only error signal.
 *
 * A HOST MUST TREAT ANY KIND IT DOES NOT KNOW AS A NO-OP. New kinds are
 * appended by newer libraries and that is not a version break. */
#define TS_EVENT_FINISH 4

/* ---- install progress kinds ---- */

/* A human-readable stage line, on the calling thread. */
#define TS_INSTALL_STAGE 0
/* Byte progress. `done` of `total`. CALLED CONCURRENTLY, see ts_install. */
#define TS_INSTALL_BYTES 1

/* ---- handle ---- */

typedef struct TsSession TsSession;
typedef struct TsServer TsServer;
typedef struct TsImageSession TsImageSession;

/*
 * One streamed generation event.
 *
 * `text` is UTF-8 of length `len`. It is NOT NUL-terminated and is valid
 * ONLY for the duration of this call: copy it before returning.
 */
typedef void (*TsEventCallback)(void *userdata, int32_t kind,
                                const char *text, size_t len,
                                uint32_t a, uint32_t b);

/* Image progress callback kinds. For TS_IMAGE_EVENT_STAGE, `text` is one of
 * text_encoder, transformer, vae_decoder or png_encode, and `a`/`b` are the
 * completed and total stage units. TS_IMAGE_EVENT_FINISH is terminal. */
typedef void (*TsImageEventCallback)(void *userdata, int32_t kind,
                                     const char *text, size_t len,
                                     uint32_t a, uint32_t b);

#define TS_IMAGE_EVENT_STAGE 0
#define TS_IMAGE_EVENT_FINISH 1

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
 *   visionSidecar     string | null (path to a standalone vision-tower
 *                       sidecar install to attach to a text-only trunk, or
 *                       "auto" to resolve one from the installed store by
 *                       the trunk's own family and hidden size; with no
 *                       pairing tower the session stays text-only, and an
 *                       ambiguity refuses the open). null means use the
 *                       trunk's own tower, if it has one.
 *   kvBits            "off" | "2" | "3" | "3.5" | "4" | null
 *                       (default off, which is what every release before this
 *                       key existed produced byte for byte). TurboQuant
 *                       KV-cache quantization. "3.5" splits into K3/V4. An
 *                       unsupported family or head_dim REFUSES this call by
 *                       name rather than silently opening at FP16.
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

/*
 * Frees the vision tower's open resources on this session (vision memory
 * sidecar): the streamer slots or mapped-residency buffer, the position
 * table, and -- when a sidecar is attached -- its own separate resident
 * weights and mapping. A session that will never see another image can give
 * all of that back without closing the whole session.
 *
 * Takes the SAME engine lock a generation turn holds for its whole
 * duration, so it must not be called concurrently with ts_generate() on the
 * same session (both already serialize through that lock; this simply
 * queues behind an in-flight turn rather than racing it).
 *
 * Does NOT forget an attached visionSidecar directory and does NOT change
 * ts_session_info_json()'s "vision" block, which is resolved once at open
 * from the install's own declaration -- releasing frees OPEN RESOURCES,
 * never the declared CAPABILITY or the sidecar ATTACHMENT. The next image
 * sent through ts_generate() reopens the tower exactly as the first one
 * did, from the attached sidecar if there is one or from this install if
 * not.
 *
 * Returns TS_OK if resources were freed (or there were none to free --
 * releasing a session whose tower is already closed, or whose install
 * declares no vision tower at all, is a harmless no-op). Returns
 * TS_ERR_UNSUPPORTED on a session that has no tower to release in the first
 * place: a scripted test session, or any session on a non-macOS build.
 */
int32_t ts_session_release_vision(const TsSession *s);

/* ---- native image generation ---- */

/* Opens a verified image install. This is a separate handle from TsSession:
 * image generation has its own backend and output lifetime. */
int32_t ts_image_session_open(const char *model_dir, TsImageSession **out);

/* Closes an image session. Do not call while ts_image_generate is running. */
void ts_image_session_close(TsImageSession *s);

/* Requests cancellation without blocking behind image work. */
void ts_image_session_cancel(const TsImageSession *s);

/* Generates one PNG. options_json is `{ "prompt", "seed", "width",
 * "height", "steps" }`; width, height and steps default to the supported
 * IG2 envelope. On success metadata_json is an owned JSON string released by
 * ts_string_free. A completed result has status "completed" and a PNG; a
 * canceled result has status "cancelled" and a zero-length PNG. */
int32_t ts_image_generate(const TsImageSession *s, const char *options_json,
                          TsImageEventCallback cb, void *userdata,
                          uint8_t **out_png, size_t *out_png_len,
                          char **out_metadata_json);

/* Frees the byte buffer returned by ts_image_generate. */
void ts_image_buffer_free(uint8_t *bytes, size_t len);

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
 *     "vision": { "active", "imageTokenId", "reason", "source",
 *                 "sidecarPath" },
 *     "kvBits",
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
 * vision.source is "install" when this session's tower (if any) comes from
 * the trunk's own directory, or "sidecar" when visionSidecar attached a
 * standalone tower instead (vision memory sidecar). It is null exactly when
 * no tower is present at all, i.e. the same case that leaves imageTokenId
 * null; when a tower IS present it is set on both the active and the
 * refused branches, so a host can say WHICH tower failed to serve an image.
 * vision.sidecarPath is the attached directory, present only when source is
 * "sidecar".
 *
 * A NON-NULL BLOCK IS A STATEMENT ABOUT THE SESSION, NOT THE NEXT TURN.
 * Acceptance is argmax(target) == proposal, exact only at temperature 0, so
 * a sampled turn decodes sequentially whatever this says -- silently, and
 * by design: this binding's own sampling default is 0.2, so a per-turn
 * warning would fire on the normal case. Send temperature 0 to speculate.
 *
 * kvBits is the RESOLVED "off" | "2" | "3" | "4" | "3.5 (K3/V4)", and unlike
 * speculation there is no auto-detect to report here: a named width either
 * opens this session or ts_session_open() fails, so what a caller asked for
 * and what this session runs at are always the same value.
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
 *   host    loopback or Tailscale IPv4 string | null (default 127.0.0.1).
 *             Tailscale binding requires a nonempty apiKey.
 *   captureText bool (default false). Bounded raw HTTP body previews in
 *             info.traffic, memory only; excludes headers and API keys.
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
 * REFUSES A DUPLICATE PUBLIC ID rather than renaming it. Every generative
 * session also advertises `claude-turbospark-<canonical-id>` through
 * GET /v1/models so Claude Code gateway discovery can select it. A canonical
 * id may not collide with another session's alias, and aliases may not
 * collide either. `ts_server_info_json` still reports canonical ids only,
 * which keeps the host's attach/detach bookkeeping stable.
 *
 * WHICH MODEL SERVES A REQUEST: an exact canonical id or discovery-alias
 * match wins. Failing that, if exactly ONE model is attached it serves the
 * request whatever name was
 * asked for -- which is what keeps a client sending its own default name
 * (Claude Code sends "claude-sonnet-4-6") working. With two or more
 * attached and no match, the request is a 404 naming what IS available.
 */
int32_t ts_server_attach_session(const TsServer *server, const TsSession *s,
                                 char **out);

/*
 * Adds an embedding model to a running server, enabling /v1/embeddings,
 * /api/embeddings, and /api/embed. model_path may be an install directory
 * containing config.json, model.safetensors, and tokenizer.json, or a model alias.
 *
 * Writes the assigned model id to *out (free with ts_string_free).
 */
int32_t ts_server_attach_embedding_model(const TsServer *server,
                                         const char *model_path,
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
 * "models" is every attached CANONICAL id, in attachment order. GET
 * /v1/models reports each canonical id followed by its Claude discovery
 * alias, but those aliases are intentionally absent here so a host can
 * continue to detach by the id it attached. "modelId" is the FIRST of them
 * ("" when none), kept for a reader written when a server could serve only
 * one; on a two-model server it is half the truth, so show "models".
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

/* What valid image-generation installs are present in ~/.turbospark. */
int32_t ts_image_installed_json(char **out);

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
 * `options_json` may be NULL, "" or "{}", all meaning every default:
 *
 *   loadGuard         same spellings as ts_session_open's, and it MUST be the
 *                     same value the host will OPEN with. This ranking and the
 *                     loader's refusal share one memory budget by construction,
 *                     which is what makes a recommendation trustworthy; ranking
 *                     under "relaxed" while sessions open under "strict"
 *                     promises a fit the loader then refuses, where the user
 *                     cannot see the two disagree.
 *   expertCacheSlots  same spellings as ts_session_open's ("auto" or a count),
 *                     and it MUST likewise be the value the host will OPEN
 *                     with. A footprint is slots x layers x expert stride, so
 *                     a ranking at one slot count and an open at another are
 *                     two configurations rather than one approximation.
 *
 * Each row carries "countedSource": "measured" | "estimated" | "unknown".
 * NEVER render an "unknown" row's countedBytes as a figure -- a row whose
 * header nobody has read reports zeros, and zero bytes reads as "fits easily".
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
 *
 * `options_json` may be NULL, "" or "{}". Keys: contextWindow (0 or absent
 * means 4096), loadGuard and expertCacheSlots, all with ts_recommend_json's
 * spellings and its rule that they must match what the host will OPEN with.
 *
 * "fit" is this machine's answer for that context and slot count, or NULL
 * when the header yielded no shape -- never a zeroed object, because an
 * absent measurement is not a measurement of zero. Its "mappedBytes" is the
 * PUBLISHED CHECKPOINT and not the .gturbo this port would write, which is
 * why it is labelled "mappedSource": "download". The two differ; the figure
 * is close for a GGUF (expert blobs are written verbatim, only the small
 * resident core is transcoded) and looser for MLX. "countedBytes" is
 * unaffected: it is built from the arch and the expert stride.
 */
int32_t ts_probe_json(const char *repo, const char *file,
                      const char *sidecar_repo, const char *options_json,
                      char **out);

/*
 * What a longer context window would cost an INSTALLED model, read off its
 * own manifest.
 *
 *   { "path", "trainedContext",
 *     "rungs": [ { "context", "kvBytes", "counted", "verdict", "runs",
 *                  "pastTrained", "isTrainedMax", "isLargestFitting" } ] }
 *
 * `options_json` takes loadGuard and expertCacheSlots, same spellings and the
 * same must-match-your-open rule as ts_recommend_json. contextWindow is
 * ignored: this call prices a LADDER of windows rather than one.
 *
 * DO NOT EXTRAPOLATE FROM ONE RUNG. KV is not linear in the window: a
 * sliding-window layer is a ring capped at sliding_window + 128 and stops
 * growing past it, while a fully-attentive layer grows forever. Measured
 * 4,096 -> 131,072 on the shipped baselines, Mistral 7B grows 32x (512 MiB to
 * 16,384) and Gemma 4 grows 9x (305 to 2,785). A caller multiplying its own
 * 4,096 figure is 3.5x high on Gemma, in the direction that refuses a window
 * that runs. Linear-attention layers contribute no KV at all.
 *
 * "rungs" is EMPTY when the install's shape could not be read. That is not a
 * model with no memory cost; it is a question nothing answered.
 */
int32_t ts_context_ladder_json(const char *model_path, const char *options_json,
                               char **out);

/*
 * Every .gguf a Hugging Face repository publishes, best quality first. One
 * API call, no header reads, no download -- cheap enough to fill a
 * quantization picker as a sheet opens.
 *
 *   { "repo", "revision", "shardedSkipped",
 *     "variants": [ { "file", "bytes", "quantLabel", "ladderRank",
 *                     "executable" } ] }
 *
 * IT CARRIES NO FIT, deliberately: a fit needs an ArchConfig, which needs a
 * header read PER FILE. Use this for the menu and ts_probe_json() for the
 * file the user picks.
 *
 * A file naming a type with no kernels here is LISTED with executable false
 * rather than hidden, because a picker showing three of a repository's eight
 * files reads as the repository having three. "bytes" and "quantLabel" are
 * null when unknown, never 0 and never "": a 0-byte row reads as a tiny file.
 * "shardedSkipped" counts multi-part files, which this port cannot walk; a
 * nonzero value is why a picker may be short or empty.
 */
int32_t ts_repo_variants_json(const char *repo, char **out);

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

/*
 * Signals every in-flight install walk to stop. Returns 1 when at least one
 * walk was running and has been signalled, 0 when nothing was running.
 *
 * The walk notices at its next step boundary or ranged chunk read (seconds,
 * not tensor boundaries) and fails with the "install cancelled" error, the
 * same death a network failure gives it: CANNOT RESUME applies, so nothing
 * of the partial install is kept. Safe to call from any thread while an
 * install is blocking another one.
 */
int32_t ts_install_cancel(void);

/*
 * How many install walks have finished (success, failure, or cancel) since
 * process start. Read it twice around ts_install_cancel to confirm a
 * cancelled walk actually exited rather than being wedged in a blocking
 * read.
 */
uint32_t ts_installs_finished(void);

/*
 * Reads the currently resolved Hugging Face token. If a token is found,
 * writes a newly-allocated string to *out (free with ts_string_free).
 * If no token is set, sets *out to NULL and returns TS_OK.
 */
int32_t ts_hf_token_get(char **out);

/*
 * Reads the currently resolved Hugging Face token and its source origin.
 * Writes JSON to *out (free with ts_string_free):
 *   {"token":"...","source":"..."}
 * If no token is set, sets *out to NULL and returns TS_OK.
 */
int32_t ts_hf_token_info_json(char **out);

/*
 * Saves a Hugging Face token to the local store (~/.turbospark/hf_token).
 */
int32_t ts_hf_token_set(const char *token);

/*
 * Clears the Hugging Face token from the local store.
 */
int32_t ts_hf_token_clear(void);

/*
 * Validates a Hugging Face token against the whoami-v2 API.
 * Writes a JSON result to *out (free with ts_string_free):
 *   {"status":"valid","name":"...","fullname":"...","email":"..."}
 *   {"status":"invalid","message":"..."}
 *   {"status":"rate_limited","retry_after_seconds":...}
 *   {"status":"unavailable","message":"..."}
 */
int32_t ts_hf_token_validate_json(const char *token, char **out);

/*
 * Reads the current Hugging Face mirror base URL into *out (free with ts_string_free).
 * Returns the default "https://huggingface.co" if unset.
 */
int32_t ts_hf_endpoint_get(char **out);

/*
 * Sets or clears the Hugging Face mirror base URL ($HF_ENDPOINT).
 * Pass NULL or an empty string to remove the override and reset to default.
 */
int32_t ts_hf_endpoint_set(const char *endpoint);

/*
 * Computes vector embeddings for a JSON array of strings using a local encoder
 * model (.safetensors directory or alias).
 *
 * texts_json is a JSON array of strings: ["text1", "text2"]
 * Writes a JSON array of float arrays to *out (free with ts_string_free):
 * [[0.1, ...], [0.2, ...]]
 */
int32_t ts_embedding_encode_json(const char *model_path,
                                 const char *texts_json,
                                 char **out);

/*
 * Computes the cosine similarity between two float vectors of length `len`.
 * Both vectors are L2-normalized first, so arbitrary nonzero inputs give a
 * true cosine in [-1, 1]; a zero vector yields 0.0.
 * Returns 0.0 if either pointer is null or len is 0, and also when an
 * internal panic is caught at the boundary (read ts_last_error to see why;
 * there is no status code here to carry TS_ERR_PANIC through, so the
 * sentinel is the only signal a caller gets).
 */
float ts_cosine_similarity(const float *a, const float *b, size_t len);

/*
 * Resolves a model alias (e.g. "gemma4") or relative path to its canonical
 * on-disk install path.
 *
 * Writes the path to *out (free with ts_string_free). Returns TS_ERR_OPEN if
 * the model is not found or directory does not exist.
 */
int32_t ts_model_resolve_path(const char *model_or_alias, char **out);

/*
 * Inspects whether a background turbospark server daemon is running.
 * Writes a JSON object to *out (free with ts_string_free):
 *   {"running":true,"pid":1234,"port":8080,"endpoint":"http://127.0.0.1:8080/v1","logPath":"..."}
 *   or {"running":false}
 */
int32_t ts_daemon_status_json(char **out);

/*
 * Stops the background turbospark server daemon if running.
 */
int32_t ts_daemon_stop(void);

/*
 * Starts the background turbospark server daemon with optional arguments.
 * args_json is a JSON array of string arguments, e.g. ["--port", "8080", "--model", "gemma4"].
 * Pass NULL or "[]" for defaults.
 */
int32_t ts_daemon_start(const char *args_json);

/*
 * Stops and restarts the background turbospark server daemon with optional arguments.
 */
int32_t ts_daemon_restart(const char *args_json);

#ifdef __cplusplus
}
#endif

#endif /* TURBOSPARK_H */
