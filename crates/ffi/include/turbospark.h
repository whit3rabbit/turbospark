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
 *     "reasoningSupport",
 *     "steering": { "active", "mode", "scale", "summary" },
 *     "speculation": { "block", "drafter", "reason" },
 *     "specialTokens": { "bosId", "eosId", "padId", "endOfTurnId",
 *                       "stopTokenIds", "thinkStartId", "thinkEndId" } }
 *
 * maxContext and expertCacheSlots are the RESOLVED values, never what was
 * asked for: under "auto" the request carries no number, and the KV cache
 * has already been allocated at the resolved one. Neither a throughput nor a
 * footprint figure is readable without the slot count beside it.
 *
 * reasoningSupport is "level" | "toggleOnly" | "none". A GUI should disable
 * its reasoning picker on "none" and grey out the LEVELS on "toggleOnly",
 * where asking for one turns thinking on but sets no level.
 *
 * steering.active is true when a control vector is loaded on this session.
 * steering.summary holds a human-readable one-line description of the edit.
 *
 * speculation.block is the resolved block size, or null when this session
 * does not draft ahead; that null IS the "is it on" test, and drafter
 * ("mtp" | "dflash") is non-null exactly when block is. speculation.reason
 * says why it is off when a caller might have expected otherwise, and is
 * null both when they asked for "off" and when it is on.
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
