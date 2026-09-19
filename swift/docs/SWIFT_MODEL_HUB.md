# Swift model hub, catalog install, and session lifecycle

The Model Hub pane (Discover, Installed), catalog install, opening and
selecting a session, and the in-process server's multi-model attach. This
page is the home for the load-bearing facts across all four: install
progress, what a hub card is allowed to claim, `activeLoadGuard`, and why
`AppModel.server` can outlive `AppModel.session`.

Read this before touching `AppModel+Installation.swift`, `AppModel+Models.swift`,
`AppModel+Server.swift`, `Catalog.swift`, `ModelHubView`, `ModelDetailPaneView`,
or `ServerPaneView`.

## Install: progress, threading, no resume

The install byte callback fires CONCURRENTLY from several download threads,
so progress can go backwards. `TurboSparkCatalog.install` documents taking
the maximum rather than the last value; `AppModel+Installation.swift` does
exactly that with a local `maxBytes`. A progress bar driven from the last
event alone jitters backwards on a real install.

The install also runs on a DEDICATED `Thread`, not a global queue slot: it
blocks for tens of minutes, and parking a shared concurrent-queue worker
that long starves the rest of the process. And the walk CANNOT RESUME, so a
cancelled or failed install restarts from zero; the UI should say so before
starting.

**Cancelling is real since `ts_install_cancel` landed on the Rust side.**
`cancelInstall()` sets the flag the blocking walk polls at its next ranged
chunk read (seconds, not tensor boundaries) AND cancels the consuming Task;
the walk then dies exactly as a network failure would, keeping nothing.
The cancelled alias sits in `abandonedInstallAliases` only until
`watchCancelledWalkExit` sees `installsFinished()` prove the walk exited
(a bounded 30 s poll), after which the model can be re-installed without
an app restart; if the walk never notices the flag, the refusal stays --
the safe pre-cancel behavior. `state#27`'s complaint was about the era
when the C ABI had no cancel call and the button only stopped watching;
the history below is kept because the epoch guard it introduced still is
the thing keeping a cancelled walk's delayed tail from clobbering a new
install's state.

`state#15`: a cancelled install's delayed tail reset the NEW install's
state; `installEpoch` guards it. `state#27`: `cancelInstall` claimed a
cancellation the engine cannot perform and reopened the install guard.
`state#44`: `installRepo` recorded no `installingAlias` and never checked
`abandonedInstallAliases`, so the two-writer protection covered catalog
installs only. `state#52`: the install `.finished` arm was not
epoch-guarded (the comment claimed the check came first; it existed only in
`catch`), and the tail never cleared `installStageText`. `state#83`: the
`.finished` arm toasted "installed and loaded" BEFORE awaiting an open that
returns silently at `guard !generating, !opening`; the toast follows the
open's outcome now. `state#102`: `installETAText` had a declaration, four
resets and a view reading it, and no writer at all -- the remaining-time
row simply never appeared. Grep for the writer, not the declaration.

## A badge or a filter that cannot fail carries no information

The Model Hub shipped three of them, all found by writing the first unit
test over the code rather than by looking at it (`ModelHubFilterTests`).

`ModelCardView` drew an unconditional `checkmark.seal.fill` captioned
"Verified model in TurboSpark catalog" on every row, while `models.json`
marks rows `verified`, `runs` or `caveat` and the view never read the field.
`ModelHubView`'s capability filter matched alias SUBSTRINGS, so
`.conversational` fell through to `break` and returned the whole catalog,
`museGlimmer` was listed under both "Reasoning" and "Dense", and the fit
filter offered "Too large" on machines where no row is refused -- a choice
that can only ever return an empty list.

**The format label was the expensive one, because it is the field a user
picks a row by.** `ModelFamilyVisuals.resolve` stated `formatLabel` by hand
in each of its 16 family branches, and a family and a quantization are
INDEPENDENT: `mistral7b` and `tinyllama` are GGUF rows that both rendered
"MLX INT4", `ornith9b` is GGUF Q8_0 rendered as MLX because its alias has no
"gguf" in it, `gemma4-gguf` is Q8_0 rendered as "Q4_K_M", and `bonsai27b` is
MLX affine 1-bit rendered as INT4. Wrong on a majority of the shipped rows,
and wrong in the direction that matters.

The fix is one derivation instead of sixteen assertions:
`ModelFamilyVisuals.formatLabel(alias:name:)` reads the parenthesised
suffix of the catalog row's own `name` up to the first comma, which IS
where the catalog states the format. `ModelHubFilter` builds its dropdown
options from that same call, so the badge and the filter cannot disagree.

Two rules out of it. Derive a display value from the data or read it off
the row; never restate it per branch -- the restatement is correct on the
day it is written and silently wrong at the next checkpoint. And build a
filter's options from what is present, not from the full enum, or the UI
offers choices that match nothing.

## A zero from an absent measurement is not a measurement of zero

A `.unknown` fit verdict means the sizing could not be determined, and the
numeric fields that arrive with it are zeros. `ModelDetailPaneView` used to
render them through the same grid as a real reading, so an unsized row
advertised "Unified Memory: Zero KB" and "Max Context: 0 tokens" beside a
summary that said "unknown (probe it)".

**The slot count is the dangerous one, and it does not even look wrong.**
With no architecture read there is no expert stride, `Auto` divides by
nothing and returns `DEFAULT_CACHE_SLOTS`, so the pane showed "Expert
Slots: 16 slots" -- an answer arrived at BY IGNORANCE that is
indistinguishable from a measured 16 (root `AGENTS.md` Gotcha 58). The card
now branches on `fitIsKnown`, every cell is guarded on being non-zero, and
the unsized state shows only what the CATALOG states plus the command that
would produce the rest.

There is deliberately no Probe button on that card. `ts_recommend_json` is
offline-only and this binding exposes no probing call, so the button would
have to lie about what it does; the pane names
`turbospark-model recommend --probe` instead. Check that the work exists
before adding the control that claims to do it.

A smaller sibling in the same pane: `TextField`'s title on macOS is a
VISIBLE LABEL, not a placeholder. `InspectorOptionsSection`'s rate-cap
field passed "Uncapped" as that title without `.labelsHidden()`, so it was
drawn beside the field and clipped to "Un-" by the inspector's width. A
stray truncated word next to a control is worth reading as a missing
`.labelsHidden()` before it is read as a layout problem.

## `activeLoadGuard`: one budget for the hub and the loader

`AppModel.activeLoadGuard` exists so the hub and the loader cannot resolve
different memory tiers. `TurboSparkCatalog.recommend` and `ts_session_open`
share one memory budget by construction, which is what makes a hub verdict
worth showing; a hub ranking under `.relaxed` while sessions open under
`.strict` promises a fit the loader then refuses, in the one place a user
cannot see the two disagree. All three `recommend` call sites
(`ModelHubView`, `CatalogSheet`, `ModelInstallView`) and `buildOpenOptions`
read that one accessor. A fourth caller building its own from
`runtimeOptions` would compile and be wrong only when the user moves the
setting off the default.

`state#21`: the expert-slot picker offered values the engine panics on, and
omitted the legal 24.

## Selecting, opening, and unloading a model

`state#14`: `selected` was keyed on `alias`, which is not unique once a
scanned row exists; keyed on `path` now. `state#50`: `open` and
`selectModel` guarded only `!generating`, so two overlapping opens each
cleared `opening` and `selected`/`session` could name different models;
`setModelURL` never set `selected`, and `reconcileSelection` then reverted
the path field. `state#73`: `deleteModel` guarded `!generating` and
`!submitting` but not `!opening`, so Delete during a load removed the
directory under a mapping still being established and the open's tail
published a session for a deleted model. `canDeleteModel` is the predicate
now; every pane that offers the button disables it accordingly. `state#84`:
`selectModel` returned whenever `selected?.path == model.path`, session or
no session, so Load Model and Load & Chat were dead after an unload or a
failed open. `state#85`: `unloadModel()` checked one of `canUnloadModel`'s
three terms and two panes called it with no `.disabled` at all. `state#43`:
`deleteModel` had no `!generating` guard (`unloadModel()` returns silently
there, so the directory was removed under a live mmap), never checked
`serverAttachedSessions`, and matched on `alias ||` past state#14.
`state#87`: `modelScanTask` held a MainActor wrapper awaiting an inner
`Task.detached`, and cancelling a parent does not cancel a detached child,
so the cancel state#51 added stopped nothing -- the stored task IS the walk
now, checking `Task.isCancelled` per entry. `state#88`: "Remove from
TurboSpark" on a scanned row destroyed its notes, tags and favorite and
left the bytes alone, and the next scan put the row straight back stripped
-- it records a reversible `excludedScanPaths` entry instead, with a
Restore button in Settings > Models. `state#89` and `state#90`: the git
diff and worktree status probes behind the model-storage picker had a
rename-format mismatch (porcelain vs numstat spell a rename differently)
and an unbounded untracked-file read on the main actor; see `state#54` for
the worktree half of the same probe.

## Server: multi-model attach outlives one chat session

Added 2026-08-30 (`TurboSparkServer`, `AppModel+Server.swift`).
`AppModel.server` OUTLIVES `AppModel.session` unless something stops it
first, and that something is every caller that clears `session`. A
`TurboSparkServer` holds its own reference to the engine on the Rust side
(`crates/ffi/AGENTS.md` Gotcha 13's whole design), so `session = nil` alone
does not stop a server started against it -- the model stays resident and
the server keeps answering requests for a model the UI no longer shows as
loaded. `open(_:)`, `unloadModel()` and `setModelURL(_:)` call
`stopServer()` before clearing `session` for exactly this reason; a fourth
call site that clears `session` directly would compile and leak the old
model for as long as the server keeps running. `TurboSparkServer` guards
the OTHER direction too: `ts_server_stop` frees its C handle, so `stop()`
and `deinit` both route through one `NSLock`-guarded idempotent path rather
than each calling the C function directly, which would double-free if a
caller stopped it explicitly and then let it go out of scope.

**Since 2026-08-30 a server serves several models, and `stopServer()` is
the wrong tool for unloading one.** Stopping the server to swap the chat
model would take every OTHER attached model down with it. `open(_:)`,
`unloadModel()` and `setModelURL(_:)` call `detachChatSessionFromServer()`
instead, which removes whatever entry that session was attached under and
leaves the rest serving. A fourth site that clears `session` without coming
through there keeps that model resident, served, and invisible in the Chat
pane -- the same failure, now per model.

**Two references hold an attached model and both have to go.** The server
holds one on the Rust side and `AppModel.serverAttachedSessions` holds the
Swift one; dropping either alone keeps the weights, the KV cache and the
compiled pipelines resident. `detachModelFromServer(id:)` does both. The
CHAT session is the deliberate exception and needs no branch:
`AppModel.session` is a third reference the Chat pane still holds, so
ejecting it from the Server pane stops it being SERVED and leaves it
loaded.

**Nothing in the UI states an address, a port or an auth state of its own;
all three are read back off `info()`.** Audited 2026-08-30. `ServerInfo`
carries `host`, with a `baseURL` helper, and `AppModel.serverInfo` is the
ONE accessor the view reads. Both status rows are a value
(`ServerStatusRows`) rather than inline view code, which is the only reason
they can be tested at all -- `ServerStatusRowsTests` needs no model, no
session and no bound socket, and every `ServerInfo` in it is DECODED from
the JSON `ts_server_info_json` emits, so the cases pin the wire spelling
too. `testAddressFollowsAHostThatIsNotLoopback` is the case that carries
the file: hardcoding the address back to loopback reddens it and the IPv6
case while leaving the reading-from-info case green.

**The auth state was invisible rather than merely unverified.**
`startServer` trims the key field and maps empty to `nil`, so a key of
nothing but spaces starts an UNAUTHENTICATED server, and the panel used to
render identically either way with `info.authEnabled` decoded and shown
nowhere. There is an Auth row now, and the toast names it. `ServerOptions`'s
doc also called an unauthenticated default "appropriate for a server bound
to loopback and reachable only from inside this process" -- a loopback TCP
socket is reachable by every process on the machine, not a property a
socket can have. The settings caption says the same thing plainly now.

**No view body calls into the binding.** `AppModel.serverInfo` is a
published SNAPSHOT rather than a computed accessor: `info()` takes a lock,
crosses the ABI and decodes JSON, and the Server pane re-evaluates far more
often than a server changes. `refreshServerInfo()` is the only writer,
called on start, attach, detach and once per poll tick for the uptime. Poll
runs at 2 Hz, driven by the timer alone -- chosen so the console feels
live, not a throughput limit: the engine's ring holds ~2,000 events, so
nothing is lost at any rate a person would pick. When something IS lost the
engine says so and `ServerMetricsStore.droppedEvents` sums it, because a
console missing rows reads exactly like a server that was idle.

**The pane's arithmetic is in two pure types so it can be tested.**
`ServerMetricsStore` folds the four events that describe a request back
into one record and derives the series; `ServerEndpointCatalog` holds the
route list and the connect snippets. That is what caught a real bug:
`trim()` ran only on the batch path, so the public single-event `ingest`
grew the window without limit.

Three things the pane refuses to state. A model with no session shows
"attached" rather than a zero context and a plausible-looking 16 slots
arrived at by ignorance (see the fit-pane section above). The endpoint list
carries no route the engine cannot serve, pinned by
`testNoEndpointIsAdvertisedThatTheEngineCannotServe` -- `/v1/embeddings` is
listed now that `crates/server/src/embeddings.rs` exists behind
`--embedding-model`. And there is no time-to-first-token chart, because
nothing inside a generation can measure one
(`crates/server/AGENTS.md` Gotcha 29); the pane shows prefill, decode and
the queue instead.

`state#28`: a failed server start left `serverStopRequested` latched, so
the next Start stopped itself; `detachModelFromServer` dropped the Swift
session reference only inside the `do`, so a throwing detach kept the
model resident with no UI row. `state#53`: same shape, a second time.
`state#72`: `stopServer()` during `attachModelToServer` orphaned a session
-- the attach holds `server` as a local across a tens-of-seconds open and
writes into the map the stop just cleared, and every remover starts with
`guard let server`, so the entry becomes unreachable. Identity-checked
after the await now, not merely non-nil, since Stop-then-Start leaves a
DIFFERENT server. `state#86`: `refreshServerInfo`'s `try?` wrote nil on a
transient failure, at 2 Hz, blanking the address, port, auth row and every
model row -- reads as the server having stopped. Keeps the last snapshot
and latches the error once now. `state#96`: `setReasoning` keyed the
remembered level on `alias ?? path`, so the path arm was dead and two
installs sharing an alias shared one level; keyed on `path` now, alias read
as a legacy fallback and the entry pruned by `deleteModel`. `state#97`:
`isToolCallingSupported` fell back to `installed.first` with nothing
selected, so an unrelated install decided the Forge guardrails default; see
`SWIFT_SESSION_CAPABILITIES.md`.

## Tests

`ModelHubFilterTests` (badges, filters, format labels), `ServerStatusRowsTests`
(the decoded-from-JSON address/auth cases), and the state-numbered
regression cases each carry their own file under `Tests/TurboSparkAppTests/`.
