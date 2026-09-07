# Swift app storage: where state lives, and what a test may touch

All app state lives in three JSON files under
`~/Library/Application Support/TurboSpark/`, plus a handful of manager
directories that split the same way. This page is the home for what each
store does on a decode failure, why `AppStorageRoot` exists, and the
three-way answer to "which directories may a test write to."

Read this before adding a persisted field, writing a test that constructs
an `AppModel`, or touching any of the seven stores listed below.

## The three JSON files

`settings.json`, `chats_archive.json`, `projects_archive.json`, each
written whole with `.atomic` on every mutation.

**A decode failure used to be silently fatal to the file's content.**
Every store's `load()` swallowed a decode failure and returned the empty
default, so a schema change that is not backwards-compatible discarded the
user's data rather than erroring -- and the NEXT atomic write overwrote the
intact file with the empty default. Add fields with `decodeIfPresent` and a
default, the way `MacAppSettings`'s hand-written `init(from:)` already
does.

**That hazard fired for real** (found 2026-08-29 on the archive on this
machine). `AppChatMessage` gained `toolCalls` and `toolResults` as
non-optional arrays on the synthesized decoder, so ONE message written
before those fields existed threw `keyNotFound`, `load()` swallowed it, and
the app came up with an empty chat list while a four-message conversation
sat intact on disk -- which the next `persistChats()` would have
overwritten for real. Nothing errors, nothing is logged, and the user
reads it as "the app lost my chats" rather than as a decode failure.
`AppChatMessage` and `AppChat` now carry hand-written tolerant `init(from:)`s
and `load()` REPORTS the error to stderr before falling back.

**Element-level tolerance only** (`state#45`): `decodeIfPresent` tolerates
an ABSENT key and nothing else. Three inner decoders threw on an unknown
enum raw value or one bad array element and took the whole archive with
them past the hand-written tolerance of the outer ones. A wrong-TYPED array
key still throws, or an intact file gets read as empty and overwritten.

`persistChats()` re-encodes the entire archive on every keystroke of the
draft, since `promptText`'s setter calls it; fine at current sizes, and the
first thing to look at if typing ever feels heavy.

## `AppStorageRoot` and why the test suite needed one

Found 2026-08-31 from a bug report: a chat titled "T10 chat" reappeared on
every launch after being deleted in the UI. It was `HookDecisionRoutingTests`'s
fixture (`AppChat(title: "T10 chat")`), sitting ALONE in the real
`chats_archive.json`.

**Seven stores each spelled their own path** -- `AppChatFileStore`,
`AppProjectFileStore`, `MacAppSettings`, `GlobalMcpFileStore`,
`AppHookStore`, `CustomToolManager` and `AppHookStdinPayload` all computed
`applicationSupportDirectory + "TurboSpark"` inline, with no test seam
anywhere. So an `AppModel()` built in a test read and WROTE real user data,
and since the archive is written WHOLE with `.atomic`, a one-chat fixture
REPLACED the user's entire history. 20 `AppModel()` constructions across 10
test files could each do it.

Three things make this class expensive. Nothing fails: the suite is green,
the app starts, and the only tell is a chat the user did not create.
Deleting it in the UI does not help, because the next `swift test` writes
it back -- reads as a broken Delete rather than as test pollution. And it
is unrecoverable: there is no backup, the write is atomic, and by the time
anyone notices, the original is several runs gone.

`AppStorageRoot` is the one root now. The redirect is AUTOMATIC (keyed on
the XCTest host, not on a flag a test sets), because a new test cannot be
trusted to opt into a protection whose failure mode is silent; and it keys
on the PROCESS rather than per test, because these stores are `.shared`
singletons and static enums whose first touch can precede any test body.
`TURBOSPARK_STATE_DIR` overrides it. `StorageIsolationTests` guards it, and
`testSavingAChatDoesNotTouchTheRealArchive` hashes the real archive around
a save -- disabling the redirect reddens 3 of its 4 cases AND changes that
hash, which is the original data loss reproduced on demand.

## The three-way answer to "which directories may a test write to"

`AppStorageRoot` covers the seven stores above. Two other answers are not
written down anywhere else and are easy to get wrong.

**The engine's `~/.turbospark` is NOT covered.** `~/.turbospark/installed.json`
is the CATALOG's, reached through the FFI, and nothing redirects it. Any
assertion driven through `refreshModels`, `deleteModel` or `installed()`
answers differently on a machine with a given alias installed -- not a
flaky test, a test measuring the developer's disk. Found 2026-09-03 by a
SURVIVING mutation: `deleteModel`'s alias-collision guard could not be
reddened, because whether a colliding row existed was a property of
`~/.turbospark` rather than of the fixture. The fix is the usual one --
extract the decision as a pure static (`AppModel.isCatalogTracked(model:in:)`)
and feed it a fixture, rather than seeding the real store.

**The managers split down the middle.** `CustomToolManager.globalToolsDirectory`
is `AppStorageRoot.subdirectory("tools")` and is safe. But
`CustomToolManager.userHomeToolsDirectory`, `SkillManager.defaultUserSkillsDirectory`
and `AgentManager.defaultUserAgentsDirectory` are literal `~/.turbospark/...`
-- so a USER-scope skill or agent fixture writes the developer's own home.
Anything whose subject is a USER-scope skill or agent has no test for that
reason, and not for lack of one being worth writing.

## `AppModel().someProperty == someDefault` tests the wrong layer

`AppModel.init()` calls `loadSettings()` as its first statement, so a
`@Published` property's declared default is unreachable through the public
initializer -- the value a fresh `AppModel()` actually reports comes from
`MacAppSettings`'s own default. `AppModel().interactionMode == .chat`
therefore does not test the property declaration; it tests whichever
`settings.json` happens to sit in the shared, per-process
`AppStorageRoot.directory`, which carries state across every test file that
ran earlier in the same `swift test` invocation. Test the `MacAppSettings`
default directly, and for an `AppModel`-level round trip, delete the
relevant file under `AppStorageRoot` first so the assertion does not
depend on execution order.

## Redirecting a prompt-time side effect

`MemoryStore.defaultBase()` (`swift/docs/SWIFT_MEMORY.md`) redirects the
same way, for a reason specific to it: prompt ASSEMBLY creates the memory
directory as a side effect, so every test building a prompt for a project
would otherwise mkdir inside the user's real `~/.turbospark` -- the exact
failure this page's header section is about, arrived at through a code path
that never looks like a storage test.
