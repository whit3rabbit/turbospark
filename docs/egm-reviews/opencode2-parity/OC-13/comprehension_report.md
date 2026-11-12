# Stage 6: Comprehension report (OC-13)

Reviewer: egm-comprehension (completed). Read the document alone, no extra context.

## What this proposes

The Swift app would check a version and capability handshake before attaching to the out-of-process background daemon. The daemon's existing status/health endpoint would return three new fields: protocolVersion, engineVersion and engineBuild. The app would probe the endpoint and accept the daemon only if the major version matches and the daemon's minor version is at least the app's minimum. On a mismatch the app refuses once, with a localized message naming both versions and a Restart action that launches the app's own daemon. The Server pane would show the daemon's version, plus a warning state on mismatch. Separately, the generated C header and the Rust FFI would each carry a contract revision constant, compared when a session is created, so a stale linked library fails loudly.

## How the existing system works, per this document

The app talks to the engine in two ways. The in-process FFI is built together with the app, so the two cannot differ in version. The daemon is a separate process that a user may have started from an older install. The FFI contract is a generated C header plus hand-written Swift wrappers. The daemon attach path has no version check, so an old daemon serving a newer app produces confusing request errors. The document says the model-file format (the "gturbo manifest") already gets a minor-version check and a "producer block". The document does not say what a "producer block" is.

## Findings

CO-1 Major. Lines 19-26, 49. "Attach" and "attach path" are central but undefined. Line 60 says the app "always launches a compatible daemon itself" while CUJ 1 (lines 74-80) has it attaching to a user-started stale daemon. Fix: add two sentences on how the app discovers or launches the daemon and when a pre-existing one is reused.

CO-2 Major. Lines 44-45, 74-80. Restart "launches the app's own daemon" but does not say what happens to the stale daemon (stopped, other port, held by another process). Fix: state whether Restart stops the old daemon and what happens if that fails.

CO-3 Major. Lines 41-46, 65-67. Version fields have no format or meaning, "app's minimum" has no source, and it is unclear which field feeds the rule. Fix: add a small table of field, example value, source, and which feeds the rule.

CO-4 Major. Lines 65-67. Open Question 1 is labeled "exact-minimum" but describes a minimum-minor rule that Goals (line 44) already commit to. Fix: rename to "minimum-minor" and mark decided, or state what is undecided.

CO-5 Major. Lines 23, 30-35. A newer daemon with a different major is not addressed, nor how the check relates to the FFI constant. Fix: add one sentence on the major-bump case or point at the Non-Goal.

CO-6 Major. Lines 22, 107-110. `ts_session` is undefined and not shown to equal "session open" at lines 50 and 93. Fix: define it as the FFI session handle created at session open.

CO-7 Major. Lines 49-53, 70, 88-95, 108-110. FFI check described inconsistently: who reads which constant, what "fatal" means in a shipping app, and Open Question 2 says "compile error" while the plan says runtime assertion. Fix: state which side reads which constant, when it runs, and whether failure is debug-only or a release crash.

CO-8 Major. Lines 128, 134. "routes.rs or equivalent" and "crates/ffi/ generator" are vague. Fix: name the real files or say they are not yet located.

CO-9 Nit. Lines 35-36. "gturbo manifest minor check and producer block" is neither defined nor linked. Fix: add a link or one-line gloss.

CO-10 Nit. Lines 3-9, 22. OpenCode "sidecar discovery" and Section 8 are not explained. Fix: gloss "sidecar" once and say what OpenCode's predicates do.

CO-11 Nit. Lines 142-156. Swift test command missing; unclear whether the mutation check is manual or automated. Fix: add the Swift test command and say the mutation check is manual.

CO-12 Nit. Line 116. "21 languages" is not listed or linked. Fix: link the localization doc.

CO-13 Nit. Lines 41-42, 100. "status/health surface" has no route, method or auth stated. Fix: give the route once.

No blockers: every section could be paraphrased.
