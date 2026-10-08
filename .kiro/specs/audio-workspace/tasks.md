# Implementation Plan

- [x] 1. Model catalog and installation
- [x] 1.1 Add the pinned task-aware Audio catalog and complete initial asset manifests
  - Verify immutable bytes, expose identities/capabilities/readiness/estimates,
    integrate Whisper and Kokoro installation with the audio store and receipts,
    and cover completeness, corruption, identity, deletion, and relocation.
  - _Boundary: crates/catalog; narrowly required model-io metadata_
  - _Requirements: 2.1, 2.2, 2.3, 2.5, 5.1_
  - _Design: 1, 2_
- [x] 1.2 Make pinned transfer resumable through the existing ranged downloader
  - Preserve exact-source validation, durable verified ranges, atomic publication,
    progress, pause/resume, cancel and interrupted retry with adversarial tests.
  - _Boundary: crates/catalog hub transfer; crates/repack ranged transfer seam_
  - _Depends: 1.1_
  - _Requirements: 2.4, 5.1_
  - _Design: 2_

- [ ] 2. Native audio runtime paths
- [x] 2.1 Adapt the pinned English frontend and Kokoro sentence synthesis contract
  - Vendor attributed MSRV-compatible frontend/resources, implement plain-text
    normalization/POS/dictionary/spelling and bounded sentence chunks, validate
    voice-row selection and unsupported input with independent parity fixtures.
  - _Boundary: crates/audio/tts/kokoro and frontend resource/dependency files;
    narrowly required catalog frontend provenance_
  - _Requirements: 3.2, 3.4, 4.1, 5.2_
  - _Design: 1, 3_
- [ ] 2.2 Implement Kokoro Metal session through shared audio operators
  - Keep portable numerical reference, implement real Metal execution of heavy
    operations, validate tensor parity and seeded synthesis, and expose runtime
    session/result/control contracts for the ABI.
  - _Boundary: crates/audio backend seam; crates/runtime Kokoro; crates/gpu operators_
  - _Depends: 2.1_
  - _Requirements: 3.2, 3.4, 4.1, 4.3, 5.2_
  - _Design: 3, 4_
- [ ] 2.3 Complete cooperative Whisper/MiniMax jobs and full-size music CLI execution
  - Use existing MiniMax Metal pipeline, add stage/step cancellation and progress,
    preserve packed dtype/layout/RNG/cache/overlap contracts and buffered STT ABI,
    replace qualified full-size music refusal, and test reset/stereo stitching.
  - _Boundary: crates/audio control; crates/runtime Whisper/Music3; crates/cli music_
  - _Depends: 2.2_
  - _Requirements: 3.1, 3.3, 3.5, 4.1, 4.3, 5.2_
  - _Design: 1, 3, 4_

- [ ] 3. Native ABI
- [ ] 3.1 Add guarded audio catalog, model, and cancellable job handles
  - Dedicated worker keeps non-Send model state local; add typed JSON requests,
    progress/bounded borrowed PCM, ownership/bounds checks, native generation
    exclusion and residency admission, preserving existing STT/catalog APIs.
  - _Boundary: crates/ffi and header; narrowly required runtime admission metadata_
  - _Depends: 1.2, 2.3_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.2_
  - _Design: 4_

- [ ] 4. Swift bindings
- [ ] 4.1 Add typed, serialized Swift audio catalog and session wrappers
  - Copy borrowed PCM safely, serialize native operations, propagate immediate
    errors, expose cancellable requests/progress, and verify close/cancel ownership.
  - _Boundary: swift/TurboSpark package and tests_
  - _Depends: 3.1_
  - _Requirements: 4.1, 4.2, 4.3, 5.2_
  - _Design: 4_

- [ ] 5. Swift workspace
- [ ] 5.1 Integrate navigation, workflows, downloads, profile lifecycle, and localization
  - Add Audio destinations/shortcuts, task pickers, shared FIFO/history/filters,
    encrypted identity selections, import/transcription, synthesis/music forms,
    playback/export and shared media/idle/profile teardown. Verify legacy settings,
    dedup, relocation, teardown and all localizations.
  - _Boundary: swift/TurboSparkApp source, localization, and tests_
  - _Depends: 4.1_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.4, 2.5, 3.1, 3.2, 3.3, 4.4, 5.1, 5.2, 5.3_
  - _Design: 5_

- [ ] 6. Integration and real checkpoint qualification
- [ ] 6.1 Qualify the exact checkpoints and complete app/workspace validation
  - Run independent reference, real Metal and full checkpoint gates for all three
    tasks, evaluate representative outputs, record timing/memory separately,
    validate navigation/playback/exports, reconcile overlapping STT tasks and
    document bounded evidence and unresolved neighboring checkout failures.
  - _Boundary: focused repairs, test/evidence scripts, docs, specification reconciliation_
  - _Depends: 5.1_
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - _Design: 1, 6_

## Implementation Notes

- Task 2.1 passed independent review and parent fresh 21-test verification.
  Full audio passed 663 tests with 38 ignored; all 26 canonical frontend cases
  and seven resource/license hashes match. Rust 1.82 Linux checks passed.
  This establishes the portable contract; Metal and checkpoint gates remain.
- Workspace build and fmt passed. The existing catalog cache authentication
  test expects a token at its HTTP mock, contrary to the official-host policy
  in commit 49459fc3. Preserve that policy and reconcile its owning tests.
- Task 2.1 debug: opening quotes must reserve an attached forward word group;
  adjacency alone cannot determine quote ownership. Keep terminal fallback for
  unpaired quotes and test grouping separately from canonical G2P parity.

## Approved expansion tasks

- [ ] 7.1 Native audio integration and typed Swift bindings
  - Audit current families, extend capability metadata/catalog, share worker
    service, add guarded audio model/job APIs and typed Swift wrappers, preserve
    existing wire contracts and validate ownership, PCM and cancellation.
  - _Boundary: audio/runtime/catalog/server/ffi; swift/TurboSpark; audio docs_
  - _Requirements: 6.1, 6.2_
  - _Design: 0_
- [ ] 7.2 Durable Swift Audio workspace
  - Implement profile library, encrypted asset lifecycle, navigation, model
    controls, task workflows, experiments, exports, accessibility and localization.
  - _Boundary: swift/TurboSparkApp; narrowly required binding additions_
  - _Depends: 7.1_
  - _Requirements: 6.3, 6.5, 6.6_
  - _Design: 0_
- [ ] 7.3 Meeting capture and draft transcription
  - Implement source selection, timestamped capture, encrypted chunk recovery,
    pause/markers, draft backlog and final processing, lifecycle and permission UI.
  - _Boundary: swift/TurboSparkApp audio; bundle permissions; capture tests_
  - _Depends: 7.2_
  - _Requirements: 6.4, 6.5, 6.6_
  - _Design: 0_
- [ ] 7.4 Qualification and source-based specification reconciliation
  - Run focused and full verification, real capture/model/app gates, preserve
    evidence boundaries and reconcile earlier tasks without inferred completion.
  - _Boundary: focused integration repairs; tests; evidence; specifications_
  - _Depends: 7.3_
  - _Requirements: 6.1, 6.6_
  - _Design: 0_

## Historical implementation notes

The approved expansion now has source implementations for native jobs and
bindings, encrypted library and asset lifecycle, task forms, capture/draft
transcription, presets, playback, and exports. See
[Audio workspace](../../../docs/AUDIO_WORKSPACE.md) and
[capability audit](../../../docs/AUDIO_WORKSPACE_CAPABILITIES.md) for current
evidence and unavailable adapters. The qualification tasks remain open:
permission-dependent device checks, full accessibility checks, missing native
model adapters, Kokoro Metal, and exact-profile model gates are not complete.
The earlier deferral of microphone capture and stored transcripts below records
the starting state; it no longer describes the implemented source.

- Task 1.2 passed fresh independent repair review: 45 focused catalog tests,
  15 ranged tests, 242 full catalog tests, live upstream checks, clippy and fmt.
  Parent fresh regressions passed and current source hashes match. Size-fallback
  HEAD requests now respect shared controls; active socket reads retain their
  existing timeout boundary. Changes remain unstaged in the shared checkout.
- Task 1.1 passed independent repair review and parent fresh 18-test verification.
  Upstream and cached lifecycle checks cover all three exact pins. The selective
  commit is deferred because catalog lib/store changes depend on neighboring
  uncommitted Music and speech foundations; all task changes remain unstaged.
- The starting checkout contains concurrent MiniMax Metal, STT, and Swift memory
  edits. Baseline copies are at /tmp/turbospark-audio-baseline-llyupikd. Preserve
  those changes; unsafe partial commits must remain unstaged with their dependency
  reported, per root AGENTS.md.
- Do not treat the older speech-to-text task status as current runtime evidence.
  This integration owns app workflows; microphone and stored transcripts remain
  deferred even when Whisper foundations already exist.
- Existing exact MiniMax installs have no Audio receipt. Adopt them only after
  validating store ownership and the complete pinned bytes; isolate incompatible
  records so one legacy install cannot break discovery for every Audio task.
- Pinned Misaki Rust output has punctuation, numeric normalization, POS-context,
  and letter-spelling defects. Independent canonical Python comparisons and the
  source POS asset hashes are under /tmp/turbospark-audio-qualification; fix these
  behaviors rather than turning the fork's defective output into golden fixtures.
- The starting Audio dependency graph fails Rust 1.82 because locked
  unicode-segmentation 1.13.3 requires Rust 1.85. Task 2.1 must narrowly restore a
  compatible dependency pin while preserving unrelated lockfile changes. Linux
  checks can use the existing Zig compiler with the temporary normalized-target
  wrapper under /tmp/turbospark-audio-qualification/cross-tools.
