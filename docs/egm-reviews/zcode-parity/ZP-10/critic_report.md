# Stage 7: Critic report (ZP-10 whiteboard canvas)

Round 1 only (no remediation, so convergence not assessed). Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All three completed. Standard: feature-spec-standard for design and structural.

## Design critic (DC)

DC-1 (Blocker). Lines 17, 77, 136-139. Wrong pane target. `SplitChatPanes.swift` is the secondary-chat column with no pane targets or tabs. The right column is one slot arbitrated by `AppRightColumnClaimant` (`Theme/AppRightColumnClaimant.swift`: none, artifact, htmlPreview, filePreview, browser, inspector, projectSummary) with AppModel flags and mutual-exclusion setters (swift/CLAUDE.md Gotcha 17). Spec never states the precedence position or how "tab" fits. Fix: re-base the file list on the claimant enum, RootView, flag/setters and toolbar entry; add a test pinning `resolve` ordering.

DC-2 (Blocker). Lines 140-143, 38, 62-65, 158-159. Composer attachment path unverified. No pasted-image path in `PromptComposerView.swift`. Attachments flow through `AppPromptAttachment` and `AppModel+Files.swift`, file-backed through the vault attachment cache. Export temp-file location, lifecycle and cache entry are unstated; the `@` panel (trigger in `ComposerAutocompleteEngine.swift`) has no file or design. Fix: name the real API, the export file lifecycle and the `@` integration, or make `@` a non-goal.

DC-3 (Major). Lines 62-65, 158-159, 45-47. No vision-capability gate. `AppModel+Files.swift` lines 38-58 accept images only when `visionIsActive`; most local models have no vision tower. Fix: define Export when vision is inactive (disable with `visionRefusalReason` or attach as file), add a failure CUJ and string, name the model for the manual check.

DC-4 (Major). Lines 83-89, 116, 144-148. Persistence self-contradictory and under-specified. `AppJSONStore` (`State/AppStorageRoot.swift:164`) is Codable JSON only with no blob API; "profile state root" vs `AppStorageRoot` unclear; encrypted-profile (vault) decision missing, so blobs may leak sketch content in plaintext; orphan handling, atomic write order, board delete, workspace move/rename and debounce flush on quit (`AppShutdownCoordinator`) unaddressed; no corrupt-blob test (`PKDrawing(data:)` throw). Fix: concrete layout, vault decision, flush-on-shutdown, corrupt-blob test.

DC-5 (Major). Lines 62, 115-116, 145-147. Export under-specified. "Natural size" undefined for an unbounded canvas: need `PKDrawing.bounds`, minimum size, padding, max pixel cap, scale and background (transparent export with dark-mode ink unreadable); off-main-thread claim unjustified; empty-board export undefined; PNG-dimensions test lacks expected values.

DC-6 (Major). Lines 36-37, 91-93, 115-120. PencilKit assumptions unverified: `drawingPolicy`/mouse support and minimum macOS target unsourced; six swatches plus slider means driving `PKInkingTool` programmatically; undo comes from the responder-chain `undoManager`, enablement binding unstated; swatch values, eraser mode and undoability of clear unspecified; "strokes count" source (`PKDrawing.strokes.count`) unnamed.

DC-7 (Major). Lines 95-111. Alternatives rationale non-compliant ("Rejected by author"). Missing alternatives: per-chat scoping, reuse of ZP-11 annotation surface (shared drawing and export machinery), storing boards in `ProfileDatabase`.

DC-8 (Major). Whole document. No Risks and Limits section (quit during debounce, blob growth, unencrypted blobs, PencilKit format compatibility, vision gate, corrupt decode, workspace moved or deleted, memory at large canvas sizes).

DC-9 (Major). Lines 49-53, 71. Open Question conflicts with Goals and CUJ 2 (already per workspace); "workspace" meaning undefined; export target chat undefined with ZP-23 concurrent chats (attachment could land in a generating chat).

DC-10 (Nit). Lines 117-118, 149-157. "21 languages" vs INDEX "ten localization parity gates"; verification lacks exact commands and target registration check; mutation checks cover naming only; ZCode claims uncited; no ZP-10/ZP-11 dependency note; no failure, empty-board or board-management CUJ; no link re-check note.

## Structural critic (ST)

ST-1 (Major). Lines 102-103, 110-111. "Rejected by author" attribution without a link, forbidden by the standard; Alternative B is a deferral, not a trade-off.

ST-2 (Major). Whole document. No Risks and Mitigations section (corrupt blobs, orphans, unbounded growth, export size, mouse-only use, PencilKit availability, PNG privacy).

ST-3 (Major). Lines 49-53 vs 33, 71, 84. Open Question contradicts downstream sections; resolve or state which sections change.

ST-4 (Major). Lines 41-47. Non-Goals omit vector editing (Alternative B), region export and cross-device sync.

ST-5 (Major). Lines 76-89, 127-150. No export-to-composer data flow; ownership of the handoff, draft storage and `@` entry unstated; `@` panel missing from Target Files.

ST-6 (Major). Lines 36-37 vs 55-72. Goals (swatches, width, eraser, undo/redo/clear, naming) lack acceptance behavior in CUJs; "non-colliding" and "natural size" undefined.

ST-7 (Nit). Lines 83-88. "PNG for export" in the stroke-data line and "archive" undefined.

ST-8 (Nit). Lines 144-150. Mutation-check sub-bullet inconsistent in structure; move to Verification Plan.

ST-9 (Nit). Lines 118, 157. "21 languages" vs "ten gates"; unverified.

ST-10 (Nit). Lines 9-11, 4-7. No ZP-ID or size line; whiteboardStore.ts and Swift source links from ZCODE_COMPARISON.md omitted.

ST-11 (Nit). Lines 22-30, 43-44. ZCode "stroke-based" and swatch-count claims uncited.

## Style critic (SY)

SY-1 (Major). Lines 22-29. Search claim has no scope or date; "first-class composer attachments" is promotional. Fix: state path and date, describe the measurable ZCode behavior.

SY-2 (Major). Lines 91-94. PencilKit platform claim unsourced; `drawingPolicy` value not stated; "ride the canvas's coordinator" vague. Fix: cite docs or spike, name `undoManager`.

SY-3 (Major). Lines 49-53, 102-103, 110-111. Open question already assumed; "Rejected by author" restates the author. Fix: resolve and drop attribution.

SY-4 (Nit). Lines 98-100. "worse feel" subjective; palm rejection irrelevant to mouse.

SY-5 (Nit). Lines 108-109. "sufficient" unsupported; state the `dataRepresentation()` round-trip need.

SY-6 (Nit). Lines 43-45. Uncited ZCode claim; duplicate of Goals lines 38-40.

SY-7 (Nit). Lines 62-64, 147, 158-159. "Natural size" undefined, test has no expected value.

SY-8 (Nit). Lines 115-116. No debounce interval.

SY-9 (Nit). Lines 119-121. Accessibility wording vague.

SY-10 (Nit). Lines 17-20, 39-40. "Native" filler; "one-click" subjective; `@` panel has no file.

SY-11 (Nit). Lines 117, 149-150, 157-159. "21 languages" vs "ten gates"; mutation bullet style; no observable for "reaches the model as an image".
