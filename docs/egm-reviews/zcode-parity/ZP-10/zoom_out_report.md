# Stage 5: Initial zoom out report (ZP-10 whiteboard canvas)

Target: docs/designs/zcode-parity/ZP-10-whiteboard-canvas.md (full file, 163 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. Verdict: no Blockers, revise before approval.

## Zoom-out reviewer (ZO)

ZO-1 (Major). Lines 17, 77, 136-139. The "side-pane tab" host is not established. `Generation/SplitChatPanes.swift` renders split chats (`SplitChatPanesView`, `promoteSplitPane`), not tabbed side panes, and has no pane-target concept. Nearest hosts are `ArtifactPanelView` and the inspector specs. Fix: name the real host and extension seam, or declare a prerequisite.

ZO-2 (Major). Lines 38-40, 140-143. Composer intake is written as a new modification, but `AppModel.addPromptAttachment(_:toChatID:)` (`State/AppModel+Chat.swift:430`) and `AttachmentImporter.importDocuments` already accept image files. Fix: write the PNG to an app-owned file and call the existing seam; drop item 4 or justify it. The "@ attachment panel" goal has no file or design: specify or cut.

ZO-3 (Major). Lines 85-88, 131-135. Persistence is an undefined hybrid: "archive" is never defined, storage root, file naming and orphan cleanup on board delete are missing, "profile state root" is unverified, and the quarantine test targets undefined behavior. Fix: reuse the `AppearanceStore`/`DisabledItemStore` pattern, define the layout, define quarantine or drop the test.

ZO-4 (Major). Lines 49-53. Board scoping stays open while Goals and CUJs already commit to per workspace. ZP-21 and ZP-23 interact with the export target (which chat receives the PNG when several chats run) and are not mentioned. Fix: record per workspace as a decision and state the target chat.

ZO-5 (Major). Lines 76-90, 123-135. Technical Plan drops to implementation altitude (`drawingPolicy`, coordinator undo, debounce). A custom toolbar re-implements `PKToolPicker` with no justification. Board list management (create, rename, delete, switch) is thin. Fix: justify the custom toolbar against DESIGN.md or use the picker; specify board management.

ZO-6 (Major). Lines 91-93, 115-116, 152-159. Platform and verification claims unverified: PencilKit on macOS via `NSViewRepresentable`, mouse drawing, minimum OS and deployment target, off-main-thread export. Verification is mostly manual. Fix: cite the minimum OS and add automated coverage.

ZO-7 (Nit). Lines 62-64. "Natural size" and background (transparent vs white) undefined; a transparent PNG may render poorly for a vision model.

ZO-8 (Nit). Lines 118, 157. "21 languages" versus INDEX "ten localization parity gates". Reference the gate by name.

ZO-9 (Nit). Lines 97-111. Both alternatives are "Rejected by author" with no evidence. Give a real rationale.

ZO-10 (Nit). Lines 126-131. New `Whiteboard/` directory: check repo convention.

ZO-11 (Nit). Lines 3-7 and INDEX. If a side-pane host must be built, "medium" size is understated.

## Inventive catalyst (IC)

IC-1 (Major). Lines 83-89, 131-135. Two-layer store (AppJSONStore metadata plus blob files) has no consistency rule: a crash between writes leaves orphans or dangling references; the test covers only corrupt metadata. Fix: blob is source of truth, atomic temp-and-rename writes, metadata rebuildable with a load-time reconcile, or one blob per board with title in the filename. Confirm `AppJSONStore` fits a per-workspace collection and flush the debounce on quit.

IC-2 (Major). Lines 62-64, 115, 140-143. Export size is unbounded: no clamp to drawing bounds, no max pixel dimension from model image limits, no toast on downscale. "Same path as a pasted image" is asserted; no pasted-image symbol found in `PromptComposerView.swift`. "Off the main thread" is risky for PencilKit rendering. Fix: clamp and downscale, name the attachment API, relax or justify the threading claim.

IC-3 (Major). Lines 17-20, 91-93, 97-103. "Native PencilKit on macOS" is taken as given with no stated deployment target; `drawingPolicy` is iOS-lineage. Alternative A's palm rejection and pressure cons are irrelevant to a mouse-driven Mac app, so the real risk (availability) is unexamined. Fix: first-task feasibility spike with the result recorded; keep any abstraction seam minimal.

IC-4 (Major). Lines 36-37, 79-80, 117-122. Custom toolbar re-implements `PKToolPicker`, forcing a state-mapping layer and 21-language strings. "Undo/redo ride the coordinator" is vague (undoManager lifecycle on board switch). Fix: evaluate `PKToolPicker` first, state the choice.

IC-5 (Major). Lines 31-39, 49-53. Scoping already decided in practice; no concurrency semantics when several chats or tabs (ZP-21, ZP-23) share a board. Fix: close the question or define single-writer or read-only second pane.

IC-6 (Nit). Lines 118, 154-159. "21 languages" versus "ten gates"; export-to-composer seam has no automated test. Move export into a pure testable helper.

IC-7 (Nit). Lines 3, 127-150. Overlap with ZP-11 (image annotation): both need a drawing surface and PNG export, neither references a shared component. Also "@ attachment panel" absent from target files, mutation-check bullet is low altitude, auto-naming scheme and collision rule unspecified.
