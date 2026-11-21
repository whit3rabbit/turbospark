# Audit: media generation, usage & limits, writing, voice

Paths are relative to swift/TurboSparkApp/Sources/TurboSparkApp. Design-system names refer to docs/design-system/components (Button, Card, Chip, Composer, EditorSurface, EmptyState, IconButton, ProgressBar, StatusDot, StatusLabel, PillPicker...).
Descriptions for the 11 slugs not fetched in full came from design_component_search (the full-record fetch dumps ~20k tokens each); verdicts for those rest on description plus app search.

| slug | verdict | value / effort |
| --- | --- | --- |
| audio-player | SKIP (already richer) | - |
| before-after | ADOPT (Images lightbox, compare to source) | M / S |
| edit-region | SKIP (no inpaint pipeline) | - |
| image-generation | IMPROVE (stage-aware progress, partial preview) | M / M |
| image-results | IMPROVE (batch grid + Vary/Use actions) | M / M |
| context-breakdown | SKIP (exists, near parity) | - |
| cost-meter | SKIP (cloud $), tokens/s already in status bar | - |
| credit-top-up | SKIP (billing) | - |
| guardrail-notice | SKIP (no server-side moderation) | - |
| model-unavailable | IMPROVE (load-failure/OOM notice with switch-model action) | M / S |
| quota-banner | SKIP as quota; IMPROVE as memory/thermal strip | M / S |
| usage-meter | SKIP (plan limits); ring already exists as context ring | - |
| ghost-completion | SKIP for chat composer (low value, costs a decode per pause) | - |
| inline-prompt | SKIP (no document editor surface) | - |
| prose-diff | IMPROVE (reuse HumanizeDiff in chat edits / Humanize accept-per-change) | M / M |
| selection-toolbar | ADOPT (Explain/Rewrite/Quote on selected message text) | M / M |
| version-history | SKIP (message versions + Rewind exist) | - |
| live-transcript | SKIP (AudioLiveTextView exists) | - |
| mic-picker | IMPROVE (add live level meter to the existing picker) | M / S |

## audio-player
Component: waveform scrubber, 5-step speed cycle, +-5s, download. Waveform is fake (seeded).
Existing: Audio/AudioPlaybackWaveform.swift (real peaks, Canvas, scrub gesture, RTL, high-contrast, native Slider for VoiceOver), Audio/AudioControlsView.swift (play, +-5s, slider, time field), Audio/AudioSpeechControls.swift (AudioSpeedControl), Audio/AudioTakeAuditionButton.swift.
Gap: none that matters. The app uses real peaks, the component fakes them. Hover-preview of scrub position and a cycle-speed pill are cosmetic.
Verdict: SKIP.

## before-after
Component: draggable divider comparing two images, labels fade near edges, keyboard 2%/10%/25%/Home/End.
Existing: Images/ImageGalleryCarousel.swift shows one image with prompt, size, seed and prev/next. No comparison view. App has no img2img/inpaint source image today (check Images/ImageGenerationSettings.swift before building; if there is no init-image path, this is only useful for upscale or quantization A/B).
Verdict: ADOPT, but gate on a pair existing. Value M, effort S.
Approach: `ImageCompareView` (two `Image(nsImage:)` in a ZStack, top one `.mask` with a Rectangle sized by a `@State fraction`; `DragGesture(minimumDistance: 0)` on the frame; handle is a Circle with `.ultraThinMaterial` and `chevron.left.chevron.right`). Accessibility: `.accessibilityRepresentation { Slider(value:in:) }` with percent value, `.accessibilityAdjustableAction`. Add as a second pane in the ImageGalleryCarousel sheet when two artifacts are selected (selecting mode already exists in ImageGalleryCard) or when an artifact has a parent. Labels as Chip. Theme tokens: `.appSurface`, `.appBorder`; no hard-coded white/black (use theme overlay tokens or material).
Maps to: Card (frame), IconButton (handle), Chip (labels).

## edit-region
Component: brush mask over image plus prompt, shimmer in masked area.
Existing: nothing. Image pipeline (Images/MLXImageGenerationSession.swift, QwenImageGenerationSession.swift) is text-to-image only per the grep (steps/stages, no mask input).
Verdict: SKIP. It is a feature (inpainting model + mask plumbing through Rust FFI), not a UI mimic. Revisit only if an inpaint-capable family is added (docs/NEW_MODEL.md).

## image-generation
Component: four progress treatments (noise to image, blur to sharp, scan line, 2x2 grid), prompt caption, Regenerate, step counter.
Existing: Images/ImagesSectionView.swift `generationProgress` (line ~227): "Image i of n", stage label (11 real stages), AppProgressBar, completed/total steps, in a material card. Unsaved result preview below it.
Gap: the card shows a bar only; the user stares at a bar for the whole denoise. The component's value is showing the image resolving. Also no Regenerate on the finished result (Use prompt exists in carousel; no re-roll with new seed in place).
Verdict: IMPROVE. Value M, effort M.
Approach: (1) while stage == "transformer", show the latest intermediate latent preview if the Rust pipeline can emit one cheaply (check `progress` callback in MLXImageGenerationSession.swift line ~165; if it only has stepIndex, skip previews). (2) Without previews, use the blur-to-sharp trick on a placeholder: a dot-field or blurred-prompt-color gradient with `.blur(radius:)` driven by `imageProgressFraction`, honoring `accessibilityReduceMotion` (fall back to opacity only). Cheap, no pipeline change. (3) Add a "Regenerate" `appAction(.secondary)` that reuses the request with a fresh seed, in `unsavedResult` and the carousel.
Do not copy the animated noise canvas (22 fps repaint competes with Metal denoise for the GPU/CPU on the same machine).
Maps to: ProgressBar, Card, Button.

## image-results
Component: 2x2 pick grid; hover actions Upscale / Vary / Download; select check; lightbox with prev/next and "Use this one".
Existing: Images/ImageGalleryCard.swift (thumb, select mode, open, reuse, trash), ImageGalleryCarousel.swift (preview with arrows via keyboard shortcuts, Use prompt, actions menu), ImagesSectionView batch index.
Gap: no "Vary" (same prompt, neighbor seed) and no per-card hover quick actions; batch results are presented as a gallery, not as "pick one of this run". Upscale is out of scope unless a model exists.
Verdict: IMPROVE. Value M, effort M.
Approach: group artifacts of one batch (same `imageRequest.prompt` and timestamp) into a run row at the top of ImagesSectionView with a "Keep this one" / trash others affordance; add `Vary` (seed+1 or random, same settings) and `Regenerate` to ImageGalleryCard's context menu and the carousel toolbar. SwiftUI: `.onHover` overlay with `AppIconButton`s, `.contextMenu`, `matchedGeometryEffect` for the open transition only if motion setting allows. Skip Upscale, skip the "2x" badge.
Maps to: Card, IconButton, Chip.

## context-breakdown
Component: stacked bar of context parts with legend, hover to highlight, "N% full".
Existing: State/AppContextUsage.swift (components: system, tools, memory, conversation, summary, attachments, draft; exact total plus apportioned rows), Generation/ContextUsageRingView.swift (ring in composer footer + `ContextUsagePopoverView` with stacked capsule bar, rows, Free space), Generation/ChatParitySheets.swift `ContextUsageSheet`, Chrome/StatusBarView.swift (meter plus "% until auto-compact" hint).
Gap: only hover-to-dim a segment (the popover has no per-segment hover detail line; `.onHover` per row/segment plus a detail line is S effort). Not worth a verdict beyond that. Note the app version is more honest (exact total vs apportioned rows).
Verdict: SKIP. Optional nit: segment hover dimming.

## cost-meter
Component: live tokens and dollars per run with budget cap.
Existing: tokens/s, memory, context in Chrome/StatusBarView.swift and ChromeTelemetryView.swift; Remote providers (Networking/RemoteProviderTypes.swift) are opt-in and have no cost UI.
Local mapping: a per-run budget cap is meaningful only as max_tokens/time; already covered by generation settings. A dollar figure only applies to the remote-provider path, which is a niche, opt-in layer.
Verdict: SKIP.

## credit-top-up
Verdict: SKIP. Billing UI; app has no credits.

## guardrail-notice
Component: calm note that part of a reply was withheld or a request declined, reason expandable.
Existing: Generation/PromptSafetyPills.swift `ForgeGuardrailsPillControl` (composer-side safety mode), tool approval views. Local models do not emit structured "withheld" signals; refusal is plain model text.
Verdict: SKIP. A notice would fabricate a distinction the engine cannot make. If the app later adds its own output filter (ForgeGuardrails), surface it with the existing `StatusLabel`/Chip + disclosure, not a new component.

## model-unavailable
Component: model busy notice, auto-retry countdown, switch-to-other-model button.
Existing: Components/ErrorBanner.swift (generic dismissible string), Generation/StallRecoveryBannerView.swift (Keep Waiting / Stop / Restart, good pattern), Chrome/ModelLoaderControl.swift, Components/LoadAnywayConfirmation.swift, Audio/AudioModelDownloadControls.swift. `model.error` is a bare String, so a failed model load (insufficient memory, corrupt install, unsupported family) becomes a one-line banner with no recovery action.
Gap: no actionable failure state with "Switch model" / "Load anyway" / "Retry". Countdown auto-retry is not meaningful locally (no server busy), skip that part.
Verdict: IMPROVE. Value M, effort S.
Approach: add a typed `ModelLoadFailure` (reason enum: memory, missing files, unsupported, other + message) alongside `model.error`; render a `ModelUnavailableBannerView` modeled exactly on StallRecoveryBannerView (strip above composer, one primary action). Actions: "Choose another model" (opens existing model picker pill), "Retry", "Load anyway" when the reason is memory (route to LoadAnywayConfirmation). Announce once for VoiceOver like the stall banner.
Maps to: Button, StatusLabel, StatusDot; follows the "attention strip" pattern.

## quota-banner
Component: near-limit and limit-reached card or slim strip above composer.
Existing as limit: none (no quotas). Local resource equivalents already surface piecemeal: context compaction hint in StatusBarView, memory pressure and thermal readouts in StatusBarView (lines ~36-39, 248), MusicMemoryAdvisor.swift for audio.
Gap: memory pressure and thermal are only in the bottom status bar; a user mid-chat on a loaded 30B model at "critical" pressure or "serious" thermal state gets no strip near the composer, and the "context almost full, will compact" hint is also only in the status bar.
Verdict: SKIP quota; IMPROVE as "resource strip". Value M, effort S.
Approach: one `ResourceWarningStrip` above the composer (same slot as StallRecoveryBannerView), shown when `telemetry.memoryPressure != normal`, thermal >= serious, or context fraction >= ~0.9 and compaction is off. Copy states the consequence and offers one action (Compact now / Unload model / Reduce context). Dismiss per level so it does not nag. Uses theme.warning, not literal orange. Make sure it only appears after a model is loaded (do not warn for unused features).
Maps to: StatusLabel + Button, StatusDot color semantics.

## usage-meter
Component: ring or 7-day bars for plan usage and reset.
Existing: Generation/ContextUsageRingView.swift (the ring, per conversation), SessionStatsSheet.
Gap: no per-day token history. A weekly bars chart of local tokens generated is a vanity metric with no limit to measure against; there is no reset.
Verdict: SKIP.

## ghost-completion
Component: grey inline suggestion after typing pause; Tab accept, Esc dismiss.
Existing: Generation/ComposerAutocompleteController.swift (`/` and `@` popup, Esc dismissal logic) but no next-token ghost text.
Reasons to skip: (1) it contends with the user's main generation for the single loaded model (one session, Metal queue); a completion request every pause would stall or evict chat decode; (2) the composer is NSTextView-backed and overlaying grey text correctly with IME, wrapping and Dynamic Type is expensive; (3) quality from the same chat model for "continue my prompt" is low.
Verdict: SKIP. If ever wanted, it needs a small dedicated draft model and must be off by default.

## inline-prompt
Component: Cmd-K bar over a paragraph, rewrite preview, accept/discard.
Existing: Humanize section (Humanize/HumanizeView + HumanizeController) is the text-rewrite surface with diff and rules; chat has message edit + regenerate (Generation/MessageEditingViews.swift).
Verdict: SKIP. No document editor in the app to host it. Humanize already covers "rewrite then review".

## prose-diff
Component: inline word diff, accept/reject per change or all.
Existing: Humanize/HumanizeDiff.swift (word-level diff, bounded work, noise folding, display-only, off-main compute) rendered in two aligned cards. Copy always sends the whole rewrite.
Gap: review is look-then-copy-all; no accept/reject per change.
Verdict: IMPROVE (narrow). Value M, effort M.
Approach: in Humanize, add per-change toggles. Data: give each `HumanizeDiff` inserted/deleted run a stable id, keep `rejected: Set<ID>` in HumanizeController, and compute `acceptedText` by replaying the diff (rejected inserted runs dropped, rejected deleted runs restored). Render as AttributedString runs with `.appSuccess`/`.appDanger` styling (and underline/strike, not color alone, per DESIGN.md accessibility), a popover or inline `Button` per run, and "Accept all / Reject all" in the header. Must keep the invariant that diff is display only: the accepted string is a new derived value, original `rewrite` stays untouched (mutation-check this). Do not build it for chat messages; no use case.
Maps to: EditorSurface, Button, StatusLabel colors via theme styles (design check ratchets raw colors).

## selection-toolbar
Component: floating toolbar over selected text with Improve, Shorten, Translate.
Existing: message text is `.textSelection(.enabled)` (Generation/MessageRowView.swift lines ~223, 233) with only the system menu. No selection actions in chat or Audio transcript editor (Audio/AudioTranscriptEditorView.swift).
Verdict: ADOPT, adapted. Value M, effort M.
Approach: SwiftUI `.textSelection` does not expose the selection range, so a floating toolbar over selected text is not feasible on the existing Text. Options: (a) macOS 15+/26 `TextSelection` binding is only for TextEditor/TextField, not read-only Text; (b) wrap assistant message body in an NSTextView-backed view (costly, loses markdown rendering path) or (c) go the native route: register a Services menu item / `contextMenu(forSelectionType:)`-free approach by adding actions to the existing right-click menu via an `NSViewRepresentable` hook. Pragmatic plan: context-menu actions "Explain selection", "Rewrite", "Quote in reply" that read `NSPasteboard`/Services selection, feeding the composer as a quoted block. Quote-in-reply is the highest-value piece (S effort) and needs no model call. Treat true floating toolbar as L and not worth it. Strings must go through bundled `Text`, 10-locale parity.
Maps to: Composer (quote insertion), IconButton/PillPicker for the toolbar if built.

## version-history
Component: history sidebar, timeline scrub, compare two versions.
Existing: message versions `< n / m >` (Generation/MessageEditingViews.swift:17; AppChat.swift:323 inactive earlier versions), Generation/ChatParitySheets.swift `RewindSheet`, chat persistence. Humanize keeps one run.
Gap: nothing worth adding. A compare view across message versions would reuse HumanizeDiff cheaply (S) but nobody has asked; fold into prose-diff if done.
Verdict: SKIP.

## live-transcript
Component: running transcript with words appearing as spoken.
Existing: Audio/AudioLiveTextView.swift (committed text never revised, in-flight word set apart, announces state not each word, auto-scroll end anchor, Reduce Motion), Audio/AudioLiveTranscriber.swift, VoiceCallController for chat voice call.
Gap: none. It already handles the hard accessibility rule (do not announce each word) that the React component ignores.
Verdict: SKIP.

## mic-picker
Component: input device dropdown plus level meter to confirm the mic hears you.
Existing: Audio/AudioRecordBar.swift `microphonePicker` (Picker menu: Default microphone + devices via `AudioCapture.microphones()`, CoreAudio), recording-time level waveform (`AppLevelWaveform`, `controller.recordingLevels`, `inputLevel` computed as RMS at AudioCapture.swift:523). VoiceCallSettingsSection has no device choice.
Gap: no level meter BEFORE recording, so the user cannot verify the chosen mic until a take starts; and the Voice Call feature (App/VoiceCallSettingsSection.swift, Audio/VoiceCallController.swift) offers no device picker at all. Also no hot-plug refresh indicator.
Verdict: IMPROVE. Value M, effort S.
Approach: add a short-lived "test input" tap: a small `AppLevelWaveform` or `MeterBar` row under the picker that starts an AVAudioEngine input tap only while the picker popover/section is visible and permission is already granted (do NOT request the mic prompt from a picker; AGENTS rule: only `RecordingPermissions` asks, after user chooses Allow). Reuse the RMS computation from AudioCapture. Add the same picker to the Voice Call settings section, persisting the device UID (not the volatile AudioObjectID, which changes across reboots/replug). Respect Reduce Motion by showing a static bar value. Stop the tap on disappear.
Maps to: PillPicker / Picker, ProgressBar (meter), StatusDot (permission state).

## Ranking of worthwhile work
1. mic-picker level meter + Voice Call device (M value, S)
2. model-unavailable typed load failure strip (M, S)
3. resource warning strip from quota-banner idea (M, S)
4. selection quote/explain actions (M, S for Quote in reply)
5. image-generation blur-to-sharp progress + Regenerate (M, M)
6. image-results Vary and batch pick (M, M)
7. prose-diff per-change accept in Humanize (M, M)
8. before-after only once an img2img or upscale pair exists (M, S)
