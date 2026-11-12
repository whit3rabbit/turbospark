# Audio

## Overview
The macOS app's Audio workspace: recording (microphone and app/system audio), live and final speech-to-text, speaker labeling and memory, text-to-speech and voice cloning, music generation, meeting notes, and the encrypted audio library with search. A MainActor `AudioWorkspaceController` owns state, model residency and persistence, and dispatches each model to a runtime (Rust worker, Qwen MLX, or the FluidAudio adapter). Views are thin SwiftUI surfaces over the controller. Persistence goes through the encrypted profile vault, and recording permissions are requested only from user-driven sheets.

## Interface
Capture and recording
- `AudioCapture` (start, pause, resume, stop, stopAcceptingAndDrain), `AudioCaptureTimeline`, `AudioCaptureBudget` [AudioCapture.swift:AudioCapture]
- `RecordingPermissions.shared`, `RecordingPermission`, `RecordingPermissionDependencies` [RecordingPermissions.swift:RecordingPermissions]
- `RecordingAccessSheet`, `View.recordingAccessGate(_:permissions:)` [RecordingAccessSheet.swift:recordingAccessGate]
- Recording lifecycle: `prepareRecording`, `startRecording`, `stopRecording`, `addMarker`, `AudioRecordingSetupDependencies` [AudioWorkspaceRecording.swift:AudioRecordingSetupDependencies]
- `AudioRecordBar`, `AudioCaptureSourcePickers`, `AudioLiveLevels` [AudioRecordBar.swift:AudioRecordBar]; `AudioRecordingPanel` [AudioRecordingPanel.swift:AudioRecordingPanel]
- Presets and tags: `AudioRecordingPreset`, `AudioLibraryOrganizer` [AudioLibraryOrganization.swift:AudioLibraryOrganizer]; `AudioRecordingPresetPicker`, `AudioLibraryFilterBar`, `AudioTagEditor` [AudioLibraryFilterBar.swift:AudioLibraryFilterBar]

Live text
- `AudioLiveFeed`, `AudioLiveConverter`, `AudioLiveMixer` [AudioLiveFeed.swift:AudioLiveFeed]
- `AudioLiveTranscriber`, `AudioLiveStreaming` [AudioLiveTranscriber.swift:AudioLiveTranscriber]
- `AudioLiveText`, `startLiveText`, `persistLiveDraft`, `liveStreamingUnavailableReason(for:)` [AudioWorkspaceLive.swift:persistLiveDraft]
- `AudioLiveTextView`, `AudioLiveTextModelRow` [AudioLiveTextView.swift:AudioLiveTextView]

Media I/O and export
- `AudioMediaIO` (wav, importFile, readMono16k, readMono24k, render, encodeLossless, encodeM4A, commitExport) [AudioMediaIO.swift:AudioMediaIO]
- `AudioExportPreset`, `AudioExportPanel` [AudioExportPanel.swift:AudioExportPanel]
- Controller import, playback, export, `scratchURL`, `AudioOperationLifetime` [AudioWorkspaceMedia.swift:AudioOperationLifetime]
- `AudioTranscriptExport.data(_:format:)` [AudioSpeakerRoster.swift:AudioTranscriptExport]

Library, search, vocabulary
- Data model: `AudioLibraryItem`, `AudioLibrarySegment`, `AudioVoiceReference`, `AudioRecipe`, `AudioWorkspacePage`, `AudioLibraryStore` [AudioLibrary.swift:AudioLibraryItem]
- `AudioLibraryView`, `AudioItemDetailView` [AudioLibraryView.swift:AudioLibraryView]; `AudioLibraryList` [AudioLibraryList.swift:AudioLibraryList]
- Search: `AudioLibrarySearch`, `AudioSearchChunker` [AudioLibrarySearch.swift:AudioLibrarySearch]; `AudioSearchFusion.fuse`, `AudioSemanticSearch` [AudioSearch.swift:AudioSearchFusion]; `AudioSearchResultsView` [AudioSearchResultsView.swift:AudioSearchResultsView]
- `AudioCustomVocabulary.apply`, `promptContext`, `AudioCustomVocabularyStore` [AudioCustomVocabulary.swift:AudioCustomVocabularyStore]; `AudioCustomVocabularySheet` [AudioCustomVocabularySheet.swift:AudioCustomVocabularySheet]
- Prompts: controller `savePrompt`, `loadPrompt` [AudioPromptLibrary.swift:savePrompt]; `AudioPromptLibraryView` [AudioPromptLibraryView.swift:AudioPromptLibraryView]

Speakers
- `AudioSpeakerLabels` (turns, assign), `labelSpeakers`, `labelSpeakersBestEffort` [AudioSpeakerLabels.swift:AudioSpeakerLabels]
- `AudioSpeakerParts.plan` / `join` [AudioSpeakerParts.swift:AudioSpeakerParts]
- `AudioSpeakerMemory`, `AudioSpeakerProfileStore` [AudioSpeakerMemory.swift:AudioSpeakerMemory]
- `AudioSpeakerRoster`, `AudioSpeakerSummary` [AudioSpeakerRoster.swift:AudioSpeakerRoster]; `AudioSpeakerNamingSheet` [AudioSpeakerNamingSheet.swift:AudioSpeakerNamingSheet]

Controller and jobs
- `AudioWorkspaceController` [AudioWorkspaceController.swift:AudioWorkspaceController]; model install, adopt, delete [AudioWorkspaceController+Installs.swift:deleteModel]
- Request building and run loop: `request(from:task:)`, `run`, `transcribe`, `summarizeMeeting` [AudioWorkspaceJobs.swift:request]; `AudioTranscriptionPlan.windows` [AudioTranscriptionPlan.swift:AudioTranscriptionPlan]
- `AudioRuntimeProvider` [AudioRuntimeProvider.swift:AudioRuntimeProvider]; FluidAudio loading [AudioWorkspaceFluid.swift:residentFluidTranscriber]; `Qwen3TTSSpeechJob` [Qwen3TTSSpeechJob.swift:Qwen3TTSSpeechJob]
- Progress: `AudioGenerationProgress`, `AudioSpeechRunProgress` [AudioGenerationProgress.swift:AudioGenerationProgress]
- `completeFinalTranscription` [AudioWorkspaceTranscriptionCompletion.swift:completeFinalTranscription]
- Meeting notes: `AudioMeetingNotesView` [AudioMeetingNotesView.swift:AudioMeetingNotesView]; `AudioMeetingNotesAdvisor` [AudioMeetingNotesAdvisor.swift:AudioMeetingNotesAdvisor]

Models and catalog
- `AudioCatalogSnapshot`, `AudioInstallStatus`, `AudioOfflineInstalls` [AudioOfflineInstalls.swift:AudioInstallStatus]
- `AudioEngineLabel`, `AudioEngineSummary` [AudioEngineLabel.swift:AudioEngineLabel]; `AudioModelState`, `AudioModelStateLabel` [AudioModelStatus.swift:AudioModelState]; `AudioModelLanguage` [AudioModelLanguage.swift:AudioModelLanguage]
- `AudioModelPickerView`, `AudioModelDownloadControls`, `AudioModelGateView`, `modelGate` [AudioModelPickerView.swift:AudioModelPickerView] [AudioModelDownloadControls.swift:AudioModelDownloadControls] [AudioModelGateView.swift:AudioModelGateView] [AudioWorkspaceModelGate.swift:modelGate]

Speech-to-text surface
- `SpeechToTextMode`, `SpeechToTextLibrary` [SpeechToTextMode.swift:SpeechToTextMode]; `SpeechToTextJob`, `SpeechToTextModels` [SpeechToTextModels.swift:SpeechToTextModels]
- `SpeechToTextWorkspaceView` [SpeechToTextWorkspaceView.swift:SpeechToTextWorkspaceView]; panes [SpeechToTextCapturePane.swift:SpeechToTextCapturePane] [SpeechToTextAnyAudioPane.swift:SpeechToTextAnyAudioPane]; `SpeechToTextModelPanel` [SpeechToTextModelPanel.swift:SpeechToTextModelPanel]
- `AudioTranscriptEditorView` [AudioTranscriptEditorView.swift:AudioTranscriptEditorView]

Speech, cloning, music
- Speech routing and consent: `cloneConsentGiven`, `speechValidationMessage` [AudioWorkspaceSpeech.swift:cloneConsentGiven]; shared controls [AudioSpeechControls.swift:AudioVoicePicker] [SpeechInputParts.swift:VoiceCloneConsentControls]
- Voice Cloning: `VoiceCloneWorkspaceView` [VoiceCloneWorkspaceView.swift:VoiceCloneWorkspaceView]; steps and blockers [AudioWorkspacePresentation.swift:VoiceCloneRunBlocker]; `VoiceCloneAttention` [VoiceCloneAttention.swift:VoiceCloneAttention]; reference helpers [AudioWorkspaceVoiceReference.swift:voiceReferenceRecordingLimit]; `VoiceReferenceCard` [VoiceReferenceCard.swift:VoiceReferenceCard]
- Music: `MusicMemoryAdvisor` [MusicMemoryAdvisor.swift:MusicMemoryAdvisor]; `MusicPromptBuilder` [MusicPromptBuilder.swift:MusicPromptBuilder]; `MusicLyricsGuide` [MusicLyricsGuide.swift:MusicLyricsGuide]; `AudioMusicInputs` [AudioWorkspaceInputs.swift:AudioMusicInputs]

Shell and navigation
- `AudioWorkspaceView` [AudioWorkspaceView.swift:AudioWorkspaceView]; `AudioSidebarPanel` [AudioSidebarPanel.swift:AudioSidebarPanel]; `AudioDestinationMenu` [AudioNavigationView.swift:AudioDestinationMenu]
- Shared controls: `AudioTransportView`, `AudioExportMenu`, `AudioDeleteConfirmation`, `AudioWorkspaceStatusView` [AudioControlsView.swift:AudioTransportView]

Children
- [FluidAudio](./FluidAudio/EGM.md): adapter to the FluidAudio Core ML package (only importer of it)

## Invariants
- Capture and views only check macOS access; only `RecordingPermissions.request` after a user action prompts. Declining never blocks features that do not need it [RecordingPermissions.swift:RecordingPermissions] [AudioCapture.swift:microphoneAccessAllowed].
- The capture sink must commit synchronously and never wait on MainActor; `stopAcceptingAndDrain` is the barrier after which no sink call occurs [AudioCapture.swift:stopAcceptingAndDrain].
- Every async continuation in the controller checks `valid(token)` (not shut down, epoch and vault profile unchanged) before persisting [AudioWorkspaceController.swift:valid].
- Unsaved items stay in `pendingSaves` and pinned until a save succeeds; transcript edits are staged then saved, and a failed save keeps the pending edit [AudioWorkspaceController.swift:save].
- Undecodable or invalid library records are copied aside and skipped, never overwritten; optional fields keep older records decodable [AudioLibrary.swift:loadReporting].
- Meeting summarization never modifies the transcript; it writes only the summary [AudioMeetingNotesView.swift:groundTruthBanner].
- Voice clone consent covers one exact reference region (`consentedReference` must equal the current reference); reusing a take's reference or changing the trim never carries consent [AudioWorkspaceSpeech.swift:cloneConsentGiven] [AudioWorkspaceVoiceReference.swift:useVoiceReference] [AudioWorkspaceJobs.swift:nativeCloneRequest].
- Inference PCM reads are bounded (`readMono16k` max 30 s, `readMono24k` 1 to 30 s); whole meetings are never held in memory; PCM is validated (8k to 192k Hz, 1 to 8 channels, finite) [AudioMediaIO.swift:readMono16k].
- Live feed `offer` does O(1) work under a short lock and drops overflow; the saved recording and final pass never depend on live text, and a backlog past `backlogLimit` stops live text for that take [AudioLiveFeed.swift:offer] [AudioLiveTranscriber.swift:pump].
- Dispatch asks `AudioRuntimeProvider`, not family strings; unsupported profiles are hidden from pickers [AudioRuntimeProvider.swift:AudioRuntimeProvider].
- Only catalog-verified installs reach FluidAudio, never a user-picked folder; loads are admitted within min(3/4 RAM, Metal working set) [AudioWorkspaceFluid.swift:residentFluidTranscriber] [MusicMemoryAdvisor.swift:evaluate].
- Resident sessions are released before model weights are deleted, to avoid crashes on mapped files; delete is refused while busy or recording [AudioWorkspaceController+Installs.swift:deleteModel].
- Speaker labels are written only onto the named transcript revision and skip manual labels; labeling failure never fails a saved transcript [AudioSpeakerLabels.swift:labelSpeakers] [AudioWorkspaceTranscriptionCompletion.swift:completeFinalTranscription].
- Speaker names and merges live on the take roster by key, not in segments; `merge` refuses self and cycles [AudioSpeakerRoster.swift:merge].
- Search embeds chunks of at most 480 tokens (model truncates silently past 512); the indexer pauses while recording or busy [AudioLibrarySearch.swift:AudioSearchChunker.tokenBudget].
- Transcript windows must satisfy duration > 0, length > 0, 0 <= overlap < length, with ownership cut mid-overlap [AudioTranscriptionPlan.swift:windows].
- State is never shown by color alone (glyph plus word), and mutating actions are disabled during recording activity or while busy [AudioModelStatus.swift:AudioModelStateLabel] [AudioControlsView.swift:AudioTransportView].
- Music lyrics tags and prompt vocabulary are literal English model tokens and are never localized; empty lyrics map to `[instrumental]` [MusicLyricsGuide.swift:sectionTags] [AudioWorkspaceJobs.swift:effectiveLyrics].

## Side Effects
- Creates and destroys a private CoreAudio aggregate device and process tap; starts `AVAudioEngine` and ScreenCaptureKit streams [AudioCapture.swift:makeSystemAudioTap].
- Shows the microphone and screen-recording prompts and opens System Settings, on user action only [RecordingPermissions.swift:request].
- Reads and writes `audio:item:` and `audio:preferences` records in the profile database; updates the search index and vectors [AudioLibrary.swift:AudioLibraryStore] [AudioLibrarySearch.swift:embed].
- Reads and writes `custom_vocabulary.json` and `speaker_profiles.json` under `AppStorageRoot.directory` [AudioCustomVocabulary.swift:AudioCustomVocabularyStore] [AudioSpeakerMemory.swift:AudioSpeakerProfileStore].
- Writes decrypted WAV scratch files under `Caches/TurboSpark/<profileID>/decrypted` (mode 0700) and deletes them after use [AudioWorkspaceMedia.swift:scratchURL].
- Stores recording, import, and take audio as managed assets; writes export files after `NSSavePanel` [AudioWorkspaceRecording.swift:startRecording] [AudioMediaIO.swift:commitExport].
- Acquires `ImageJobCoordinator.shared` and `NativeHeavyWorkPermit` around jobs, drafts and embeddings [AudioWorkspaceJobs.swift:beginRun] [Qwen3TTSSpeechJob.swift:generate].
- Opens and closes model sessions, clears the MLX cache, compiles Core ML models for the Neural Engine, and manages security-scoped model access [AudioWorkspaceController.swift:residentSession] [AudioWorkspaceFluid.swift:residentFluidTranscriber].
- Spawns a detached thread and a 100 ms pump timer per live transcription; the controller runs a 0.25 s tick timer [AudioLiveTranscriber.swift:start] [AudioWorkspaceController.swift:init].
- Plays audio via `AVAudioPlayer`; writes to `NSPasteboard.general` and `NSSavePanel` from notes and prompt views [AudioWorkspaceMedia.swift:finishPreparingPlayback] [AudioMeetingNotesView.swift:copyToClipboard].

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
