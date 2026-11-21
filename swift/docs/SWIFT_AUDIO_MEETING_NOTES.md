# Swift Audio and Meeting Notes Architecture

This document describes the architecture, data models, audio capture pipelines,
speaker intelligence, and post-meeting notes workflow in TurboSparkApp.

---

## 1. Architectural Overview

The audio meeting note-taking system provides high-fidelity, privacy-preserving
meeting transcription, speaker identification, and structured note generation on
Apple Silicon.

```
+----------------------------------------------------------------------------------+
|                              AudioStudio Workspace                               |
|   SpeechToTextWorkspaceView  <-->  AudioTranscriptEditorView  <-->  LibraryView  |
+----------------------------------------------------------------------------------+
                                        |
      +---------------------------------+---------------------------------+
      |                                 |                                 |
      v                                 v                                 v
+--------------------+        +--------------------+        +--------------------+
|  Phase 5: Capture  |        |  Phase 4: Speaker  |        |  Phase 2: Notes    |
|  AudioCapture      |        |  AudioSpeakerMemory|        |  AudioMeetingNotes |
|  - CATapDescription|        |  - Mic RMS Energy  |        |  - Detail Switcher |
|  - ScreenCaptureKit|        |  - vDSP Voiceprints|        |  - Qwen 3.5 Advisor|
|  - Aggregate Drift |        |  - Cosine Matching |        |  - Timeline Chips  |
|  - WAV Ring Buffer |        |  - Naming Sheet    |        |  - Immutability    |
+--------------------+        +--------------------+        +--------------------+
                                        |
                                        v
                              +--------------------+
                              |  Phase 1: Jargon   |
                              |  AudioCustomVocab  |
                              |  - Regex Word \b   |
                              |  - Prompt Context  |
                              |  - Dictionary Sheet|
                              +--------------------+
```

---

## 2. Phase 5: Audio Capture Overhaul

### CoreAudio Process Tap (`CATapDescription`)

Prior versions relied solely on ScreenCaptureKit (`SCStream`) for system audio.
While functional, ScreenCaptureKit triggers macOS Screen & System Audio
Recording permissions prompts and displays an invasive purple recording indicator
in the menu bar.

On macOS 14.2+, `AudioCapture` uses native CoreAudio HAL process taps:
- **API**: `CATapDescription(stereoGlobalTapButExcludeProcesses:)` via AudioHardware
  private framework bindings or standard HAL tap objects.
- **Permission**: Declares `NSAudioCaptureUsageDescription` in `Info.plist`.
- **User Experience**: Captures system audio without requesting Screen Recording
  permissions and avoids the purple screen-sharing indicator.
- **Fallback**: Automatically falls back to ScreenCaptureKit on macOS 14.0 through 14.1
  or when HAL tap initialization fails.

### Aggregate Device and Timeline Alignment

Microphone input (`AVAudioEngine`) and system audio (`CATapDescription` or
`SCStream`) arrive on distinct clocks.
1. `AudioCapture` runs a drift-compensating ring buffer that resamples both
   streams to uniform 16 kHz or 24 kHz mono floating-point buffers.
2. Channel energy is tracked per sample window (512 samples) to support
   microphone vs system energy comparison.
3. Pause intervals are excised cleanly from the persistent timeline, ensuring
   recording markers and transcript offsets correspond strictly to active audio.

### Live background-noise reduction

"Reduce background noise" (`AudioWorkspacePreferences.reduceLiveNoise`, off
by default, shown beside the live text model only when the
`voice-isolator-v1-realtime` install is present and can run) cleans the
microphone for live text only (`Audio/AudioLiveVoiceCleaner.swift`):
- On the transcriber's queue, microphone packets from the live feed are
  converted to 48 kHz mono, pushed through `AudioEnhanceStream`, and the
  released voice goes to the mixer, which resamples it to 16 kHz. The system
  audio tap is never cleaned. Capture persists its own copy of every packet,
  so the saved recording and the final pass are the original audio.
- Timing: the stream's output is sample-aligned with its input and released
  about 63 ms (plus compute) later. Each released sample is placed back at
  its input's timeline offset, so live word times stay on the recording
  timeline; words only appear that much later.
- Until the stream opens, the microphone passes raw. If pushes report `late`
  for more than 1 s of audio in a row, the feed dropped packets, or more than
  2 s of microphone audio waits in one drain, the tail is flushed, cleaning
  stops for the rest of the take, live text hears the raw microphone, and
  `AudioLiveText.cleaningStatus` shows one line. A failed open or push does
  the same. Tests: `AudioLiveNoiseReductionTests`.

---

## 3. Phase 4: Speaker Intelligence & Memory

### Local Speaker ("You") Energy Isolation

When a meeting is recorded with both microphone and system audio enabled,
identifying the local user is accomplished via comparative RMS energy:
- For each diarized speech turn `[start, end]`:
  $$\text{Energy}_{\text{mic}} = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} x_{\text{mic}}[i]^2}$$
  $$\text{Energy}_{\text{sys}} = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} x_{\text{sys}}[i]^2}$$
- When $\text{Energy}_{\text{mic}} > 2.5 \times \text{Energy}_{\text{sys}}$ across
  the majority of speech frames, the turn is classified as local.
- `AudioSpeakerNamingSheet` presents a 1-click `You (Microphone)` chip next to
  the speaker name field.

### Voiceprint Extraction via Accelerate vDSP

For persistent cross-meeting speaker identification, `AudioSpeakerMemory`
extracts acoustic embeddings using vDSP:
1. **Windowing**: Extracts up to 15 seconds of clean speech per speaker across
   their longest continuous turns.
2. **STFT**: 512-point Discrete Fourier Transform with a Hann window and 50% overlap.
3. **Log-Mel Filterbank**: 64 triangular mel filterbanks from 80 Hz to 7600 Hz.
4. **Voiceprint Vector**: Mean-pooled and L2-normalized 64-dimensional acoustic
   embedding stored in `AudioSpeakerVoiceprint(vector: [Float])`.

### Cross-Meeting Cosine Similarity

Speaker profiles are persisted in `~/Library/Application Support/TurboSpark/speaker_profiles.json`:
- **Matching**: Cosine similarity between candidate turn voiceprints and stored
  profiles:
  $$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Threshold**: Matches with cosine similarity $\ge 0.78$ generate name
  suggestions with confidence percentages (for example, `Alice (87%)`).
- **Learning**: When the user names or confirms a speaker in `AudioSpeakerNamingSheet`,
  `AudioSpeakerMemory.recordSpeakerNaming(...)` updates the profile store with
  the new voiceprint embedding.

---

## 4. Phase 2: Post-Meeting Notes & Summary

### Meeting layout: the user's notes beside the transcript

Speech-to-Text's Meeting mode shows a take in two columns
(`AudioMeetingColumns`, `Audio/AudioMeetingLayout.swift`): the user's own
notes on the leading side, and `[ Transcript | Summary | Ask ]` on the
trailing side. The columns state ideal widths scaled by text size, so the pair
stacks under Extra Large text or a narrow window. The Library and the other
modes keep the single column with `[ Transcript | Notes & Summary ]`.

- **My notes** (`AudioMeetingNotesEditor`) works during the recording too: the
  live view puts it beside the words as they are heard, and the word still
  being heard (`AudioLiveText.openWord`) is drawn lighter after the committed
  text. Typing stages `pendingUserNotes`; a 600 ms debounce, losing focus,
  leaving the view, Retry, and shutdown save it through `persistMutation`,
  which re-reads the record, so capture autosaves and notes saves never
  overwrite each other (`AudioMeetingWorkspaceTests`).
- `item.userNotes` is the user's text only. No model output writes it; notes
  generation and Ask read it.
- The Meeting and Voice Notes lists have their own search field, a scan of
  title, notes, summary, tags and transcript (`SpeechToTextLibrary.matching`).
  The Library keeps the indexed words, meaning and sound search.

### Generating notes

`AppModel.summarizeAudioTranscript` (prompts in `AudioMeetingNotesPrompt`):
- A transcript that fits one chunk (about 8k characters) gets the final format
  in one call: Summary, Key Decisions, Action Items, Discussion.
- A longer one is read in parts (terse bullets with `[m:ss]` times), then
  merged six at a time, level by level. Only the last merge writes the final
  format.
- The user's notes go into the final step as the points that matter most.
- The final call streams into `summaryPreview`, which the Summary tab draws
  while it runs; a stage line covers the part and merge calls. Only the
  finished result is saved, into `item.summary`.
- "Write notes after each meeting" (`AudioWorkspacePreferences.autoMeetingNotes`,
  off by default) runs this after a Meeting take's final pass, with the loaded
  chat model or the advisor's pick.

### Ask

The Ask tab answers questions from one meeting, or with "Search all meetings"
from every Meeting take (`Audio/AudioMeetingAsk.swift`,
`AudioMeetingAskView.swift`):
- Passages are the user's notes, the summary, and transcript runs of about 360
  words with speaker names, built in memory from the take.
- When they fit about 14k characters the model gets all of them. Past that,
  BM25 over the question's words and EmbeddingGemma 2 meaning hits from the
  library index (`AudioLibrarySearch.meaningHits`) are fused by reciprocal
  rank (k = 60), the user's notes always kept, and the chosen passages are
  sent in meeting and time order.
- The answer streams through `AppModel.runAudioPrompt`, cites `[m:ss]`, and
  lists the moments it drew on as buttons that play from there. Answers live
  for the session only and are never saved.
- Ask runs after a recording ends. Meaning hits need the take indexed, which
  happens after the final pass; until then, and without EmbeddingGemma 2
  installed, ranking is by words.

### Model Recommendation & Advisor

Summarization runs through local LLMs (`AppModel`). To assist users in picking
appropriate lightweight models, `AudioMeetingNotesAdvisor` ranks installed models:
- **Tier 2 (Explicit Recommendation)**: `Qwen 3.5 1.5B` and `Qwen 1.5B` variants.
  Marked with `Recommended: Fast, 32k context`.
- **Tier 1 (General Recommendation)**: Lightweight models under 4B parameters or
  Qwen-family models. Marked with `Recommended: High speed`.
- **Tier 0 (Standard)**: All other installed text models.

### Timeline Jump Chips

The summary Markdown parser extracts temporal references and speaker attributions:
- Markdown lists and sections render interactive timestamp jump chips
  (for example, `[04:12]`).
- Clicking a jump chip seeks the audio player directly to that second, enabling
  instant verification of action items against ground-truth audio.

### Transcript Ground-Truth Immutability Guarantee

A fundamental architectural constraint:
- Generating, regenerating, or editing meeting notes **must never** mutate or
  overwrite `item.transcripts` or `item.userNotes`.
- `AudioWorkspaceController.saveSummaryText(_:for:)` mutates only `item.summary`.
- `item.summary` stores `transcriptRevisionID` to keep summaries explicitly tied
  to the transcript revision they were generated from.

---

## 5. Phase 1: Custom Vocabulary & Jargon Dictionary

### Word-Boundary Safe Normalization

Domain-specific acronyms, technical terms, and proper nouns are managed by
`AudioCustomVocabulary`:
- Each entry defines a canonical `term` and optional `aliases`.
- Replacement expressions use regex word boundaries `\b(?i:alias)\b` to prevent
  corrupting substrings within larger words (for example, preventing `API` from
  replacing `rapid`).
- Normalization applies:
  1. To live speech segments in `AudioWorkspaceLive`.
  2. To final batch transcript segments in `AudioWorkspaceJobs`.
  3. When generating model prompt bias glossaries (`promptContext`).

### Persistence & Management Sheet

- Store path: `~/Library/Application Support/TurboSpark/custom_vocabulary.json`.
- Access UI: `AudioCustomVocabularySheet` accessible from:
  1. The Speech-to-Text header bar button.
  2. The transcript editor action bar (`Vocabulary...`).
- Allows adding terms, adding comma-separated phonetic or spelling aliases,
  searching terms, and deleting obsolete jargon.

---

## 6. Storage & State Ledger

All persistent data structures conform to `Codable`, `Sendable`, and `Equatable`.

| Structure | Storage Path / Location | Purpose |
| :--- | :--- | :--- |
| `AudioLibraryItem` | Encrypted SQLite / JSON | Library metadata, clips, transcripts, the user's notes, summary, speaker names |
| `AudioSpeakerVoiceprint` | `item.speakerVoiceprints` | 64-band log-mel voiceprint vectors on take |
| `AudioSpeakerProfileStore` | `speaker_profiles.json` | Cross-meeting learned speaker acoustic embeddings |
| `AudioCustomVocabularyStore`| `custom_vocabulary.json`| Custom jargon terms and aliases |

---

## 7. Verification Checklist

When modifying audio meeting or notes infrastructure, verify:
1. `swift test --filter AudioCaptureTests`: Validates ring buffer bounding and packet timing.
2. `swift test --filter AudioSpeakerMemoryTests`: Validates voiceprint generation, vector normalization, and cosine matching.
3. `swift test --filter AudioMeetingNotesTests`: Validates advisor model tiers, chunking, and transcript immutability.
   `swift test --filter AudioMeetingWorkspaceTests`: Validates user notes saving beside capture writes, the list filter, notes prompts, and Ask retrieval.
4. `swift test --filter AudioCustomVocabularyTests`: Validates regex word boundary safety and serialization round-trip.
5. `swift test --filter AudioSpeakerRosterTests`: Validates roster naming, merging, and cycle prevention.
