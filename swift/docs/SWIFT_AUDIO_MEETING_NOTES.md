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

### Segmented Detail Content Switcher

In `AudioLibraryView`, selected takes offer a segmented control:
`[ Transcript | Notes & Summary ]`:
- **Transcript Tab**: Full verbatim conversation transcript, word-level timing,
  speaker renaming, and export.
- **Notes & Summary Tab**: Structured Markdown executive summary, action items,
  decisions, model selector, and interactive jump chips.

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
  overwrite `item.transcripts`.
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
| `AudioLibraryItem` | Encrypted SQLite / JSON | Library metadata, clips, transcripts, summary, speaker names |
| `AudioSpeakerVoiceprint` | `item.speakerVoiceprints` | 64-band log-mel voiceprint vectors on take |
| `AudioSpeakerProfileStore` | `speaker_profiles.json` | Cross-meeting learned speaker acoustic embeddings |
| `AudioCustomVocabularyStore`| `custom_vocabulary.json`| Custom jargon terms and aliases |

---

## 7. Verification Checklist

When modifying audio meeting or notes infrastructure, verify:
1. `swift test --filter AudioCaptureTests`: Validates ring buffer bounding and packet timing.
2. `swift test --filter AudioSpeakerMemoryTests`: Validates voiceprint generation, vector normalization, and cosine matching.
3. `swift test --filter AudioMeetingNotesTests`: Validates advisor model tiers, chunking, and transcript immutability.
4. `swift test --filter AudioCustomVocabularyTests`: Validates regex word boundary safety and serialization round-trip.
5. `swift test --filter AudioSpeakerRosterTests`: Validates roster naming, merging, and cycle prevention.
