# Stage 6: Comprehension report (OC-08 local voice call)

Target: docs/designs/opencode2-parity/OC-08-local-voice-call.md (full file). Reviewer: egm-comprehension (completed). Read the document alone.

## What this proposes

Add a "Call" mode to the app. The user presses Call next to the composer, speaks, an on-device streaming speech-to-text model transcribes, voice activity detection (VAD) plus a silence window decides the utterance ended, and the transcript is sent as the prompt through the normal send path. The agent's reply is then spoken aloud (system voice by default, local TTS optional). The user can interrupt (barge-in), mute, or End. Everything is local, with a floating panel showing state glyph plus text and captions.

## How the existing system works, per this document

The app already has mic capture, live streaming STT, VAD, local TTS jobs, and a system-voice read-aloud class, all aimed at recordings and read-aloud rather than conversation. Nothing currently connects a live transcript to the chat agent or speaks its reply. A "send path"/queue handles typed prompts (steering, permissions, usage accounting) and a "recording access gate" handles mic permission. An `AppModel` holds the selected chat. The document does not say how any of these work internally.

## Findings (CO)

No Blockers: every section could be paraphrased, though CO-1, CO-3 and CO-4 leave the core turn and barge-in behavior under-specified.

CO-1 Major, lines 130-131 vs 75-85, 70-71, 162-164. "Draft mode" is in the Technical Plan as an option, tagged "Open Questions candidate", but absent from Open Questions, Goals, CUJs, persistence and tests. Whether it is in v1 and its default are unknown. "Draft" also means two things (line 70-71 approvals vs transcript review).

CO-2 Major, lines 50-52, 135-136, 163. The silence window has no default, no owner and no settings UI, though it is persisted as a preference. The latency NFR "under 600 ms plus the silence window" and "existing synthesizer start cost" cannot be checked.

CO-3 Major, lines 53-54, 84-85, 100-101, 183. Barge-in is not self-consistent. "Lead window" and "lead audio" are undefined and used differently in CUJ 2 and the open question. Nothing explains how the mic avoids hearing the app's own speaker output; only "use headphones" in manual testing.

CO-4 Major, lines 100-102 vs 114-125. State machine inconsistent: goals list "paused", the diagram omits it and omits End, error, turn failure, cancelled and empty reply. Speaking during "thinking" (queued, steered, ignored) is undefined; "steering behaves identically" is not explained.

CO-5 Major, lines 61-62, 104-110. CUJ 3 "no state change" is ambiguous (which state, what the user sees). "Existing access sheet", "recording access gate" and "Allow flow" are neither described nor linked. A macOS denial in System Settings does not re-prompt; no path to Settings.

CO-6 Major, lines 19-21, 57-59, 123, 182. STT and TTS model prerequisites unstated: no path if no STT model is installed or loaded; "installed local TTS voice" undefined; the call settings UI where the voice is chosen is not in the file list.

CO-7 Major, lines 29-41, 127-129, 162, 173, 66-67, 184. Undefined project names: `AppModel`, "send path", "queue path", "steering", "usage accounting", "archive", "DESIGN.md section 9 checklist", "offline policy", "RecordingPermissionsTests style". A newcomer cannot judge whether "the same queue path" is workable.

CO-8 Nit, lines 39-40. "OpenCode ranks this gap eighth" has no cited source here; "OpenCode" is not defined in one line.

CO-9 Nit, lines 140-141, 180. "21 languages" and "localization gates" unlisted and unlinked.

CO-10 Nit, lines 3-4, 9. "Missing in Swift" not explained (native macOS app?). Relative links `swift/TurboSparkApp/...` may not resolve from the file's folder.

CO-11 Nit, lines 77-79, 154. Panel shape is listed as open but Goals and file list already assume a floating draggable panel; keyboard operability of a draggable panel is unspecified.

CO-12 Nit, line 171. The mutation-check instruction has no stated way or time it runs (manual or CI).
