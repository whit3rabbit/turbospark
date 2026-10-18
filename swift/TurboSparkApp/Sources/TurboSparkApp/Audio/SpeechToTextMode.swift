import Foundation

/// The Speech-to-Text page's three jobs, picked with the pill switcher. Each
/// records with its own preset, so its takes land in its own collection and
/// list, and the Record page's own choice is never touched.
enum SpeechToTextMode: String, CaseIterable, Identifiable, Sendable {
    case notes, meeting, anyAudio
    var id: String { rawValue }
    var titleKey: String {
        switch self {
        // Not "Notes": that key is a remarks field elsewhere and translates so.
        case .notes: return "Voice Notes"
        case .meeting: return "Meeting"
        case .anyAudio: return "Any Audio"
        }
    }
    var symbol: String {
        switch self {
        case .notes: return "mic"
        case .meeting: return "person.2"
        case .anyAudio: return "macwindow"
        }
    }
    var recordingPreset: AudioRecordingPreset {
        switch self {
        case .notes: return .voiceNote
        case .meeting: return .meeting
        case .anyAudio: return .appAudio
        }
    }
}

/// Which page started the running capture. Only one capture runs at a time,
/// so every page shows that one take, and the top bar badge opens the page
/// that owns it.
enum AudioCaptureOrigin: Equatable, Sendable {
    case recordPage
    case speechToText(SpeechToTextMode)
    case voiceReference
}

/// What each mode lists. Pure so tests can drive it without a controller.
enum SpeechToTextLibrary {
    /// A take belongs to the mode whose preset recorded it. Takes saved
    /// before presets were stored match by their preset's collection. Any
    /// Audio also lists imported files, and transcripts whose source is gone.
    static func items(for mode: SpeechToTextMode, in items: [AudioLibraryItem], collections: [String: String]?) -> [AudioLibraryItem] {
        let preset = mode.recordingPreset
        let legacyCollection = collections?[preset.rawValue]
        let ids = Set(items.map(\.id))
        return items.filter { item in
            guard !item.isEmptyGeneratedAttempt else { return false }
            if let raw = item.recordingPreset { return raw == preset.rawValue }
            if item.kind == .recording, let legacyCollection, !legacyCollection.isEmpty {
                return item.collection == legacyCollection
            }
            guard mode == .anyAudio else { return false }
            if item.kind == .imported { return true }
            if item.kind == .transcription { return item.sourceID.map { !ids.contains($0) } ?? true }
            return false
        }.sorted { $0.createdAt > $1.createdAt }
    }

    /// The list row a selection highlights: a transcript made from a listed
    /// source selects that source's row.
    static func listSelection(_ selectedID: UUID?, in items: [AudioLibraryItem]) -> UUID? {
        guard let selectedID, let item = items.first(where: { $0.id == selectedID }) else { return selectedID }
        if item.kind == .transcription, let source = item.sourceID, items.contains(where: { $0.id == source }) { return source }
        return selectedID
    }

    /// The item whose transcript a source shows: the source itself when it
    /// was transcribed in place (every take), else its newest transcript.
    static func transcriptItem(for source: AudioLibraryItem, in items: [AudioLibraryItem]) -> AudioLibraryItem? {
        if source.preferredTranscript != nil { return source }
        return items.filter { $0.kind == .transcription && $0.sourceID == source.id }
            .max { $0.createdAt < $1.createdAt }
    }
}

extension AudioWorkspaceController {
    var speechToTextMode: SpeechToTextMode {
        preferences.speechToTextMode.flatMap(SpeechToTextMode.init(rawValue:)) ?? .notes
    }

    /// Switching only changes what the page shows while a capture runs. The
    /// running take keeps its sources and live text, which stop and the
    /// final pass still read.
    func selectSpeechToTextMode(_ mode: SpeechToTextMode) {
        guard mode != speechToTextMode else { return }
        stopPlayback()
        preferences.speechToTextMode = mode.rawValue
        if !hasRecordingActivity { applySpeechToTextSources() }
        flushPreferences()
    }

    /// The preset a new take files under. On Speech-to-Text it follows the
    /// mode, so a meeting recorded without the microphone is still a meeting.
    var activeRecordingPreset: AudioRecordingPreset? {
        page == .transcribe ? speechToTextMode.recordingPreset : effectiveRecordingPreset
    }

    func applySpeechToTextSources() {
        let preset = speechToTextMode.recordingPreset
        captureMicrophone = preset.capturesMicrophone
        captureSystemAudio = preset.capturesSystemAudio
        liveTranscription = preset.liveText
    }

    /// Puts the page's own sources back while idle: the mode's on
    /// Speech-to-Text, the saved preset's on the Record page.
    func restoreCaptureSources() {
        guard !hasRecordingActivity else { return }
        if page == .transcribe { applySpeechToTextSources() } else { restoreRecordingPreset() }
    }

    /// Run before Start. On Speech-to-Text the mode decides the sources, which
    /// also turns live text back on after a failing draft switched it off.
    /// The Record page keeps sources the user changed by hand.
    func prepareRecordingSources() {
        guard page == .transcribe, !hasRecordingActivity else { return }
        applySpeechToTextSources()
    }

    /// Where a capture starting now belongs.
    func originForNewCapture(asVoiceReference: Bool) -> AudioCaptureOrigin {
        if asVoiceReference { return .voiceReference }
        return page == .transcribe ? .speechToText(speechToTextMode) : .recordPage
    }

    var activeCaptureOrigin: AudioCaptureOrigin? { hasRecordingActivity ? captureOrigin : nil }

    /// The Speech-to-Text mode whose capture is running, if one is.
    var liveSpeechToTextMode: SpeechToTextMode? {
        if case .speechToText(let mode)? = activeCaptureOrigin { return mode }
        return nil
    }

    /// Opens the page that owns the running capture, for the top bar badge.
    func openActiveCapture() {
        if isRecordingVoiceReference { selectPage(.voiceClone); return }
        if let mode = liveSpeechToTextMode {
            selectPage(.transcribe)
            selectSpeechToTextMode(mode)
            return
        }
        selectPage(.record)
    }

    /// The take a mode's live view shows, while that mode's capture runs.
    func liveItem(for mode: SpeechToTextMode) -> AudioLibraryItem? {
        guard liveSpeechToTextMode == mode, let id = recordingID else { return nil }
        return items.first { $0.id == id }
    }
}
