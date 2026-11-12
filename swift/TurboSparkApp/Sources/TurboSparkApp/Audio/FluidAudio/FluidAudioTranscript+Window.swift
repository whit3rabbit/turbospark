import Foundation
import TurboSpark

extension FluidAudioTranscript {
    /// Marks that end a sentence in the languages these models write: the
    /// ideographic full stop and full-width exclamation and question marks,
    /// then their ASCII forms. One segment per sentence keeps token timing
    /// usable in the editor and in subtitles.
    static let sentenceEnds: Set<Character> = ["\u{3002}", "\u{FF01}", "\u{FF1F}", "!", "?"]

    /// This window's result in the form the workspace saves.
    ///
    /// Token times are relative to the window. Only tokens that start in the
    /// part of the recording the window owns are kept, so overlapping windows
    /// do not repeat words, which matters for Japanese and Chinese: the
    /// workspace's word reconcile splits on spaces and never matches them.
    /// Without token times the text spans the whole window and is marked as a
    /// window extent, not alignment.
    func transcript(in window: AudioTranscriptionWindow, language: String?) -> AudioTranscript {
        guard !tokens.isEmpty else {
            let text = self.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let segments = text.isEmpty ? [] : [AudioSegment(startSeconds: 0, endSeconds: window.duration, text: text, timing: "window")]
            return AudioTranscript(text: text, language: language, segments: segments)
        }
        let owned = tokens.filter {
            let start = window.start + $0.start
            return start >= window.ownedFrom && start < window.ownedUntil
        }
        var segments: [AudioSegment] = []
        var sentence: [Token] = []
        func close() {
            let text = sentence.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            if let first = sentence.first, let last = sentence.last, !text.isEmpty {
                segments.append(AudioSegment(startSeconds: first.start, endSeconds: last.end, text: text, timing: "token"))
            }
            sentence.removeAll()
        }
        for token in owned {
            sentence.append(token)
            if let mark = token.text.trimmingCharacters(in: .whitespaces).last, Self.sentenceEnds.contains(mark) { close() }
        }
        close()
        let text = owned.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
        return AudioTranscript(text: text, language: language, segments: segments)
    }
}
