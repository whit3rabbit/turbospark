import Foundation

/// One recognizer call over a saved recording, in recording seconds: where
/// it starts, how much it reads, and which part of it owns the words when
/// windows overlap.
struct AudioTranscriptionWindow: Equatable, Sendable {
    var start: Double
    var duration: Double
    /// Words that start in `ownedFrom..<ownedUntil` belong to this window.
    /// The cut sits mid-overlap, where both windows heard context on each
    /// side, so a word is kept once.
    var ownedFrom: Double = 0
    var ownedUntil: Double = .infinity
    var end: Double { start + duration }
}

enum AudioTranscriptionPlan {
    /// Windows of at most `length` seconds, each starting `length - overlap`
    /// after the last, until the recording ends.
    ///
    /// With `skipsCoveredTail`, a last window whose audio the previous one
    /// already read is not run again. The Rust session loop keeps it, and its
    /// word reconcile removes the repeat, so it passes false and its windows
    /// stay exactly what they were.
    static func windows(duration: Double, length: Double, overlap: Double,
                        skipsCoveredTail: Bool = false) -> [AudioTranscriptionWindow] {
        guard duration > 0, length > 0, overlap >= 0, overlap < length else { return [] }
        var windows: [AudioTranscriptionWindow] = []
        for start in stride(from: 0.0, to: duration, by: length - overlap) {
            let window = AudioTranscriptionWindow(start: start, duration: min(length, duration - start))
            if skipsCoveredTail, let previous = windows.last, window.end <= previous.end { break }
            windows.append(window)
        }
        for index in windows.indices.dropFirst() {
            let cut = (windows[index].start + windows[index - 1].end) / 2
            windows[index - 1].ownedUntil = cut
            windows[index].ownedFrom = cut
        }
        return windows
    }
}
