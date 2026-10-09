import Foundation

/// Lyrics-side reference for MiniMax Music: the section tags the model reads
/// and helpers that add them to the lyrics field.
///
/// The tag names are literal tokens the model reads, so they are not
/// localized. The writing tips live in the view, where they are.
enum MusicLyricsGuide {
    /// Section tags from the model's prompting guide.
    static let sectionTags = [
        "intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "instrumental", "solo", "outro"
    ]

    /// Common layouts, as tag sequences. Shown by their sequence, so no
    /// layout needs a name or a translation.
    static let layouts: [[String]] = [
        ["verse", "chorus", "verse", "chorus", "outro"],
        ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "outro"],
        ["intro", "verse", "chorus", "verse", "chorus", "solo", "chorus", "outro"],
        ["intro", "instrumental", "solo", "instrumental", "outro"]
    ]

    static func line(for tag: String) -> String { "[\(tag)]" }

    /// Append `tags` to `lyrics`, each on its own line. The model drops any
    /// text that shares a line with a leading tag, so a tag never shares one,
    /// and sections are separated by a blank line to leave room to write.
    ///
    /// The bare instrumental token means "no vocals" for the whole song, so
    /// adding sections to it replaces it rather than leaving a contradiction.
    static func appending(_ tags: [String], to lyrics: String) -> String {
        guard !tags.isEmpty else { return lyrics }
        let existing = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let block = tags.map(line(for:)).joined(separator: "\n\n") + "\n"
        if existing.isEmpty || existing == AudioWorkspaceController.instrumentalLyrics { return block }
        return existing + "\n\n" + block
    }
}
