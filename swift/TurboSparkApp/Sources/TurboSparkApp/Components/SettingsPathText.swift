import SwiftUI

/// A filesystem path shown in a settings row: one line, truncated in the
/// middle so both the volume and the leaf stay visible, selectable so it
/// can be copied, and with the full path as its tooltip because the
/// truncation otherwise hides exactly the part a reader is looking for.
///
/// A path is data, not UI text, so it is never localized.
struct SettingsPathText: View {
    let path: String
    var size: AppFontStep = .small

    init(_ path: String, size: AppFontStep = .small) {
        self.path = path
        self.size = size
    }

    var body: some View {
        Text(verbatim: path)
            .themedCode(size)
            .textSelection(.enabled)
            .lineLimit(1)
            .truncationMode(.middle)
            .help(Text(verbatim: path))
    }
}
