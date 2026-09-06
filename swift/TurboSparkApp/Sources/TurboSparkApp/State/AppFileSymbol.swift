import Foundation

/// The one extension-to-SF-Symbol map in this app.
///
/// It was a `switch` private to `AppPromptAttachment.symbolName` until the
/// artifact card needed the same answer. A second copy is the failure
/// `swift/CLAUDE.md` Gotcha 22 records twice over: a display value restated
/// per branch is correct on the day it is written, and the two copies then
/// drift silently -- the artifact card would draw a generic `doc` for a
/// `.swift` file that the attachment chip draws a code glyph for, with
/// nothing anywhere going red.
public enum AppFileSymbol {
    /// SF Symbol for a file extension, lowercased and without the dot.
    ///
    /// `isImage` is passed in rather than recomputed here, because the image
    /// extension list belongs to `AppPromptAttachment` and having two readers
    /// of it decide "is this a picture" differently is the same hazard one
    /// level down.
    public static func name(forExtension ext: String, isImage: Bool) -> String {
        if isImage { return "photo" }
        switch ext {
        case "pdf": return "doc.richtext"
        case "docx", "doc": return "doc.text"
        case "xlsx", "xls", "csv": return "tablecells"
        case "pptx", "ppt": return "rectangle.on.rectangle"
        case "json", "yaml", "yml", "toml": return "curlybraces"
        case "swift", "rs", "py", "c", "cpp", "h", "js", "ts", "html", "css":
            return "chevron.left.forwardslash.chevron.right"
        case "md", "markdown", "mdx": return "doc.richtext"
        case "txt": return "doc.plaintext"
        default: return "doc"
        }
    }
}
