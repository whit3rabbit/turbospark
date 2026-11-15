import AppKit

/// The pen colors, as fixed sRGB values. A board is paper: its ink is data
/// the user drew, so like the annotation ink (`AnnotationRenderer.ink`) it
/// does not follow the app theme. Names are catalog keys.
enum WhiteboardSwatch: String, CaseIterable, Identifiable {
    case black, red, orange, green, blue, purple

    var id: String { rawValue }

    var color: NSColor {
        switch self {
        case .black: return NSColor(srgbRed: 0.08, green: 0.08, blue: 0.09, alpha: 1)
        case .red: return NSColor(srgbRed: 0.84, green: 0.16, blue: 0.18, alpha: 1)
        case .orange: return NSColor(srgbRed: 0.93, green: 0.53, blue: 0.08, alpha: 1)
        case .green: return NSColor(srgbRed: 0.13, green: 0.58, blue: 0.27, alpha: 1)
        case .blue: return NSColor(srgbRed: 0.10, green: 0.40, blue: 0.86, alpha: 1)
        case .purple: return NSColor(srgbRed: 0.52, green: 0.27, blue: 0.78, alpha: 1)
        }
    }

    /// Catalog key for the VoiceOver name and tooltip.
    var name: String {
        switch self {
        case .black: return "Black"
        case .red: return "Red"
        case .orange: return "Orange"
        case .green: return "Green"
        case .blue: return "Blue"
        case .purple: return "Purple"
        }
    }
}

/// Paper. Fixed white so the export (opaque white) matches the screen.
enum WhiteboardPaper {
    static let color = NSColor.white
}
