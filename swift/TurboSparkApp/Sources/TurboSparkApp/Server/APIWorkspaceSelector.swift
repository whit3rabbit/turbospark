import SwiftUI

enum APIWorkspaceTab: Int, CaseIterable, Identifiable {
    // Embeddings sits before TypeSafe: it shares the TurboSpark listener
    // with the first three, and TypeSafe is a separate daemon.
    case text, image, audio, embeddings, typeSafe

    var id: Int { rawValue }
    /// The English catalog key. Rendered through `titleText`, which follows
    /// the in-app language picker; `String(localized:)` followed the system
    /// language instead.
    var title: String {
        switch self {
        case .text: return "Text"
        case .image: return "Image"
        case .audio: return "Audio"
        case .embeddings: return "Embeddings"
        case .typeSafe: return "TypeSafe"
        }
    }

    var titleText: Text { Text(LocalizedStringKey(title), bundle: .module) }

    func moved(_ offset: Int) -> Self {
        Self(rawValue: min(max(rawValue + offset, 0), Self.allCases.count - 1)) ?? self
    }
}

// The row of workspace buttons that used to sit across the top of the API
// pane moved to the panel beside the rail (`APISidebarPanel`), so the pane
// and the panel do not both offer the same choice.
