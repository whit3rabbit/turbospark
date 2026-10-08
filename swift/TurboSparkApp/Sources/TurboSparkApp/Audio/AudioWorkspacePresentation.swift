import Foundation

extension AudioWorkspaceController {
    var runTitle: String {
        switch task {
        case .speechToText: return "Transcribe"
        case .textToSpeech: return "Generate speech"
        case .music: return "Generate music"
        default: return "Run"
        }
    }

    var workspaceResult: AudioLibraryItem? {
        guard let item = selectedItem else { return nil }
        switch page {
        case .music: return item.kind == .music ? item : nil
        case .voiceover: return item.kind == .voiceover ? item : nil
        default: return item
        }
    }

    var workspaceHistory: [AudioLibraryItem] {
        guard AudioWorkspacePage.destinations.contains(page) else { return [] }
        return items.filter { $0.kind == kind && $0.id != selectedID }
            .sorted { $0.createdAt > $1.createdAt }
    }
}
