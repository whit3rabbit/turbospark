import AppKit
import SwiftUI

@MainActor
enum ModelLocationPicker {
    static func choose(for model: AppModel) {
        let panel = NSOpenPanel()
        panel.title = "Choose Model Folder"
        panel.message = "Choose the model directory containing manifest.json."
        panel.prompt = "Choose Model"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.resolvesAliases = true

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        model.setModelURL(selectedURL)
    }
}
