import AppKit
import SwiftUI

@MainActor
enum ModelLocationPicker {
    static func choose(for model: AppModel) {
        let panel = NSOpenPanel()
        // NSOpenPanel takes plain Strings, so these go through AppStrings to
        // follow the in-app language picker rather than the process locale.
        panel.title = AppStrings.localized("Choose Model Folder")
        panel.message = AppStrings.localized("Choose the model directory containing manifest.json.")
        panel.prompt = AppStrings.localized("Choose Model")
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.resolvesAliases = true

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        model.setModelURL(selectedURL)
    }
}
