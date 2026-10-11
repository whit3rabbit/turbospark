import AppKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct ThemeGalleryView: View {
    @ObservedObject var manager: AppearanceManager
    @Environment(\.appTheme) private var theme
    @State private var name = ""
    @State private var renamingID: String?
    @State private var showsNameEditor = false
    @State private var errorMessage: String?
    @State private var importPresented = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Theme gallery", bundle: .module)
                    .settingsControl("Theme gallery", pane: .appearance, timing: .immediate).themedFont(.large, weight: .semibold)
                Spacer()
                // No saved or preset match means the colors were edited by
                // hand; that state's name is a UI string, a theme's is not.
                Group {
                    if manager.currentThemeID == nil {
                        Text("Custom", bundle: .module)
                    } else {
                        Text(verbatim: manager.currentThemeName)
                    }
                }
                .themedFont(.small)
            }
            Text("Choose a palette for both Light and Dark. Your fonts and text sizes stay the same.", bundle: .module)
                .themedFont(.small)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170))], spacing: 12) {
                ForEach(manager.availableThemes) { preset in
                    Button { manager.applyPreset(preset) } label: {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 0) {
                                thumbnail(preset.light)
                                thumbnail(preset.dark)
                            }
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            HStack {
                                Text(verbatim: preset.name).themedFont(.small, weight: .medium)
                                Spacer()
                                if manager.currentThemeID == preset.id {
                                    Image(systemName: "checkmark.circle.fill").foregroundStyle(theme.accent)
                                        .accessibilityHidden(true)
                                }
                            }
                        }
                        .padding(10)
                        .background(theme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                            .stroke(manager.currentThemeID == preset.id ? theme.accent : theme.border, lineWidth: 2))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(Text(verbatim: preset.name))
                    .accessibilityAddTraits(manager.currentThemeID == preset.id ? [.isSelected] : [])
                }
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: AppMetrics.actionSpacing) { libraryActions }
                VStack(alignment: .leading, spacing: AppMetrics.actionSpacing) { libraryActions }
            }
            if manager.lightConfig.textContrastRatio < 4.5 || manager.darkConfig.textContrastRatio < 4.5 {
                Label { Text("Text contrast is low in one or both modes. Adjust foreground or background colors.", bundle: .module) }
                    icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small)
            }
            if let errorMessage {
                Text(errorMessage).foregroundStyle(.red).themedFont(.small)
            }
        }
        .alert(Text("Theme name", bundle: .module), isPresented: $showsNameEditor) {
            TextField(text: $name, prompt: Text("Name", bundle: .module)) {
                Text("Name", bundle: .module)
            }
            Button { saveName() } label: { Text("Save", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        }
        .fileImporter(isPresented: $importPresented, allowedContentTypes: [.json]) { result in
            do {
                let url = try result.get()
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                _ = try manager.importTheme(data: Data(contentsOf: url))
                errorMessage = nil
            } catch { errorMessage = error.localizedDescription }
        }
    }

    @ViewBuilder private var libraryActions: some View {
        Button {
            renamingID = nil
            // A suggested name the person edits; written in the UI language.
            name = AppStrings.localized("\(manager.currentThemeName) Copy")
            showsNameEditor = true
        } label: { Text("Save as new theme", bundle: .module)
                    .settingsControl("Save as new theme", pane: .appearance, timing: .immediate) }
        .appAction(.secondary)
        if let id = manager.currentThemeID, manager.savedThemes.contains(where: { $0.id == id }) {
            Button {
                renamingID = id
                name = manager.currentThemeName
                showsNameEditor = true
            } label: { Text("Rename", bundle: .module)
                    .settingsControl("Rename", pane: .appearance, timing: .immediate) }
            .appAction(.secondary)
            Button(role: .destructive) { manager.deleteTheme(id: id) } label: {
                Text("Delete saved theme", bundle: .module)
                    .settingsControl("Delete saved theme", pane: .appearance, timing: .immediate)
            }
            .appAction(.destructive)
            .help(Text("Keeps the current colors as Custom.", bundle: .module))
        }
        Menu {
            Button { importPresented = true } label: { Text("Import JSON file", bundle: .module)
                    .settingsControl("Import JSON file", pane: .appearance, timing: .immediate) }
            Button {
                do {
                    guard let text = NSPasteboard.general.string(forType: .string) else { throw ThemeLibraryError.invalidTheme }
                    _ = try manager.importTheme(data: Data(text.utf8))
                    errorMessage = nil
                } catch { errorMessage = error.localizedDescription }
            } label: { Text("Import from clipboard", bundle: .module)
                    .settingsControl("Import from clipboard", pane: .appearance, timing: .immediate) }
            Button { exportFile() } label: { Text("Export JSON file", bundle: .module)
                    .settingsControl("Export JSON file", pane: .appearance, timing: .immediate) }
            Button {
                do {
                    let data = try manager.exportTheme()
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
                } catch { errorMessage = error.localizedDescription }
            } label: { Text("Copy theme JSON", bundle: .module)
                    .settingsControl("Copy theme JSON", pane: .appearance, timing: .immediate) }
        } label: { Text("Import / Export", bundle: .module)
                    .settingsControl("Import / Export", pane: .appearance, timing: .immediate) }
        .appMenuAction()
    }

    private func thumbnail(_ config: ThemeModeConfig) -> some View { ThemeModeThumbnail(config: config) }

    private func saveName() {
        do {
            if let id = renamingID { try manager.renameTheme(id: id, name: name) }
            else { _ = try manager.saveTheme(named: name) }
            errorMessage = nil
        } catch { errorMessage = error.localizedDescription }
    }

    private func exportFile() {
        do {
            let data = try manager.exportTheme()
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.json]
            panel.nameFieldStringValue = "theme.json"
            if panel.runModal() == .OK, let url = panel.url { try data.write(to: url, options: .atomic) }
        } catch { errorMessage = error.localizedDescription }
    }
}

struct ThemeModeThumbnail: View {
    let config: ThemeModeConfig
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            RoundedRectangle(cornerRadius: 2).fill(Color(hex: config.accentHex) ?? .blue).frame(width: 24, height: 5)
            RoundedRectangle(cornerRadius: 2).fill(Color(hex: config.foregroundHex) ?? .primary).frame(height: 4)
            RoundedRectangle(cornerRadius: 2).fill((Color(hex: config.foregroundHex) ?? .primary).opacity(0.65)).frame(height: 4)
        }
        .padding(14)
        .frame(maxWidth: .infinity)
        .background(Color(hex: config.backgroundHex) ?? .white)
        .accessibilityHidden(true)
    }
}
