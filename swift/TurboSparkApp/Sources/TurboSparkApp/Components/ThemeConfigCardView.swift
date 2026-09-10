import AppKit
import SwiftUI

/// Card view for customizing a specific theme mode configuration (Light or Dark).
public struct ThemeConfigCardView: View {
    @Environment(\.appTheme) private var theme
    public let title: String
    public let isDark: Bool
    @Binding public var config: ThemeModeConfig
    @ObservedObject public var manager: AppearanceManager

    @State private var copiedToast = false
    @State private var importFailed = false

    public init(
        title: String,
        isDark: Bool,
        config: Binding<ThemeModeConfig>,
        manager: AppearanceManager
    ) {
        self.title = title
        self.isDark = isDark
        self._config = config
        self.manager = manager
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text(title)
                    .font(theme.ui(.large, weight: .semibold))

                Spacer()

                HStack(spacing: 8) {
                    Button(importFailed ? "Not a theme" : "Import") {
                        importThemePreset()
                    }
                    .buttonStyle(.plain)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .appPointerCursor()

                    Button(copiedToast ? "Copied!" : "Copy theme") {
                        copyThemeJSON()
                    }
                    .buttonStyle(.plain)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .appPointerCursor()

                    // Preset menu
                    Menu {
                        ForEach(ThemePreset.presets) { preset in
                            Button(preset.name) {
                                manager.applyPreset(preset, forMode: isDark)
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text(verbatim: "Aa")
                                .font(theme.ui(.small, weight: .bold))
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 4))

                            Text(config.preset)
                                .font(theme.ui(.small))
                            Image(systemName: "chevron.up.chevron.down")
                                .font(theme.ui(.tiny))
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.appSurface)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .appPointerCursor()
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider()

            // Settings Rows
            VStack(spacing: 0) {
                accentRow
                Divider().padding(.leading, 16)

                colorRow(label: "Background", hexString: $config.backgroundHex)
                Divider().padding(.leading, 16)

                colorRow(label: "Foreground", hexString: $config.foregroundHex)
                Divider().padding(.leading, 16)

                // No font rows here. Fonts are GLOBAL: `AppearanceManager
                // .setUIFont` / `setCodeFont` write both configs, so a
                // per-mode font was never representable, and a row on each
                // card promised an independence the storage refused. The
                // Preferences card below carries the one set of font controls.
                translucentSidebarRow
                Divider().padding(.leading, 16)

                contrastRow
            }
        }
        .background(.appSurface.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
        )
    }

    private var accentRow: some View {
        HStack {
            Text("Accent", bundle: .module)
                    .settingsControl("Accent", pane: .appearance, timing: .immediate)
                .font(theme.ui(.base))
            Spacer()

            // Always carries the selected value, so the menu cannot render
            // blank against a preset or a custom color. See `AccentOption`.
            let accentOptions = AccentOption.options(
                isDark: isDark,
                selectedHex: config.accentHex,
                selectedName: config.accentName)

            Picker("", selection: $config.accentHex) {
                ForEach(accentOptions) { option in
                    HStack {
                        Circle()
                            .fill(Color(hex: option.hex) ?? .primary)
                            .frame(width: 8, height: 8)
                        Text(option.name)
                            .font(theme.ui(.base))
                    }
                    .tag(option.hex)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 140)
            .labelsHidden()
            .onChange(of: config.accentHex) { _, newHex in
                config.accentName = AccentOption.name(forHex: newHex, isDark: isDark)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private func colorRow(label: String, hexString: Binding<String>) -> some View {
        HStack {
            Text(label)
                .font(theme.ui(.base))
            Spacer()

            HStack(spacing: 8) {
                ColorPicker("", selection: Binding(
                    get: { Color(hex: hexString.wrappedValue) ?? (isDark ? .black : .white) },
                    set: { hexString.wrappedValue = $0.toHex() }
                ), supportsOpacity: false)
                .labelsHidden()

                TextField("", text: hexString)
                    .font(theme.code(.small))
                    .frame(width: 80)
                    .textFieldStyle(.roundedBorder)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var translucentSidebarRow: some View {
        HStack {
            Text("Translucent sidebar", bundle: .module)
                    .settingsControl("Translucent sidebar", pane: .appearance, timing: .immediate)
                .font(theme.ui(.base))
            Spacer()

            Toggle("", isOn: $config.translucentSidebar)
                .toggleStyle(.switch)
                .labelsHidden()
                .appPointerCursor()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var contrastRow: some View {
        HStack(spacing: 16) {
            Text("Contrast", bundle: .module)
                    .settingsControl("Contrast", pane: .appearance, timing: .immediate)
                .font(theme.ui(.base))
            Spacer()

            Slider(value: $config.contrast, in: 0...100, step: 1)
                .frame(width: 160)

            Text("\(Int(config.contrast))%", bundle: .module)
                .font(theme.ui(.small).monospacedDigit())
                .foregroundStyle(.appSecondary)
                .frame(width: 36, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private func copyThemeJSON() {
        if let data = try? JSONEncoder().encode(config),
           let json = String(data: data, encoding: .utf8) {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(json, forType: .string)
            copiedToast = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                copiedToast = false
            }
        }
    }

    /// Applies a `ThemeModeConfig` from the clipboard, or says why not.
    ///
    /// This used to fall through to applying the TurboSpark PRESET whenever
    /// the clipboard did not decode, so an Import with anything else copied
    /// silently replaced the user's theme and reported nothing. A miss now
    /// changes no state and names itself on the button for a moment.
    ///
    /// The imported config's font fields are discarded: fonts are global
    /// (see the note above `translucentSidebarRow`), and writing one mode's
    /// fonts here would desynchronise the two configs behind the Preferences
    /// card's back.
    private func importThemePreset() {
        guard let string = NSPasteboard.general.string(forType: .string),
              var imported = ThemeModeConfig.fromClipboardJSON(string) else {
            importFailed = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                importFailed = false
            }
            return
        }
        imported.uiFontFamily = config.uiFontFamily
        imported.uiFontWeight = config.uiFontWeight
        imported.codeFontFamily = config.codeFontFamily
        imported.codeFontWeight = config.codeFontWeight
        config = imported
    }
}
