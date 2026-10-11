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

    /// The card's visible title, from the catalog: `title` is a key
    /// ("Light theme", "Dark theme").
    private var titleText: Text {
        Text(LocalizedStringKey(title), bundle: .module)
    }

    /// "Light theme, Accent": the mode first, so the light and dark copies of
    /// each control stay distinguishable in VoiceOver's control list.
    private func spokenName(_ control: Text) -> Text {
        titleText + Text(verbatim: ", ") + control
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            AppAdaptiveHeader {
                titleText
                    .font(theme.ui(.large, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
            } actions: {
                headerActions
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
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
        )
    }

    /// Clipboard import and copy, then the preset menu. Quiet small actions:
    /// they sit inside a card next to a picker, and none of them is the
    /// pane's main action.
    @ViewBuilder
    private var headerActions: some View {
        Button {
            importThemePreset()
        } label: {
            importFailed ? Text("Not a theme", bundle: .module) : Text("Import", bundle: .module)
        }
        .appAction(.quiet, size: .small)
        .appPointerCursor()

        Button {
            copyThemeJSON()
        } label: {
            copiedToast ? Text("Copied!", bundle: .module) : Text("Copy theme", bundle: .module)
        }
        .appAction(.quiet, size: .small)
        .appPointerCursor()

        // Preset names are palette names, shown as written.
        Menu {
            ForEach(ThemePreset.presets) { preset in
                Button {
                    manager.applyPreset(preset, forMode: isDark)
                } label: {
                    Text(verbatim: preset.name)
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
                    .accessibilityHidden(true)

                Text(verbatim: config.preset)
                    .font(theme.ui(.small))
                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.tiny))
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(.appBorder.opacity(0.4), lineWidth: 1)
            )
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .appPointerCursor()
        .help(Text("Preset", bundle: .module))
        .accessibilityLabel(spokenName(Text("Preset", bundle: .module)))
        .accessibilityValue(Text(verbatim: config.preset))
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

            Picker(selection: $config.accentHex) {
                ForEach(accentOptions) { option in
                    HStack {
                        Circle()
                            .fill(Color(hex: option.hex) ?? .primary)
                            .frame(width: 8, height: 8)
                        // Curated names are catalog keys; a carried preset
                        // or custom name has no key and shows as written.
                        Text(LocalizedStringKey(option.name), bundle: .module)
                            .font(theme.ui(.base))
                    }
                    .tag(option.hex)
                }
            } label: {
                spokenName(Text("Accent", bundle: .module))
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(minWidth: 120, idealWidth: 140, maxWidth: 200)
            .onChange(of: config.accentHex) { _, newHex in
                config.accentName = AccentOption.name(forHex: newHex, isDark: isDark)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    /// `label` is a catalog key ("Background", "Foreground").
    private func colorRow(label: String, hexString: Binding<String>) -> some View {
        let name = Text(LocalizedStringKey(label), bundle: .module)
        return HStack {
            name
                .font(theme.ui(.base))
            Spacer()

            HStack(spacing: 8) {
                ColorPicker(selection: Binding(
                    get: { Color(hex: hexString.wrappedValue) ?? (isDark ? .black : .white) },
                    set: { hexString.wrappedValue = $0.toHex() }
                ), supportsOpacity: false) {
                    spokenName(name)
                }
                .labelsHidden()

                TextField(text: hexString, prompt: Text(verbatim: "#RRGGBB")) {
                    spokenName(name) + Text(verbatim: ", ") + Text("Hex code", bundle: .module)
                }
                .labelsHidden()
                .font(theme.code(.small))
                .frame(minWidth: 84, idealWidth: 96, maxWidth: 120)
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

            Toggle(isOn: $config.translucentSidebar) {
                spokenName(Text("Translucent sidebar", bundle: .module))
            }
            .toggleStyle(.switch)
            .labelsHidden()
            .appPointerCursor()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    /// Locale-aware percent ("45 %" in French), shared by the readout and
    /// the slider's VoiceOver value.
    private var contrastPercent: String {
        (config.contrast / 100).formatted(.percent.precision(.fractionLength(0)))
    }

    private var contrastRow: some View {
        HStack(spacing: 16) {
            Text("Contrast", bundle: .module)
                    .settingsControl("Contrast", pane: .appearance, timing: .immediate)
                .font(theme.ui(.base))
            Spacer()

            Slider(value: $config.contrast, in: 0...100, step: 1) {
                spokenName(Text("Contrast", bundle: .module))
            }
            .labelsHidden()
            .frame(minWidth: 120, idealWidth: 160, maxWidth: 220)
            .accessibilityValue(Text(verbatim: contrastPercent))

            Text(verbatim: contrastPercent)
                .font(theme.ui(.small).monospacedDigit())
                .foregroundStyle(.appSecondary)
                .fixedSize()
                .frame(minWidth: 36, alignment: .trailing)
                .accessibilityHidden(true)
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
