import AppKit
import SwiftUI

/// Card view for customizing a specific theme mode configuration (Light or Dark).
public struct ThemeConfigCardView: View {
    public let title: String
    public let isDark: Bool
    @Binding public var config: ThemeModeConfig
    @ObservedObject public var manager: AppearanceManager

    @State private var copiedToast = false

    /// Families narrowed to what will actually render on this machine.
    ///
    /// These were two array literals, which is how Inter, JetBrains Mono and
    /// Fira Code came to be on the menu while no font file for any of them
    /// existed in the tree: `Font.custom` falls back to the system face
    /// without erroring, so picking one did nothing and read as the setting
    /// being ignored. `swift/CLAUDE.md` Gotcha 22 is the same rule on the
    /// model hub -- build the options from what is present.
    private var installedFamilies: Set<String> { AppFontCatalog.installedFamilies() }

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
                    .font(.headline.weight(.semibold))

                Spacer()

                HStack(spacing: 8) {
                    Button("Import") {
                        importThemePreset()
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .appPointerCursor()

                    Button(copiedToast ? "Copied!" : "Copy theme") {
                        copyThemeJSON()
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
                            Text("Aa")
                                .font(.caption.weight(.bold))
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 4))

                            Text(config.preset)
                                .font(.caption)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption2)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
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

                fontPickerRow(
                    label: "UI font",
                    family: $config.uiFontFamily,
                    weight: $config.uiFontWeight,
                    familyOptions: AppFontCatalog.availableUIFamilies(installed: installedFamilies)
                )
                Divider().padding(.leading, 16)

                fontPickerRow(
                    label: "Code font",
                    family: $config.codeFontFamily,
                    weight: $config.codeFontWeight,
                    familyOptions: AppFontCatalog.availableCodeFamilies(installed: installedFamilies)
                )
                Divider().padding(.leading, 16)

                translucentSidebarRow
                Divider().padding(.leading, 16)

                contrastRow
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
        )
    }

    private var accentRow: some View {
        HStack {
            Text("Accent")
                .font(.subheadline)
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
                .font(.subheadline)
            Spacer()

            HStack(spacing: 8) {
                ColorPicker("", selection: Binding(
                    get: { Color(hex: hexString.wrappedValue) ?? (isDark ? .black : .white) },
                    set: { hexString.wrappedValue = $0.toHex() }
                ), supportsOpacity: false)
                .labelsHidden()

                TextField("", text: hexString)
                    .font(.caption.monospaced())
                    .frame(width: 80)
                    .textFieldStyle(.roundedBorder)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private func fontPickerRow(
        label: String,
        family: Binding<String>,
        weight: Binding<String>,
        familyOptions: [String]
    ) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
            Spacer()

            HStack(spacing: 8) {
                Picker("", selection: family) {
                    ForEach(familyOptions, id: \.self) { fam in
                        Text(fam).tag(fam)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 130)
                .labelsHidden()

                Picker("", selection: weight) {
                    Text("Regular").tag("Regular")
                    Text("Medium").tag("Medium")
                    Text("Semibold").tag("Semibold")
                    Text("Bold").tag("Bold")
                }
                .pickerStyle(.menu)
                .frame(width: 100)
                .labelsHidden()
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var translucentSidebarRow: some View {
        HStack {
            Text("Translucent sidebar")
                .font(.subheadline)
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
            Text("Contrast")
                .font(.subheadline)
            Spacer()

            Slider(value: $config.contrast, in: 0...100, step: 1)
                .frame(width: 160)

            Text("\(Int(config.contrast))")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 28, alignment: .trailing)
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

    private func importThemePreset() {
        if let preset = ThemePreset.presets.first(where: { $0.id == "turbospark" }) {
            manager.applyPreset(preset, forMode: isDark)
        }
    }
}
