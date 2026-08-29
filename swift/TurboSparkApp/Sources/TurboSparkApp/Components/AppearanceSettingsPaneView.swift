import AppKit
import SwiftUI

public struct AppearanceSettingsPaneView: View {
    @ObservedObject private var manager = AppearanceManager.shared
    @Environment(\.colorScheme) private var colorScheme

    public init() {}

    private var isCurrentlyDark: Bool {
        switch manager.appearance {
        case .system:
            return colorScheme == .dark
        case .light:
            return false
        case .dark:
            return true
        }
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                themeSelectionSection

                ThemeCodePreviewView(manager: manager, isDark: isCurrentlyDark)

                ThemeConfigCardView(
                    title: "Light theme",
                    isDark: false,
                    config: $manager.lightConfig,
                    manager: manager
                )

                ThemeConfigCardView(
                    title: "Dark theme",
                    isDark: true,
                    config: $manager.darkConfig,
                    manager: manager
                )

                AppearancePreferencesCardView(manager: manager)
            }
            .padding(20)
        }
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.6))
    }

    // MARK: - Theme Mode Selector
    private var themeSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Theme")
                .font(.headline)
                .foregroundStyle(.primary)

            HStack(spacing: 16) {
                themeCard(mode: .system, title: "System") {
                    // Split card preview
                    HStack(spacing: 0) {
                        lightThumbnailMockup
                        darkThumbnailMockup
                    }
                }

                themeCard(mode: .light, title: "Light") {
                    lightThumbnailMockup
                }

                themeCard(mode: .dark, title: "Dark") {
                    darkThumbnailMockup
                }
            }
        }
    }

    private func themeCard<Content: View>(
        mode: AppAppearance,
        title: String,
        @ViewBuilder preview: () -> Content
    ) -> some View {
        let isSelected = manager.appearance == mode
        return Button {
            manager.appearance = mode
        } label: {
            VStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color(nsColor: .controlBackgroundColor))

                    preview()
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(
                            isSelected ? manager.activeAccentColor(isDark: isCurrentlyDark) : Color(nsColor: .separatorColor).opacity(0.3),
                            lineWidth: isSelected ? 2.5 : 1
                        )
                }
                .frame(height: 80)
                .shadow(color: isSelected ? manager.activeAccentColor(isDark: isCurrentlyDark).opacity(0.2) : Color.black.opacity(0.04), radius: 4, y: 2)

                Text(title)
                    .font(.caption.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .primary : .secondary)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
        .appPointerCursor()
    }

    private var lightThumbnailMockup: some View {
        ZStack {
            Color(red: 0.94, green: 0.94, blue: 0.96)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Circle().fill(Color.gray.opacity(0.4)).frame(width: 5, height: 5)
                    RoundedRectangle(cornerRadius: 2).fill(Color.gray.opacity(0.3)).frame(width: 28, height: 4)
                }
                .padding(.top, 6)
                .padding(.leading, 8)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white)
                    .padding(.horizontal, 6)
                    .overlay(
                        VStack(alignment: .leading, spacing: 3) {
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.4)).frame(width: 40, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.2)).frame(width: 55, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.2)).frame(width: 32, height: 3)
                        }
                        .padding(6),
                        alignment: .topLeading
                    )
            }
        }
    }

    private var darkThumbnailMockup: some View {
        ZStack {
            Color(red: 0.12, green: 0.12, blue: 0.14)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Circle().fill(Color.gray.opacity(0.5)).frame(width: 5, height: 5)
                    RoundedRectangle(cornerRadius: 2).fill(Color.gray.opacity(0.4)).frame(width: 28, height: 4)
                }
                .padding(.top, 6)
                .padding(.leading, 8)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(red: 0.18, green: 0.18, blue: 0.20))
                    .padding(.horizontal, 6)
                    .overlay(
                        VStack(alignment: .leading, spacing: 3) {
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.4)).frame(width: 40, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.2)).frame(width: 55, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.2)).frame(width: 32, height: 3)
                        }
                        .padding(6),
                        alignment: .topLeading
                    )
            }
        }
    }
}

// MARK: - Theme Config Card (Light & Dark)
private struct ThemeConfigCardView: View {
    let title: String
    let isDark: Bool
    @Binding var config: ThemeModeConfig
    @ObservedObject var manager: AppearanceManager

    @State private var copiedToast = false

    var body: some View {
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
                    familyOptions: ["System default", "SF Pro", "Inter", "Helvetica Neue", "Avenir"]
                )
                Divider().padding(.leading, 16)

                fontPickerRow(
                    label: "Code font",
                    family: $config.codeFontFamily,
                    weight: $config.codeFontWeight,
                    familyOptions: ["System default", "SF Mono", "Menlo", "JetBrains Mono", "Fira Code", "Courier"]
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

            let accentOptions = [
                ("Emerald", isDark ? "#6ABA71" : "#237D32"),
                ("Blue", isDark ? "#38BDF8" : "#2563EB"),
                ("Purple", isDark ? "#C084FC" : "#7C3AED"),
                ("Orange", isDark ? "#FB923C" : "#EA580C"),
                ("Black", "#000000"),
                ("White", "#FFFFFF")
            ]

            Picker("", selection: $config.accentHex) {
                ForEach(accentOptions, id: \.1) { name, hex in
                    HStack {
                        Circle()
                            .fill(Color(hex: hex) ?? .primary)
                            .frame(width: 8, height: 8)
                        Text(name)
                    }
                    .tag(hex)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 140)
            .labelsHidden()
            .onChange(of: config.accentHex) { newHex in
                if let matched = accentOptions.first(where: { $0.1 == newHex }) {
                    config.accentName = matched.0
                } else {
                    config.accentName = "Custom"
                }
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

// MARK: - Preferences Card
private struct AppearancePreferencesCardView: View {
    @ObservedObject var manager: AppearanceManager

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Preferences")
                .font(.headline.weight(.semibold))
                .padding(.horizontal, 16)
                .padding(.vertical, 14)

            Divider()

            // 1. Use pointer cursors
            preferenceRow(
                title: "Use pointer cursors",
                description: "Change the cursor to a pointer when hovering over interactive elements"
            ) {
                Toggle("", isOn: $manager.usePointerCursors)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .appPointerCursor()
            }

            Divider().padding(.leading, 16)

            // 2. Dock icon
            preferenceRow(
                title: "Dock icon",
                description: "Choose the icon the app will use in the dock"
            ) {
                HStack(spacing: 8) {
                    dockIconOption(.emeraldSpark, systemSymbol: "sparkles", gradientColors: [Color(red: 0.1, green: 0.3, blue: 0.15), Color(red: 0.05, green: 0.15, blue: 0.08)])
                    dockIconOption(.codexDark, systemSymbol: "chevron.left.forwardslash.chevron.right", gradientColors: [Color(white: 0.25), Color(white: 0.12)])
                    dockIconOption(.terminalPro, systemSymbol: "terminal.fill", gradientColors: [Color(red: 0.1, green: 0.25, blue: 0.5), Color(red: 0.05, green: 0.1, blue: 0.25)])
                    dockIconOption(.minimalist, systemSymbol: "bolt.fill", gradientColors: [Color(white: 0.15), Color(white: 0.08)])
                }
            }

            Divider().padding(.leading, 16)

            // 3. Reduce motion
            preferenceRow(
                title: "Reduce motion",
                description: "Reduce animations or match your system"
            ) {
                Picker("", selection: $manager.reduceMotion) {
                    ForEach(ReduceMotionPreference.allCases) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 4. UI font size
            preferenceRow(
                title: "UI font size",
                description: "Adjust the base size used for the TurboSpark UI"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.uiFontSize, in: 11...20, step: 1)
                        .labelsHidden()

                    Text("\(Int(manager.uiFontSize))")
                        .font(.subheadline.monospacedDigit())
                        .frame(width: 24, alignment: .center)

                    Text("px")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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

            Divider().padding(.leading, 16)

            // 5. Code font size
            preferenceRow(
                title: "Code font size",
                description: "Adjust the base size used for code across chats and diffs"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.codeFontSize, in: 10...18, step: 1)
                        .labelsHidden()

                    Text("\(Int(manager.codeFontSize))")
                        .font(.subheadline.monospacedDigit())
                        .frame(width: 24, alignment: .center)

                    Text("px")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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

            Divider().padding(.leading, 16)

            // 6. Diff markers
            preferenceRow(
                title: "Diff markers",
                description: "Show changes using colors or +/- markers"
            ) {
                Picker("", selection: $manager.diffMarkers) {
                    ForEach(DiffMarkerPreference.allCases) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 7. Font smoothing
            preferenceRow(
                title: "Font smoothing",
                description: "Use native macOS font anti-aliasing"
            ) {
                Toggle("", isOn: $manager.fontSmoothing)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .appPointerCursor()
            }
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
        )
    }

    private func preferenceRow<Accessory: View>(
        title: String,
        description: String,
        @ViewBuilder accessory: () -> Accessory
    ) -> some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)

                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 16)
            accessory()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func dockIconOption(_ icon: AppDockIcon, systemSymbol: String, gradientColors: [Color]) -> some View {
        let isSelected = manager.dockIcon == icon
        return Button {
            manager.dockIcon = icon
        } label: {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(LinearGradient(colors: gradientColors, startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 32, height: 32)

                Image(systemName: systemSymbol)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)

                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(
                        isSelected ? Color.accentColor : Color.clear,
                        lineWidth: isSelected ? 2 : 0
                    )
            }
            .frame(width: 32, height: 32)
        }
        .buttonStyle(.plain)
        .help(icon.label)
        .appPointerCursor()
    }
}
