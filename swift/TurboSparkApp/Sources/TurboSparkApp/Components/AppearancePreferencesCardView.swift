import AppKit
import SwiftUI

/// Preferences card view for pointer cursors, status bar, and font sizing.
public struct AppearancePreferencesCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject public var manager: AppearanceManager

    public init(manager: AppearanceManager) {
        self.manager = manager
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Preferences", bundle: .module)
                    .settingsControl("Preferences", pane: .appearance, timing: .immediate)
                .font(theme.ui(.large, weight: .semibold))
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

            // 2. Status bar benchmarks view mode
            preferenceRow(
                title: "Status bar benchmarks",
                description: "Display bottom toolbar metrics as numeric text or live sparkline graphs"
            ) {
                Picker("", selection: $manager.statusBarViewMode) {
                    ForEach(StatusBarViewMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 175)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 3. Dock icon. The manager's `didSet` re-renders the icon, so
            // this row is the whole feature: the preference had persisted and
            // applied for months with no control anywhere to change it
            // (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
            preferenceRow(
                title: "Dock icon",
                description: "Choose the icon TurboSpark shows in the Dock"
            ) {
                Picker("", selection: $manager.dockIcon) {
                    ForEach(AppDockIcon.allCases) { icon in
                        Text(icon.label).tag(icon)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 140)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 4. Reduce motion
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

            // 5. Text size
            preferenceRow(
                title: "Text size",
                description: "Scale interface, messages, and transcript text"
            ) {
                Picker("", selection: $manager.textSize) {
                    ForEach(AppTextSize.allCases) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 6. UI font
            preferenceRow(
                title: "UI font",
                description: "Select the typeface and weight for the TurboSpark interface"
            ) {
                fontPicker(
                    family: Binding(
                        get: { manager.uiFontFamily },
                        set: { manager.setUIFont(family: $0) }
                    ),
                    weight: Binding(
                        get: { manager.uiFontWeight },
                        set: { manager.setUIFont(weight: $0) }
                    ),
                    options: AppFontCatalog.availableUIFamilies(installed: AppFontCatalog.installedFamilies()),
                    isCodeFont: false
                )
            }

            Divider().padding(.leading, 16)

            // 7. UI font size
            preferenceRow(
                title: "UI font size",
                description: "Adjust the base size used for the TurboSpark UI"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.uiFontSize, in: 11...28, step: 1)
                        .labelsHidden()

                    Text(verbatim: "\(Int(manager.uiFontSize))")
                        .font(theme.ui(.callout, weight: .medium).monospacedDigit())
                        .frame(width: 28, alignment: .center)

                    Text(verbatim: "px")
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
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

            Divider().padding(.leading, 16)

            // 8. Code font
            preferenceRow(
                title: "Code font",
                description: "Select the typeface and weight for code blocks and diffs"
            ) {
                fontPicker(
                    family: Binding(
                        get: { manager.codeFontFamily },
                        set: { manager.setCodeFont(family: $0) }
                    ),
                    weight: Binding(
                        get: { manager.codeFontWeight },
                        set: { manager.setCodeFont(weight: $0) }
                    ),
                    options: AppFontCatalog.availableCodeFamilies(installed: AppFontCatalog.installedFamilies()),
                    isCodeFont: true
                )
            }

            Divider().padding(.leading, 16)

            // 9. Code font size
            preferenceRow(
                title: "Code font size",
                description: "Adjust the base size used for code across chats and diffs"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.codeFontSize, in: 10...24, step: 1)
                        .labelsHidden()

                    Text(verbatim: "\(Int(manager.codeFontSize))")
                        .font(theme.ui(.callout, weight: .medium).monospacedDigit())
                        .frame(width: 28, alignment: .center)

                    Text(verbatim: "px")
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
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

            Divider().padding(.leading, 16)

            // 10. Diff markers
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

        }
        .background(.appSurface.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
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
                    .settingsControl(title, pane: .appearance)
                    .font(theme.ui(.base, weight: .medium))
                    .foregroundStyle(.appText)

                Text(description)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 16)
            accessory()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func fontPicker(
        family: Binding<String>,
        weight: Binding<String>,
        options: [String],
        isCodeFont: Bool = false
    ) -> some View {
        HStack(spacing: 8) {
            Picker("", selection: family) {
                ForEach(options, id: \.self) { fam in
                    Text(fam)
                        .font(AppFontDescriptor(family: fam, weight: .regular, size: 13, isCode: isCodeFont).font)
                        .tag(fam)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 140)
            .labelsHidden()

            Picker("", selection: weight) {
                Text("Regular", bundle: .module)
                    .settingsControl("Regular", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .regular, size: 13, isCode: isCodeFont).font).tag("Regular")
                Text("Medium", bundle: .module)
                    .settingsControl("Medium", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .medium, size: 13, isCode: isCodeFont).font).tag("Medium")
                Text("Semibold", bundle: .module)
                    .settingsControl("Semibold", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .semibold, size: 13, isCode: isCodeFont).font).tag("Semibold")
                Text("Bold", bundle: .module)
                    .settingsControl("Bold", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .bold, size: 13, isCode: isCodeFont).font).tag("Bold")
            }
            .pickerStyle(.menu)
            .frame(width: 100)
            .labelsHidden()
        }
    }
}
