import AppKit
import SwiftUI

/// Preferences card view for pointer cursors, status bar, and font sizing.
@MainActor
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

            // Two halves as separate properties: ten generic rows in one
            // builder expression is the shape that strains the macOS 14 SDK
            // type-checker (see `EngineSettingsPaneView.body`).
            displayRows

            Divider().padding(.leading, 16)

            typographyRows
        }
        .background(.appSurface.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
        )
    }

    private var displayRows: some View {
        VStack(alignment: .leading, spacing: 0) {
            // 1. Use pointer cursors
            preferenceRow(
                title: "Use pointer cursors",
                description: "Change the cursor to a pointer when hovering over interactive elements"
            ) {
                Toggle(isOn: $manager.usePointerCursors) {
                    Text("Use pointer cursors", bundle: .module)
                }
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
                adaptivePicker(Text("Status bar benchmarks", bundle: .module), selection: $manager.statusBarViewMode) {
                    ForEach(StatusBarViewMode.allCases) { mode in
                        Label { Text(LocalizedStringKey(mode.label), bundle: .module) } icon: {
                            Image(systemName: mode.systemImage)
                        }
                        .tag(mode)
                    }
                }
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
                Picker(selection: $manager.dockIcon) {
                    ForEach(AppDockIcon.allCases) { icon in
                        Text(LocalizedStringKey(icon.label), bundle: .module).tag(icon)
                    }
                } label: {
                    Text("Dock icon", bundle: .module)
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(minWidth: 120, idealWidth: 140, maxWidth: 200)
            }

            Divider().padding(.leading, 16)

            // 4. Reduce motion
            preferenceRow(
                title: "Reduce motion",
                description: "Reduce animations or match your system"
            ) {
                adaptivePicker(Text("Reduce motion", bundle: .module), selection: $manager.reduceMotion) {
                    ForEach(ReduceMotionPreference.allCases) { opt in
                        Text(LocalizedStringKey(opt.label), bundle: .module).tag(opt)
                    }
                }
            }

            Divider().padding(.leading, 16)

            // 5. Text size: the shared option rows, so this row, General and
            // the View menu list the same sizes under the same names.
            preferenceRow(
                title: "Text size",
                description: "Scale interface, messages, and transcript text"
            ) {
                adaptivePicker(Text("Text size", bundle: .module), selection: $manager.textSize) {
                    TextSizeOptions()
                }
            }
        }
    }

    private var typographyRows: some View {
        VStack(alignment: .leading, spacing: 0) {
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
                sizeStepper(Text("UI font size", bundle: .module), value: $manager.uiFontSize, range: 11...28)
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
                sizeStepper(Text("Code font size", bundle: .module), value: $manager.codeFontSize, range: 10...24)
            }

            Divider().padding(.leading, 16)

            // 10. Diff markers
            preferenceRow(
                title: "Diff markers",
                description: "Show changes using colors or +/- markers"
            ) {
                adaptivePicker(Text("Diff markers", bundle: .module), selection: $manager.diffMarkers) {
                    ForEach(DiffMarkerPreference.allCases) { opt in
                        Group {
                            // The marker glyphs are not words.
                            if opt == .plusMinus {
                                Text(verbatim: opt.label)
                            } else {
                                Text(LocalizedStringKey(opt.label), bundle: .module)
                            }
                        }
                        .tag(opt)
                    }
                }
            }
        }
    }

    /// One preference: title and description leading, the control trailing.
    /// `title` and `description` are catalog keys; the title doubles as the
    /// settings-search anchor, which `scripts/generate-settings-catalog.py`
    /// reads from the `preferenceRow(title:` literal at each call site.
    private func preferenceRow<Accessory: View>(
        title: String,
        description: String,
        @ViewBuilder accessory: () -> Accessory
    ) -> some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text(LocalizedStringKey(title), bundle: .module)
                    .settingsControl(title, pane: .appearance)
                    .font(theme.ui(.base, weight: .medium))
                    .foregroundStyle(.appText)
                    .fixedSize(horizontal: false, vertical: true)

                Text(LocalizedStringKey(description), bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 16)
            accessory()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    /// A segmented control when its segments fit beside the row's text, a
    /// pop-up menu of the same options when they do not (a large text size,
    /// a long translation). The fixed 140-220pt frames this replaces clipped
    /// segment titles instead. Both are the same native Picker with the
    /// same accessible name.
    private func adaptivePicker<Value: Hashable, Options: View>(
        _ name: Text,
        selection: Binding<Value>,
        @ViewBuilder options: @escaping () -> Options
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            Picker(selection: selection, content: options) { name }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            Picker(selection: selection, content: options) { name }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
        }
    }

    /// A stepper with its value in pixels beside it. The stepper carries
    /// the name and speaks the value; the visible number is hidden from
    /// VoiceOver so it is not read twice.
    private func sizeStepper(_ name: Text, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack(spacing: 6) {
            Stepper(value: value, in: range, step: 1) { name }
                .labelsHidden()
                .accessibilityValue(Text("\(Int(value.wrappedValue)) pixels", bundle: .module))

            Text(verbatim: "\(Int(value.wrappedValue))")
                .font(theme.ui(.callout, weight: .medium).monospacedDigit())
                .fixedSize()
                .frame(minWidth: 28, alignment: .center)
                .accessibilityHidden(true)

            Text(verbatim: "px")
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
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

    private func fontPicker(
        family: Binding<String>,
        weight: Binding<String>,
        options: [String],
        isCodeFont: Bool = false
    ) -> some View {
        HStack(spacing: 8) {
            // Family names are typeface names, shown as written.
            Picker(selection: family) {
                ForEach(options, id: \.self) { fam in
                    Text(verbatim: fam)
                        .font(AppFontDescriptor(family: fam, weight: .regular, size: 13, isCode: isCodeFont).font)
                        .tag(fam)
                }
            } label: {
                isCodeFont ? Text("Code font family", bundle: .module) : Text("UI font family", bundle: .module)
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(minWidth: 120, idealWidth: 140, maxWidth: 220)

            Picker(selection: weight) {
                Text("Regular", bundle: .module)
                    .settingsControl("Regular", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .regular, size: 13, isCode: isCodeFont).font).tag("Regular")
                Text("Medium", bundle: .module)
                    .settingsControl("Medium", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .medium, size: 13, isCode: isCodeFont).font).tag("Medium")
                Text("Semibold", bundle: .module)
                    .settingsControl("Semibold", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .semibold, size: 13, isCode: isCodeFont).font).tag("Semibold")
                Text("Bold", bundle: .module)
                    .settingsControl("Bold", pane: .appearance, timing: .immediate).font(AppFontDescriptor(family: family.wrappedValue, weight: .bold, size: 13, isCode: isCodeFont).font).tag("Bold")
            } label: {
                isCodeFont ? Text("Code font weight", bundle: .module) : Text("UI font weight", bundle: .module)
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(minWidth: 90, idealWidth: 110, maxWidth: 160)
        }
    }
}
