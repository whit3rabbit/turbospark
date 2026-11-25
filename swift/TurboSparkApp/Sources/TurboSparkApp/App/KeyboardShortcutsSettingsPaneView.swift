import SwiftUI
import TurboSpark

/// Settings pane presenting the keyboard shortcuts, with a per-row record
/// control for the commands a menu item carries (OC-01). The rows come from
/// `KeyboardShortcutCatalog` resolved through the override store, which is
/// what keeps them honest against the commands in `TurboSparkApp.swift`.
@MainActor
struct KeyboardShortcutsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var keybinds = KeybindOverrideStore.shared

    /// The row whose recorder is armed. One at a time: arming another row
    /// disarms this one, so there is never more than one event monitor.
    @State private var recordingID: KeyboardCommandID?
    /// The last refusal or confirmation, shown under the row it concerns.
    @State private var notice: Notice?
    @State private var confirmReset = false

    private struct Notice: Equatable {
        let commandID: KeyboardCommandID
        let text: String
        let isError: Bool
    }

    var body: some View {
        Form {
            Section {
                Text(
                    "Select a shortcut, then press the new keys. Escape cancels. A shortcut marked Fixed is shared with a control in the window and cannot be changed.",
                    bundle: .module
                )
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                HStack {
                    Spacer(minLength: 0)
                    Button(role: .destructive) {
                        confirmReset = true
                    } label: {
                        Text("Reset All Shortcuts", bundle: .module)
                    }
                    .appAction(.destructive, size: .small)
                    .disabled(keybinds.map.overrides.isEmpty)
                }
            }

            ForEach(KeyboardShortcutCatalog.sections(using: keybinds.map)) { section in
                Section {
                    ForEach(section.rows) { row in
                        shortcutRow(row)
                    }
                } header: {
                    // Section titles and row labels are catalog keys, the
                    // same keys the menu items use, so the pane and the menu
                    // bar read alike in every language.
                    Text(LocalizedStringKey(section.title), bundle: .module)
                }
            }
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .confirmationDialog(
            Text("Reset all shortcuts?", bundle: .module), isPresented: $confirmReset
        ) {
            Button(role: .destructive) {
                recordingID = nil
                keybinds.resetAll()
                notice = nil
                announce(AppStrings.localized("All shortcuts reset to their defaults."))
            } label: {
                Text("Reset All Shortcuts", bundle: .module)
            }
        } message: {
            Text("Every shortcut you changed goes back to its default.", bundle: .module)
        }
    }

    /// One row per command. Side by side when it fits; the keys drop under
    /// the label at a large text size or with a long translation instead of
    /// truncating the label. A fixed row reads to VoiceOver as one element,
    /// "label, keys, fixed"; a rebindable row is its record button (named for
    /// the command, valued with the chord) and, when changed, a restore
    /// button, so neither is buried in a combined element.
    private func shortcutRow(_ row: KeyboardShortcutRow) -> some View {
        let label = Text(LocalizedStringKey(row.label), bundle: .module)
        return VStack(alignment: .leading, spacing: 4) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    label.font(theme.ui(.base)).accessibilityHidden(row.rebindable)
                    Spacer(minLength: 12)
                    controls(row)
                }
                VStack(alignment: .leading, spacing: 6) {
                    label
                        .font(theme.ui(.base))
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityHidden(row.rebindable)
                    controls(row)
                }
            }
            if let notice, notice.commandID == row.commandID {
                noticeView(notice)
            }
        }
        .modifier(FixedRowAccessibility(row: row, label: label, active: !row.rebindable))
    }

    @ViewBuilder
    private func controls(_ row: KeyboardShortcutRow) -> some View {
        HStack(spacing: 8) {
            if let alt = row.altKeys {
                Text("also \(alt)", bundle: .module)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(row.rebindable)
            }
            if row.rebindable {
                KeybindCaptureField(
                    name: Text(LocalizedStringKey(row.label), bundle: .module),
                    keys: row.keys,
                    alternateKeys: row.altKeys,
                    isRecording: recordingBinding(row.commandID),
                    onCapture: { capture($0, for: row) },
                    onUnsupported: {
                        show(
                            row, AppStrings.localized("That key cannot be used in a shortcut."),
                            isError: true)
                    })
                if row.isCustom {
                    restoreButton(row)
                }
            } else {
                fixedBadge(row)
            }
        }
        .fixedSize()
    }

    private func fixedBadge(_ row: KeyboardShortcutRow) -> some View {
        HStack(spacing: 6) {
            Text(verbatim: row.keys)
                .font(theme.code(.small))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.appSurface)
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 1)
                )
            // A glyph and a word, so "fixed" never depends on color.
            Label { Text("Fixed", bundle: .module) } icon: { Image(systemName: "lock") }
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
        .help(Text("Shared with a control in the window, so it cannot be changed.", bundle: .module))
    }

    private func restoreButton(_ row: KeyboardShortcutRow) -> some View {
        let name = AppStrings.localized(String.LocalizationValue(row.label))
        return Button {
            restore(row)
        } label: {
            Label { Text("Restore Default", bundle: .module) } icon: {
                Image(systemName: "arrow.uturn.backward")
            }
            .labelStyle(.iconOnly)
        }
        .buttonStyle(.borderless)
        .help(Text("Restore Default", bundle: .module))
        .accessibilityLabel(Text("Restore default for \(name)", bundle: .module))
    }

    private func noticeView(_ notice: Notice) -> some View {
        Label {
            Text(verbatim: notice.text)
        } icon: {
            Image(systemName: notice.isError ? "exclamationmark.triangle" : "checkmark.circle")
        }
        .font(theme.ui(.small))
        .foregroundStyle(notice.isError ? AppThemeColor.appDanger : AppThemeColor.appSecondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Actions

    private func recordingBinding(_ id: KeyboardCommandID) -> Binding<Bool> {
        Binding(
            get: { recordingID == id },
            set: { recording in
                if recording {
                    notice = nil
                    recordingID = id
                } else if recordingID == id {
                    recordingID = nil
                }
            })
    }

    private func capture(_ chord: KeyChord, for row: KeyboardShortcutRow) {
        let name = AppStrings.localized(String.LocalizationValue(row.label))
        switch keybinds.assign(chord, to: row.commandID) {
        case .assigned, .restoredDefault:
            let text = AppStrings.localized("Shortcut for \(name) set to \(chord.display).")
            show(row, text, isError: false)
        case .unchanged:
            notice = nil
        case .refused(let refusal):
            show(row, refusal.message(for: chord), isError: true)
        }
    }

    private func restore(_ row: KeyboardShortcutRow) {
        let name = AppStrings.localized(String.LocalizationValue(row.label))
        let fallback = KeyboardShortcutCatalog.defaultChord(for: row.commandID)
        switch keybinds.restoreDefault(row.commandID) {
        case .assigned, .restoredDefault:
            let keys = fallback?.display ?? row.keys
            show(
                row, AppStrings.localized("Shortcut for \(name) restored to \(keys)."),
                isError: false)
        case .unchanged:
            notice = nil
        case .refused(let refusal):
            if let fallback {
                show(row, refusal.message(for: fallback), isError: true)
            }
        }
    }

    private func show(_ row: KeyboardShortcutRow, _ text: String, isError: Bool) {
        notice = Notice(commandID: row.commandID, text: text, isError: isError)
        announce(text)
    }

    private func announce(_ text: String) {
        AccessibilityNotification.Announcement(text).post()
    }
}

/// A fixed row is one VoiceOver element, "label, keys, fixed". A rebindable
/// row is left alone: its record button carries its own name and value.
private struct FixedRowAccessibility: ViewModifier {
    let row: KeyboardShortcutRow
    let label: Text
    let active: Bool

    func body(content: Content) -> some View {
        if active {
            content
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(spoken)
        } else {
            content.accessibilityElement(children: .contain)
        }
    }

    private var spoken: Text {
        var text = label + Text(verbatim: ", ") + Text(verbatim: row.keys)
        if let alt = row.altKeys {
            text = text + Text(verbatim: ", ") + Text("also \(alt)", bundle: .module)
        }
        return text + Text(verbatim: ", ") + Text("Fixed", bundle: .module)
    }
}
