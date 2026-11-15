import SwiftUI

/// The alternate chords shared with unsloth studio, registered beside the
/// menu commands so both habits work.
///
/// A SwiftUI menu item carries exactly one keyboard shortcut, so a chord the
/// menu does not show can only exist as a window-level hidden button -- the
/// same mechanism the Cmd+K palette uses for its arrows (swift/CLAUDE.md
/// Gotcha 58), which keeps firing while a text field is first responder.
/// The Keyboard Shortcuts pane lists these under their row's alternate keys;
/// the menus show only the primary chord.
///
/// The chords come from `KeyboardShortcutCatalog.alternate(for:)`, the one
/// inventory the Shortcuts pane and the OC-01 conflict check also read. An
/// alternate is a MIRROR of its menu item, never an override source: the user
/// rebinds the menu chord, these stay fixed, and the conflict check refuses
/// any menu chord that equals one of them.
///
/// Every button's disabled state mirrors the menu command it shadows, so an
/// alternate never reaches an action the advertised key would refuse. The
/// labels are verbatim: the buttons are invisible and hidden from VoiceOver,
/// so no reader ever sees or hears them.
struct AlternateShortcutBridge: View {
    @ObservedObject var model: AppModel
    let toggleSidebar: () -> Void

    var body: some View {
        VStack {
            // unsloth newChat: Cmd+Shift+O. Its own Cmd+N alternate is the
            // chord the Chat menu already advertises.
            Button { model.createChat() } label: { Text(verbatim: "New Chat Alternate") }
                .alternateChord(.newChat)
                .disabled(model.isRunning)

            // unsloth previousChat / nextChat: Cmd+Shift+[ and Cmd+Shift+].
            Button { model.selectPreviousChat() } label: { Text(verbatim: "Previous Chat Alternate") }
                .alternateChord(.previousChat)
                .disabled(model.isRunning || model.orderedChats.isEmpty)
            Button { model.selectNextChat() } label: { Text(verbatim: "Next Chat Alternate") }
                .alternateChord(.nextChat)
                .disabled(model.isRunning || model.orderedChats.isEmpty)

            // unsloth toggleSidebar: Cmd+B.
            // Mirrors the View menu's Toggle Sidebar disabled state.
            Button { toggleSidebar() } label: { Text(verbatim: "Toggle Sidebar Alternate") }
                .alternateChord(.toggleSidebar)
                .disabled(!model.activeSection.hasSidebarPanel)

            // unsloth switchToChat..switchToExport: Ctrl+1..Ctrl+9 on macOS.
            // The rail has eight sections, so the range stops at eight.
            ForEach(AppModel.AppNavigationSection.allCases) { section in
                Button(section.title) { model.activeSection = section }
                    .alternateChord(.navigation(section))
            }
        }
        .opacity(0)
        .frame(width: 0, height: 0)
        .accessibilityHidden(true)
    }
}

private extension View {
    /// The catalog's hidden alternate for `id`, or no shortcut if it has none.
    @ViewBuilder
    func alternateChord(_ id: KeyboardCommandID) -> some View {
        if let chord = KeyboardShortcutCatalog.alternate(for: id) {
            keyboardShortcut(chord.keyEquivalent, modifiers: chord.eventModifiers)
        } else {
            self
        }
    }
}
