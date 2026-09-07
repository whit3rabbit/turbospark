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
/// Every button's disabled state mirrors the menu command it shadows, so an
/// alternate never reaches an action the advertised key would refuse.
struct AlternateShortcutBridge: View {
    @ObservedObject var model: AppModel
    let toggleSidebar: () -> Void

    var body: some View {
        VStack {
            // unsloth newChat: Cmd+Shift+O. Its own Cmd+N alternate is the
            // chord the Chat menu already advertises.
            Button("New Chat Alternate") { model.createChat() }
                .keyboardShortcut("o", modifiers: [.command, .shift])
                .disabled(model.isRunning)

            // unsloth previousChat / nextChat: Cmd+Shift+[ and Cmd+Shift+].
            Button("Previous Chat Alternate") { model.selectPreviousChat() }
                .keyboardShortcut("[", modifiers: [.command, .shift])
                .disabled(model.isRunning || model.orderedChats.isEmpty)
            Button("Next Chat Alternate") { model.selectNextChat() }
                .keyboardShortcut("]", modifiers: [.command, .shift])
                .disabled(model.isRunning || model.orderedChats.isEmpty)

            // unsloth toggleSidebar: Cmd+B.
            Button("Toggle Chat Sidebar Alternate") { toggleSidebar() }
                .keyboardShortcut("b", modifiers: .command)

            // unsloth switchToChat..switchToExport: Ctrl+1..Ctrl+9 on macOS.
            // The rail has five sections, so the range stops at five.
            ForEach(AppModel.AppNavigationSection.allCases) { section in
                Button(section.title) { model.activeSection = section }
                    .keyboardShortcut(KeyEquivalent(section.shortcutKey), modifiers: .control)
            }
        }
        .opacity(0)
        .frame(width: 0, height: 0)
        .accessibilityHidden(true)
    }
}
