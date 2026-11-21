import SwiftUI

/// The user's own notes on a meeting, typed during or after the call. The
/// summary reads them as the points that matter; nothing generated writes
/// them back.
@MainActor
struct AudioMeetingNotesEditor: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    @FocusState private var focused: Bool
    @Environment(\.appTheme) private var theme

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Label { Text("My notes", bundle: .module) } icon: { Image(systemName: "square.and.pencil") }
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
                .padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 4)
            AudioMeasuredTextEditor(
                text: Binding(get: { controller.userNotes(for: item) },
                              set: { controller.stageUserNotes($0, for: item.id) }),
                label: Text("My notes", bundle: .module),
                placeholder: Text("Jot down what matters. Generated notes build on these.", bundle: .module),
                minHeight: (160 * scale).rounded(), maxHeight: (520 * scale).rounded(),
                focus: $focused)
                .padding(.horizontal, 11)
            Text("Saved as you type. Markdown is fine.", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary)
                .padding(.horizontal, 16).padding(.bottom, 10)
        }
        .appComposerSurface(focused: focused, halo: false)
        .onChange(of: focused) { _, isFocused in
            if !isFocused { controller.flushUserNotes(for: item.id) }
        }
        .onDisappear { controller.flushUserNotes(for: item.id) }
    }
}

/// Notes beside the meeting's transcript and summary when there is room,
/// stacked when there is not. Both columns state an ideal width, so the
/// switch point follows text size the way `AppMetrics.textScale` scales it.
@MainActor
struct AudioMeetingColumns<Notes: View, Panel: View>: View {
    @ViewBuilder let notes: () -> Notes
    @ViewBuilder let panel: () -> Panel
    @Environment(\.appTheme) private var theme

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 20) {
                notes()
                    .frame(minWidth: (260 * scale).rounded(), idealWidth: (320 * scale).rounded(),
                           maxWidth: (440 * scale).rounded(), alignment: .topLeading)
                panel()
                    .frame(minWidth: (380 * scale).rounded(), idealWidth: (460 * scale).rounded(),
                           maxWidth: .infinity, alignment: .topLeading)
            }
            VStack(alignment: .leading, spacing: 16) {
                notes()
                panel()
            }
        }
    }
}
