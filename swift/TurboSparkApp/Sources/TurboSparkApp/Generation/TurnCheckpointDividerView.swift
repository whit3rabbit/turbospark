import SwiftUI

/// The row above a turn that changed files: what the checkpoint holds and a
/// Restore action. Renders nothing for a turn with no checkpoint.
///
/// Observes only the checkpoint index, so a capture mid-turn redraws these
/// rows and not the whole transcript.
struct TurnCheckpointDividerSlot: View {
    @ObservedObject var index: TurnCheckpointIndex
    let model: AppModel
    let chatID: UUID
    let turnID: UUID
    /// Passed in rather than read from `model`, which this view does not
    /// observe: the parent redraws when a turn starts or ends.
    let isRunning: Bool

    var body: some View {
        if let manifest = index.manifestLoading(chatID: chatID, turnID: turnID), !manifest.entries.isEmpty {
            TurnCheckpointDividerView(manifest: manifest, isRunning: isRunning) {
                model.requestCheckpointRestore(chatID: chatID, turnID: turnID)
            }
            .confirmationDialog(
                Text("Restore files to before this turn?", bundle: .module),
                isPresented: isPresented,
                titleVisibility: .visible,
                presenting: index.pendingRestore
            ) { pending in
                Button(role: .destructive) {
                    Task { await model.confirmCheckpointRestore(pending) }
                } label: {
                    Text("Restore Files", bundle: .module)
                }
                Button(role: .cancel) {
                    index.pendingRestore = nil
                } label: {
                    Text("Cancel", bundle: .module)
                }
            } message: { pending in
                Text(verbatim: TurnCheckpointRestoreText.message(pending.plan))
            }
        }
    }

    /// Only the slot of the turn being restored presents the dialog.
    private var isPresented: Binding<Bool> {
        Binding(
            get: { index.pendingRestore?.chatID == chatID && index.pendingRestore?.turnID == turnID },
            set: { shown in
                if !shown, index.pendingRestore?.turnID == turnID { index.pendingRestore = nil }
            })
    }
}

struct TurnCheckpointDividerView: View {
    @Environment(\.appTheme) private var theme
    let manifest: TurnCheckpointManifest
    let isRunning: Bool
    let onRestore: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "clock.arrow.circlepath")
                    .frame(width: (ConversationLayout.activityIconWidth * AppMetrics.textScale(theme)).rounded())
                Text("Checkpoint before this turn", bundle: .module)
                Image(systemName: "doc")
                Text(verbatim: "\(manifest.entries.count)").monospacedDigit()
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(Text("Checkpoint before this turn. Files: \(manifest.entries.count)", bundle: .module))

            // Status is glyph plus word, never color alone.
            if manifest.isIncomplete {
                Label {
                    Text("Incomplete", bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .foregroundStyle(.appWarning)
                .help(Text("Some files were not captured: too large, over the turn limit, sensitive, or unreadable.", bundle: .module))
            }
            if manifest.restoredAt != nil {
                Label {
                    Text("Restored", bundle: .module)
                } icon: {
                    Image(systemName: "checkmark.circle")
                }
                .foregroundStyle(.appSuccess)
            }

            ConversationDivider()

            Button(action: onRestore) {
                Text("Restore", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .disabled(isRunning)
            .help(isRunning
                ? Text("Wait for the current turn to finish before restoring.", bundle: .module)
                : Text("Shows the files that will be put back before anything changes.", bundle: .module))
        }
        .themedFont(.small, weight: .medium)
        .foregroundStyle(.appSecondary)
        .frame(minHeight: (ConversationLayout.activityHeight * AppMetrics.textScale(theme)).rounded())
        .accessibilityElement(children: .contain)
    }
}
