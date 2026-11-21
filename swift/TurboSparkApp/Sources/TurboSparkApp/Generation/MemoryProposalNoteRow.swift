import Combine
import SwiftUI

/// The transcript note under a `memory` tool step: what the model proposed
/// to remember, and Undo / Approve for that one claim.
///
/// The claim is found by the tool call's id (`MemoryProposalNote.resolution`),
/// never by text. While it is pending the row offers Undo (dismiss) and
/// Approve; once it is decided, here or in Settings, the row shows the
/// status instead. When the claim cannot be addressed (a ledger from before
/// call ids were recorded) only Review is offered, as before. The words
/// carry the state, so the accent glyph is decoration.
struct MemoryProposalNoteRow: View {
    @Environment(\.appTheme) private var theme
    let note: MemoryProposalNote
    /// Reads the claim's current state. Called on appear and whenever the
    /// ledger commits, so a Settings decision shows without a reload.
    let resolve: () -> MemoryProposalNote.Resolution?
    /// Applies Undo or Approve and returns the state after it.
    let act: (MemoryProposalNote.Action) -> MemoryProposalNote.Resolution?
    let onReview: () -> Void

    @State private var resolution: MemoryProposalNote.Resolution?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "brain")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Proposed for memory", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text(verbatim: note.text)
                    .themedFont(.small)
                    .foregroundStyle(theme.metadataForeground)
                    .lineLimit(2)
                if case .settled(let status) = resolution {
                    MemoryClaimText.status(status)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(accessibilityLabel)
            Spacer(minLength: 8)
            if case .pending = resolution {
                Button { resolution = act(.undo) } label: {
                    Text("Undo", bundle: .module)
                }
                .appAction(.quiet, size: .small)
                Button { resolution = act(.approve) } label: {
                    Text("Approve", bundle: .module)
                }
                .appAction(.secondary, size: .small)
            }
            Button(action: onReview) {
                Text("Review", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .help(Text("Open the Memory pane to approve or dismiss this proposal", bundle: .module))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
        .onAppear { resolution = resolve() }
        .onReceive(NotificationCenter.default
            .publisher(for: MemoryLedgerStore.didChangeNotification)
            .receive(on: DispatchQueue.main)) { _ in resolution = resolve() }
    }

    private var accessibilityLabel: Text {
        if case .settled(let status) = resolution {
            return Text(
                "Proposed for memory: \(note.text). Status: \(MemoryClaimText.status(status))",
                bundle: .module)
        }
        return Text(
            "Proposed for memory: \(note.text). It is not active until you approve it.",
            bundle: .module)
    }
}
