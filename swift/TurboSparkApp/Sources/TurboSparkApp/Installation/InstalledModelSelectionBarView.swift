import SwiftUI
import TurboSpark

/// The persistent bar of multi-select delete: how many are ticked, the space
/// deleting them would free, Select All, and the one destructive action.
///
/// Built on `AppAdaptiveHeader` so a long translation or Extra Large text moves
/// Delete under the counts instead of clipping either. Delete is disabled with
/// nothing ticked (the count beside it says why) and while a turn or a load
/// holds the engine (the tooltip says so).
struct InstalledModelSelectionBarView: View {
    let count: Int
    let bytes: UInt64
    /// Every selectable row is ticked, so the control now deselects.
    let allSelected: Bool
    let hasSelectable: Bool
    /// The app may delete right now (`AppModel.canDeleteModel`).
    let canDelete: Bool
    let isDeleting: Bool
    let onToggleAll: () -> Void
    let onDelete: () -> Void

    var body: some View {
        AppAdaptiveHeader {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("\(count) selected", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    if count > 0 {
                        Text("Frees \(MetricFormat.storage(bytes))", bundle: .module)
                            .themedFont(.tiny).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    }
                }
                Button(action: onToggleAll) {
                    if allSelected {
                        Text("Deselect All", bundle: .module)
                    } else {
                        Text("Select All", bundle: .module)
                    }
                }
                .appAction(.quiet, size: .small)
                .disabled(!hasSelectable || isDeleting)
            }
        } actions: {
            if isDeleting { ProgressView().controlSize(.small) }
            deleteButton
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.appElevated)
    }

    private var deleteButton: some View {
        Button(role: .destructive, action: onDelete) {
            Label { Text("Delete \(count) Models", bundle: .module) } icon: { Image(systemName: "trash") }
        }
        .appAction(.destructive, size: .small)
        .disabled(count == 0 || !canDelete || isDeleting)
        .help(
            canDelete
                ? Text("Delete \(count) Models", bundle: .module)
                : Text("Wait for the current reply or load to finish.", bundle: .module))
    }
}
