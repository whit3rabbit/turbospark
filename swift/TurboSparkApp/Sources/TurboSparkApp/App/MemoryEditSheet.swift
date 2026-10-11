import SwiftUI

/// Edits the text of one approved memory claim.
///
/// Its own view (it was inline in `MemorySettingsPaneView`) so the editor
/// carries the shared surface, an accessible name and the standard sheet
/// keys: Return saves, Escape cancels.
struct MemoryEditSheet: View {
    @Binding var draftText: String
    /// Returns whether the save took; a failed save keeps the sheet and the
    /// draft open so nothing typed is lost.
    let onSave: () -> Bool
    let onCancel: () -> Void

    private var canSave: Bool {
        !draftText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit Memory", bundle: .module)
                .themedFont(.title3, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            TextEditor(text: $draftText)
                .themedFont(.base)
                .appEditorSurface(minHeight: 220)
                .frame(minWidth: 520)
                .accessibilityLabel(Text("Memory text", bundle: .module))
            HStack(spacing: AppMetrics.actionSpacing) {
                Spacer()
                Button(role: .cancel) { onCancel() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
                Button {
                    _ = onSave()
                } label: { Text("Save", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
            }
        }
        .padding(20)
    }
}

/// Display names for the memory ledger's stored identifiers. The raw values
/// are persisted in the encrypted profile and must not change, so the
/// translation happens here, one literal per known value.
enum MemoryClaimText {
    static func status(_ status: MemoryClaimStatus) -> Text {
        switch status {
        case .pending: return Text("Pending", bundle: .module)
        case .active: return Text("Active", bundle: .module)
        case .legacy: return Text("Legacy", bundle: .module)
        case .superseded: return Text("Superseded", bundle: .module)
        case .retracted: return Text("Retracted", bundle: .module)
        case .dismissed: return Text("Dismissed", bundle: .module)
        }
    }

    /// `kind` is open-ended (a model may propose its own), so only the two
    /// the app writes itself are translated; any other renders as stored.
    static func kind(_ kind: String) -> Text {
        switch kind {
        case "preference": return Text("Preference", bundle: .module)
        case "legacy": return Text("Legacy", bundle: .module)
        default: return Text(verbatim: kind.capitalized)
        }
    }

    /// A `String` because it is interpolated into the "Evidence" line.
    static func confidence(_ confidence: String) -> String {
        switch confidence {
        case "source-matched": return AppStrings.localized("Source matched")
        case "source-unverified": return AppStrings.localized("Source unverified")
        case "user-authored": return AppStrings.localized("User authored")
        case "user-confirmed": return AppStrings.localized("User confirmed")
        default: return confidence
        }
    }
}
