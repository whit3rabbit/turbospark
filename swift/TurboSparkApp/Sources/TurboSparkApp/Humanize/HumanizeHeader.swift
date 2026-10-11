import SwiftUI

/// The section header: the title, then the model bar and the Advanced toggle.
///
/// `AppAdaptiveHeader` rather than one fixed row, which truncated the model
/// bar in a narrow window and at the largest text size. The actions move
/// under the title when the row does not fit; in the narrowest layout the
/// Advanced toggle folds into a labeled menu so the model bar, which holds
/// Load and Download, stays visible.
@MainActor
struct HumanizeHeader: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: HumanizeController
    @Binding var showAdvanced: Bool
    let onDelete: () -> Void

    var body: some View {
        AppAdaptiveHeader {
            HStack(spacing: 12) {
                Image(systemName: "text.badge.checkmark")
                    .symbolRenderingMode(.monochrome)
                    .themedFont(.title2).foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Humanize", bundle: .module)
                    .themedFont(.title2, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
            }
        } actions: {
            modelBar
            AppChip(Text("Advanced", bundle: .module), systemImage: "sidebar.right", isOn: showAdvanced) {
                TSMotion.animate(TSMotion.pane) { showAdvanced.toggle() }
            }
        } compactActions: {
            modelBar
            Menu {
                // A Toggle, so the menu item carries the panel's on state as
                // a checkmark the way the chip carries it as a tint.
                Toggle(isOn: animatedAdvanced) {
                    Label { Text("Advanced", bundle: .module) } icon: { Image(systemName: "sidebar.right") }
                }
            } label: {
                Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction()
        }
        .padding(.horizontal, 24).padding(.vertical, 12)
    }

    private var modelBar: some View {
        HumanizeModelBar(model: model, controller: controller, onDelete: onDelete)
    }

    /// The panel slides in from the menu exactly as it does from the chip.
    private var animatedAdvanced: Binding<Bool> {
        Binding(get: { showAdvanced }, set: { value in
            TSMotion.animate(TSMotion.pane) { showAdvanced = value }
        })
    }
}
