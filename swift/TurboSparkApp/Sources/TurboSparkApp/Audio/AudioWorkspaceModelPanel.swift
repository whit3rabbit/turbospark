import SwiftUI

/// Keep model setup in view when a take cannot run. An installed model
/// takes one line and opens the same model controls when needed.
@MainActor
struct AudioWorkspaceModelPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        Group {
            if let profile = controller.selectedProfile, controller.modelInstalled {
                DisclosureGroup {
                    picker.padding(.top, 8)
                } label: {
                    Label {
                        Text(verbatim: profile.displayName)
                            .fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: "cpu").foregroundStyle(.appAccentText)
                    }
                    .themedFont(.small, weight: .medium)
                }
            } else {
                picker
            }
        }
        .appCardSurface(padding: 12)
    }

    private var picker: some View {
        AudioModelPickerView(model: model, controller: controller)
    }
}
