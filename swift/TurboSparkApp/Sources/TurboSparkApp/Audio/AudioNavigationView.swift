import SwiftUI

/// Shared by the rail (while the Audio panel is hidden), the workspace
/// header, and the native View menu. `AudioSidebarPanel` lists the same pages
/// in the same order.
@MainActor
struct AudioDestinationMenu: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ForEach(AudioWorkspacePage.destinations) { page in
            Button { model.openAudio(page) } label: {
                Label { Text(LocalizedStringKey(page.title), bundle: .module) }
                    icon: { Image(systemName: page.symbol) }
            }
        }
        Divider()
        Button { model.openAudio(.library) } label: {
            Label { Text("Audio library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
        }
        ForEach(AudioWorkspacePage.tools) { page in
            Button { model.openAudio(page) } label: {
                Label { Text(LocalizedStringKey(page.title), bundle: .module) }
                    icon: { Image(systemName: page.symbol) }
            }
        }
    }
}
