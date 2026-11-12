import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Audio section's panel beside the icon rail: every Audio page, one click
/// away, in the same order as `AudioDestinationMenu` (destinations, Library,
/// tools).
///
/// This replaced the Audio sub-rows that expanded under the Audio row of the
/// old labeled section list. The rail's Audio item is a plain button while
/// this panel is shown, and falls back to `AudioDestinationMenu` when the
/// panel is hidden, so the pages stay one step away either way.
@MainActor
struct AudioSidebarPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        SidebarPanelList(title: Text("Audio", bundle: .module)) {
            ForEach(AudioWorkspacePage.destinations) { row($0) }
            SidebarDivider().padding(.horizontal, 8).padding(.vertical, 6)
            row(.library)
            SidebarDivider().padding(.horizontal, 8).padding(.vertical, 6)
            ForEach(AudioWorkspacePage.tools) { row($0) }
        }
        .tsAnimation(TSMotion.select, value: controller.page)
    }

    private func row(_ page: AudioWorkspacePage) -> some View {
        SidebarPanelRow(
            title: Text(LocalizedStringKey(page.title), bundle: .module),
            systemImage: page.symbol,
            isSelected: controller.page == page
        ) { model.openAudio(page) }
    }
}
