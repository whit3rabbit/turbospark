import SwiftUI

@MainActor
struct AudioMusicWorkspaceView<PromptActions: View>: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let showGenerator: () -> Void
    @ViewBuilder let promptActions: () -> PromptActions

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                let wide = geometry.size.width >= 1000 * scale
                let layout = wide
                    ? AnyLayout(HStackLayout(alignment: .top, spacing: 24))
                    : AnyLayout(VStackLayout(alignment: .leading, spacing: 24))
                layout {
                    composer
                        .frame(maxWidth: 720 * scale, alignment: .leading)
                    listening
                        .frame(maxWidth: (wide ? 420 : 720) * scale, alignment: .leading)
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .top)
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 14) {
            AudioMusicInputs(controller: controller, showGenerator: showGenerator)
            Divider()
            AudioRunControls(controller: controller, actions: promptActions)
        }
        .appCardSurface(padding: 16)
    }

    private var listening: some View {
        let takes = controller.musicTakes
        return VStack(alignment: .leading, spacing: 16) {
            if let gate = controller.modelGate {
                AudioModelGateView(model: model, controller: controller, gate: gate, inline: true)
            } else {
                AudioWorkspaceModelPanel(model: model, controller: controller)
            }
            if let item = controller.workspaceResult, item.status != .processing {
                AudioItemDetailView(controller: controller, item: item, showsComparisonMenu: false)
            }
            if takes.isEmpty {
                AppEmptyState(
                    systemImage: "waveform",
                    title: Text("No outputs yet", bundle: .module),
                    message: Text("Describe a soundtrack and keep each take for comparison.", bundle: .module)
                )
                .padding(.vertical, 12)
            } else {
                AppAdaptiveHeader {
                    AppSectionHeader(title: Text("Takes", bundle: .module))
                } actions: {
                    if let item = controller.workspaceResult, item.hasPlayableAudio {
                        let candidates = takes.filter { $0.id != (controller.playbackItemID ?? item.id) }
                        if !candidates.isEmpty { compareMenu(candidates) }
                    }
                }
                AudioLibraryList(controller: controller, items: takes)
            }
        }
    }

    private func compareMenu(_ candidates: [AudioLibraryItem]) -> some View {
        Menu {
            ForEach(candidates) { candidate in
                Button { controller.compare(with: candidate) } label: { Text(verbatim: candidate.title) }
            }
        } label: {
            Label { Text("Compare at this position", bundle: .module) } icon: {
                Image(systemName: "arrow.left.arrow.right")
            }
        }
        .appMenuAction(size: .small, quiet: true)
        .disabled(controller.hasRecordingActivity)
        .help(Text("Switch takes while keeping the playback position.", bundle: .module))
    }
}
