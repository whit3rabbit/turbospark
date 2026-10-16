import SwiftUI
import TurboSpark

/// Words as they are spoken, while a take records with streaming live text.
/// Committed text is never revised; the word still being heard is shown
/// apart from it. Announces state changes, not each word.
@MainActor
struct AudioLiveTextView: View {
    @ObservedObject var live: AudioLiveText
    @ObservedObject private var appearance = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let status = live.status {
                Label {
                    Text(status).fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: live.isStreaming ? "hourglass" : "exclamationmark.circle").accessibilityHidden(true)
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            if !live.text.isEmpty || live.isStreaming {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            transcript
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Color.clear.frame(height: 1).id(Self.end)
                        }
                        .padding(10)
                    }
                    .frame(minHeight: 72, maxHeight: 220)
                    .background(.appPage, in: RoundedRectangle(cornerRadius: 8))
                    .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder, lineWidth: 1) }
                    .onChange(of: live.text) { _, _ in
                        let reduce = appearance.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
                        if reduce { proxy.scrollTo(Self.end, anchor: .bottom) }
                        else { withAnimation(TSMotion.select) { proxy.scrollTo(Self.end, anchor: .bottom) } }
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(Text("Live transcript", bundle: .module))
            }
        }
    }

    private static let end = "live-text-end"

    @ViewBuilder private var transcript: some View {
        if live.text.isEmpty && live.isStreaming {
            Text("Listening...", bundle: .module).foregroundStyle(.appSecondary)
        } else {
            Text(verbatim: live.text)
        }
    }
}

/// Live text's model and why a take will or will not stream, with the
/// download when the model is missing. A disabled state always says why.
@MainActor
struct AudioLiveTextModelRow: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    private var download: ModelDownload? {
        controller.liveTextProfile.flatMap { model.audioDownload(for: $0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let reason = controller.liveStreamingUnavailableReason {
                Label { Text(reason).fixedSize(horizontal: false, vertical: true) } icon: {
                    Image(systemName: "text.bubble").accessibilityHidden(true)
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                if let profile = controller.liveTextProfile, controller.liveTextModelPath == nil, profile.capabilities.canRun {
                    if let download, !download.status.isTerminal {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Downloading the live text model", bundle: .module).themedFont(.small)
                        }
                    } else {
                        Button { controller.installLiveTextModel() } label: {
                            Label {
                                Text("Download live text model (\(ByteCountFormatter.string(fromByteCount: Int64(profile.downloadBytes), countStyle: .file)))", bundle: .module)
                            } icon: { Image(systemName: "arrow.down.circle") }
                        }
                        .appAction(.secondary, size: .small)
                        .disabled(!model.canQueueModelDownload(.audio(identity: profile.identity)) || controller.hasRecordingActivity)
                    }
                }
            } else {
                Label {
                    Text("Words appear as you speak, about a second behind (English).", bundle: .module)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: { Image(systemName: "text.bubble.fill").accessibilityHidden(true) }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
        }
    }
}
