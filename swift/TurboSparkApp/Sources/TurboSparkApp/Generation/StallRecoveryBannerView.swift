import SwiftUI

/// Sits above the composer when the engine has gone silent mid-reply (OC-14):
/// Keep Waiting, Stop Turn or Restart TurboSpark. Modeled on
/// `ErrorBanner` and `NewChatSuggestionBanner`, the other attention strips in
/// the conversation chrome.
///
/// Keep Waiting is the one primary action (the safe default). It carries no
/// `.defaultAction` shortcut on purpose: this is an inline strip, not a
/// sheet, and Return belongs to the composer. All three are ordinary
/// buttons, so Tab and Space reach them. The strip is announced once when it
/// appears (`AppModel.makeStallWatchdog`).
@MainActor
struct StallRecoveryBannerView: View {
    let notice: GenerationStallNotice?
    let onKeepWaiting: () -> Void
    let onStop: () -> Void
    let onRestart: () -> Void

    /// The chat strip: reads the notice and routes the three actions through
    /// the app model.
    init(model: AppModel) {
        notice = model.generationStallNotice
        onKeepWaiting = { model.keepWaitingAfterStall() }
        onStop = { model.stopStalledTurn() }
        onRestart = { model.restartAfterStall() }
    }

    /// Any other local engine call (Humanize owns its session and its own
    /// monitor), same strip, same copy.
    init(
        notice: GenerationStallNotice?,
        onKeepWaiting: @escaping () -> Void,
        onStop: @escaping () -> Void,
        onRestart: @escaping () -> Void
    ) {
        self.notice = notice
        self.onKeepWaiting = onKeepWaiting
        self.onStop = onStop
        self.onRestart = onRestart
    }

    var body: some View {
        if let notice {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.title3)
                        .foregroundStyle(AppAttention.color)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("The model has stopped responding", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)
                        Text(
                            "No output for about \(notice.silenceSeconds) seconds. A very large prompt can take this long.",
                            bundle: .module
                        )
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        Text(
                            "Stop ends this turn, but a stuck engine keeps later messages waiting. Restart saves your chats and reopens TurboSpark.",
                            bundle: .module
                        )
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityElement(children: .combine)
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { actions }
                    VStack(alignment: .leading, spacing: 8) { actions }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background {
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .fill(.appSurface)
                    .overlay {
                        RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                            .stroke(AppAttention.color, lineWidth: 1)
                    }
            }
            .transition(.opacity.combined(with: .move(edge: .bottom)))
            .accessibilityElement(children: .contain)
        }
    }

    @ViewBuilder
    private var actions: some View {
        Button {
            onKeepWaiting()
        } label: {
            Text("Keep Waiting", bundle: .module).themedFont(.small, weight: .semibold)
        }
        .appAction(.primary, size: .small)
        .help(Text("Hide this notice. It returns if there is still no progress.", bundle: .module))

        Button {
            onStop()
        } label: {
            Text("Stop Turn", bundle: .module).themedFont(.small)
        }
        .appAction(.secondary, size: .small)
        .help(Text("End this turn and keep the partial reply", bundle: .module))

        Button {
            onRestart()
        } label: {
            Text("Restart TurboSpark", bundle: .module).themedFont(.small)
        }
        .appAction(.secondary, size: .small)
        .help(Text("Save your chats and reopen the app", bundle: .module))
    }
}
