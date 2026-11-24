import SwiftUI

/// Sits above the composer when the chat model did not load: a title, the
/// cause in one line, and the fixes that exist for it (`ModelLoadFailure`).
/// Modeled on `StallRecoveryBannerView`, the other attention strip in this
/// slot.
///
/// The first action is the primary button. None carries `.defaultAction`:
/// Return belongs to the composer. The strip describes an event, so it can be
/// dismissed, and it is announced once per failure.
@MainActor
struct ModelLoadFailureBannerView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        if let failure = model.modelLoadFailure {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.title3)
                        .foregroundStyle(AppAttention.color)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(verbatim: failure.title())
                            .themedFont(.small, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)
                        Text(verbatim: failure.cause())
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(3)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                    .accessibilityElement(children: .combine)
                    // A specific cause replaces the engine's words on screen;
                    // they stay one hover away for a bug report.
                    .help(Text(verbatim: failure.detail))

                    Spacer(minLength: 8)

                    AppIconButton("xmark", label: Text("Dismiss error", bundle: .module)) {
                        model.modelLoadFailure = nil
                    }
                    .appAction(.quiet, size: .small)
                    .accessibilityHint(Text("Dismisses this message", bundle: .module))
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { actions(failure) }
                    VStack(alignment: .leading, spacing: 8) { actions(failure) }
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
            .onChange(of: failure.id, initial: true) { _, _ in
                let spoken = "\(failure.title()). \(failure.cause())"
                _ = AccessibilityNotification.Announcement.post(.init(spoken))
            }
        }
    }

    @ViewBuilder
    private func actions(_ failure: ModelLoadFailure) -> some View {
        ForEach(Array(failure.kind.actions.enumerated()), id: \.offset) { index, action in
            Button {
                perform(action, failure)
            } label: {
                label(for: action)
                    .themedFont(.small, weight: index == 0 ? .semibold : .regular)
            }
            .appAction(index == 0 ? .primary : .secondary, size: .small)
        }
    }

    private func label(for action: ModelLoadFailure.Action) -> Text {
        switch action {
        case .retry: return Text("Try again", bundle: .module)
        case .chooseModel: return Text("Installed Models", bundle: .module)
        case .engineSettings: return Text("Engine settings...", bundle: .module)
        }
    }

    private func perform(_ action: ModelLoadFailure.Action, _ failure: ModelLoadFailure) {
        switch action {
        case .retry:
            model.modelLoadFailure = nil
            model.requestLoad(failure.model, purpose: .chat(switchToChat: false))
        case .chooseModel:
            model.activeSection = .modelManager
        case .engineSettings:
            model.openSettings(tab: .engine)
        }
    }
}
