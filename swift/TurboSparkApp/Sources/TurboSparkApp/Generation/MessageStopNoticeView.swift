import AppKit
import SwiftUI
import TurboSpark

/// Why a committed assistant reply is incomplete, read from its persisted
/// `stopReason`.
///
/// The reason was stored on every row but only the global `ErrorBanner`
/// reported a failure, and that banner is transient and dismissible, so a
/// reopened chat could not tell a finished answer from one an engine error
/// or a full context window cut off. Pure so the reason-to-notice mapping is
/// testable without a model, the way `ChatActivityStatus` is.
enum MessageStopNotice: Equatable {
    case stoppedByUser
    case engineError
    case contextFull
    case lengthLimit
    case turnLimit

    enum Severity: Equatable {
        /// The user chose this; incomplete is expected, not a problem.
        case neutral
        case warning
        case danger
    }

    /// The notice for `message`, or nil for a reply that finished normally
    /// or carries a reason this mapping does not know. Unknown stays silent
    /// on purpose: a newer engine's reason must not be labeled as a failure.
    static func resolve(_ message: AppChatMessage) -> MessageStopNotice? {
        guard message.role == .assistant, let reason = message.stopReason else { return nil }
        switch reason {
        // `finishCancelled` and the engine's own cancelled reason share a spelling.
        case "cancelled": return .stoppedByUser
        // "failed" is the subagent runner's spelling of the same outcome.
        case "error", "failed": return .engineError
        case "context_overflow": return .contextFull
        case GenerationResult.StopReason.maxTokens.rawValue: return .lengthLimit
        case "max_turns": return .turnLimit
        default: return nil
        }
    }

    var severity: Severity {
        switch self {
        case .stoppedByUser: return .neutral
        case .lengthLimit, .turnLimit: return .warning
        case .contextFull, .engineError: return .danger
        }
    }

    /// The glyph that carries the meaning alongside color (DESIGN.md: status
    /// is never color alone). Octagon for danger, triangle for warning.
    var symbolName: String {
        switch self {
        case .stoppedByUser: return "stop.circle"
        case .engineError: return "xmark.octagon"
        case .contextFull: return "rectangle.compress.vertical"
        case .lengthLimit, .turnLimit: return "exclamationmark.triangle"
        }
    }

    /// Plain text for the pasteboard: the localized cause plus the raw
    /// stored reason, which is what a bug report needs to find the code path.
    func copyDetails(stopReason: String) -> String {
        [localizedTitle, localizedDetail, "stop_reason=\(stopReason)"].joined(separator: "\n")
    }

    private var localizedTitle: String {
        switch self {
        case .stoppedByUser: return AppStrings.localized("Stopped by you")
        case .engineError: return AppStrings.localized("Stopped because of an error")
        case .contextFull: return AppStrings.localized("Context window full")
        case .lengthLimit: return AppStrings.localized("Reached the length limit")
        case .turnLimit: return AppStrings.localized("Reached the turn limit")
        }
    }

    private var localizedDetail: String {
        switch self {
        case .stoppedByUser:
            return AppStrings.localized("This reply stopped before it finished.")
        case .engineError:
            return AppStrings.localized("The engine failed before this reply finished.")
        case .contextFull:
            return AppStrings.localized(
                "The conversation no longer fits the context window. Run /compact or start a new chat, then retry.")
        case .lengthLimit:
            return AppStrings.localized("This reply used its whole token budget and may be cut off.")
        case .turnLimit:
            return AppStrings.localized("The subagent used all its turns before finishing.")
        }
    }
}

/// The footer strip under an incomplete assistant reply: glyph, cause, one
/// line of what to do, and quiet Retry and Copy details actions. The global
/// `ErrorBanner` stays for errors that belong to no message.
struct MessageStopNoticeView: View {
    let notice: MessageStopNotice
    let stopReason: String
    /// Nil when this row cannot be retried (not the turn's last reply).
    let retryAction: (() -> Void)?
    let actionsDisabled: Bool
    /// The existing `/compact` path, offered only on a context-full notice.
    var compactAction: (() -> Void)? = nil
    /// Why Compact is refused right now, shown as its tooltip.
    var compactBlocker: ContextMeterActions.Blocker? = nil

    private var tone: AnyShapeStyle {
        switch notice.severity {
        case .neutral: return AnyShapeStyle(.appSecondary)
        case .warning: return AnyShapeStyle(.appWarning)
        case .danger: return AnyShapeStyle(.appDanger)
        }
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: notice.symbolName)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(tone)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    title
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(tone)
                    detail
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 8)
            if notice == .contextFull, let compactAction {
                Button(action: compactAction) {
                    Label { Text("Compact now", bundle: .module) } icon: { Image(systemName: "rectangle.compress.vertical") }
                }
                .appAction(.quiet, size: .small)
                .disabled(compactBlocker != nil)
                .help(compactBlocker?.helpText
                    ?? Text("Summarize older messages now and keep recent turns verbatim", bundle: .module))
            }
            if let retryAction {
                Button(action: retryAction) {
                    Label { Text("Retry", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                .appAction(.quiet, size: .small)
                .disabled(actionsDisabled)
                .help(Text("Regenerate response", bundle: .module))
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(notice.copyDetails(stopReason: stopReason), forType: .string)
            } label: {
                Label { Text("Copy details", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
            }
            .appAction(.quiet, size: .small)
            .help(Text("Copy why this reply stopped", bundle: .module))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("message-stop-notice")
    }

    @ViewBuilder
    private var title: some View {
        switch notice {
        case .stoppedByUser: Text("Stopped by you", bundle: .module)
        case .engineError: Text("Stopped because of an error", bundle: .module)
        case .contextFull: Text("Context window full", bundle: .module)
        case .lengthLimit: Text("Reached the length limit", bundle: .module)
        case .turnLimit: Text("Reached the turn limit", bundle: .module)
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch notice {
        case .stoppedByUser:
            Text("This reply stopped before it finished.", bundle: .module)
        case .engineError:
            Text("The engine failed before this reply finished.", bundle: .module)
        case .contextFull:
            Text(
                "The conversation no longer fits the context window. Run /compact or start a new chat, then retry.",
                bundle: .module)
        case .lengthLimit:
            Text("This reply used its whole token budget and may be cut off.", bundle: .module)
        case .turnLimit:
            Text("The subagent used all its turns before finishing.", bundle: .module)
        }
    }
}
