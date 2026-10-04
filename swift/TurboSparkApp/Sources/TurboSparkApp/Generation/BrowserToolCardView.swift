import Foundation
import SwiftUI

struct BrowserToolCardPresentation: Equatable, Sendable {
    enum Outcome: String, Sendable {
        case running = "Running..."
        case completed = "Completed"
        case denied = "Denied"
        case failed = "Failed"
        case pendingApproval = "Needs Approval"
    }

    let action: String
    let target: String?
    let outcome: Outcome
    let durationText: String?

    init(call: AppToolCall, result: AppToolResult?, pendingOrigin: BrowserOrigin? = nil) {
        action = ToolPresentation.resolve(call.name).localizedLabel
        target = result?.browserCardMetadata?.displayTarget
            ?? Self.navigationOrigin(call: call)
            ?? pendingOrigin?.canonicalString

        switch ToolPresentation.status(call: call, result: result) {
        case .running: outcome = .running
        case .completed: outcome = .completed
        case .denied: outcome = .denied
        case .failed: outcome = .failed
        case .pendingApproval: outcome = .pendingApproval
        }

        if let seconds = result?.durationSeconds, seconds.isFinite, seconds >= 0 {
            durationText = String(format: "%.2f s", locale: Locale(identifier: "en_US_POSIX"), seconds)
        } else {
            durationText = nil
        }
    }

    static func isBrowserTool(_ name: String) -> Bool {
        BrowserToolDefinitions.commandKind(for: name.lowercased()) != nil
    }

    var summary: ToolCallSummaryInfo {
        ToolCallSummaryInfo(action: action, target: target ?? "")
    }

    private static func navigationOrigin(call: AppToolCall) -> String? {
        guard call.name.lowercased() == "browser_navigate",
              let rawURL = call.arguments["url"], let url = URL(string: rawURL)
        else { return nil }
        return BrowserOrigin(url: url)?.canonicalString
    }
}

enum BrowserToolCardMetadataBuilder {
    static func make(
        value: BrowserControlValue,
        fallbackOrigin: BrowserOrigin?
    ) -> AppToolBrowserCardMetadata {
        let resultOrigin: BrowserOrigin?
        switch value {
        case .navigated(let navigation):
            resultOrigin = URL(string: navigation.url).flatMap { BrowserOrigin(url: $0) }
        case .state(let snapshot):
            resultOrigin = snapshot.url.flatMap { rawURL in
                URL(string: rawURL).flatMap { BrowserOrigin(url: $0) }
            }
        default:
            resultOrigin = nil
        }

        let elementReference: String?
        switch value {
        case .clicked(let action), .typed(let action):
            elementReference = action.reference
        case .keyPressed(let action):
            elementReference = action.reference
        case .scrolled(let action):
            elementReference = action.reference
        default:
            elementReference = nil
        }

        return AppToolBrowserCardMetadata(
            origin: resultOrigin ?? fallbackOrigin,
            elementReference: elementReference)
    }
}

/// Renders only the browser duration. The parent card supplies status and the
/// safe target; raw arguments and output never enter this view.
struct BrowserToolCardView: View {
    @Environment(\.appTheme) private var theme
    let presentation: BrowserToolCardPresentation

    var body: some View {
        Group {
            if let durationText = presentation.durationText {
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .themedFont(.micro)
                        .foregroundStyle(.appSecondary)
                    Text(verbatim: durationText)
                        .font(theme.code(.tiny))
                        .foregroundStyle(.appSecondary)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}
