import Foundation

/// What the approval sheet shows for one untrusted project tool.
///
/// `id` is the trust fingerprint of the definition AS REVIEWED. Approval
/// re-checks it against the file on disk, so a definition rewritten between
/// "sheet shown" and "Approve clicked" is never trusted unseen.
public struct ProjectToolReview: Identifiable, Equatable, Sendable {
    public let id: String
    public let name: String
    public let declaredCategory: AppToolCategory
    public let effectiveCategory: AppToolCategory
    public let executionKind: CustomToolExecutionType
    /// The complete command, script, or request, never truncated.
    public let fullText: String
    public let sourcePath: String?

    /// True when the file asked for a weaker category than the tool really
    /// needs, which is worth a visible warning.
    public var categoryWasRaised: Bool { declaredCategory != effectiveCategory }
}

public enum ProjectToolApprovalOutcome: Equatable, Sendable {
    case approved
    /// The definition on disk no longer matches what the user reviewed (or
    /// the tool is gone). Nothing was trusted; the list should be refreshed.
    case changedSinceReview
}

/// Decision logic for the project custom-tool approval sheet, kept apart
/// from SwiftUI so it can be tested. Reject is deliberately not a stored
/// decision: an untrusted tool simply stays off.
public enum ProjectToolApproval {
    public static func review(of tool: CustomToolDefinition) -> ProjectToolReview {
        ProjectToolReview(
            id: CustomToolTrustStore.fingerprint(of: tool),
            name: tool.name,
            declaredCategory: tool.category,
            effectiveCategory: tool.effectiveCategory,
            executionKind: tool.execution.type,
            fullText: fullText(of: tool.execution),
            sourcePath: tool.sourcePath)
    }

    /// Everything that decides what the tool runs, as plain text. Arguments,
    /// environment and headers are included because they change behavior just
    /// as much as the command line does.
    static func fullText(of execution: CustomToolExecution) -> String {
        var lines: [String] = []
        switch execution.type {
        case .command:
            lines.append(execution.command ?? "")
        case .script:
            lines.append("#! \(execution.scriptInterpreter ?? "/bin/zsh")")
            lines.append(execution.scriptContent ?? "")
            if let args = execution.arguments, !args.isEmpty {
                lines.append("arguments: " + args.joined(separator: " "))
            }
        case .http:
            lines.append("\(execution.httpMethod ?? "POST") \(execution.httpURL ?? "")")
            for (key, value) in (execution.httpHeaders ?? [:]).sorted(by: { $0.key < $1.key }) {
                lines.append("header \(key): \(value)")
            }
        }
        for (key, value) in (execution.environment ?? [:]).sorted(by: { $0.key < $1.key }) {
            lines.append("env \(key)=\(value)")
        }
        return lines.joined(separator: "\n")
    }

    /// Untrusted tools for the project, minus those the user already
    /// rejected this session (`rejected` holds review ids).
    public static func pending(
        for projectURL: URL,
        rejected: Set<String> = [],
        manager: CustomToolManager = .shared
    ) -> [ProjectToolReview] {
        manager.untrustedProjectTools(for: projectURL)
            .map(review(of:))
            .filter { !rejected.contains($0.id) }
    }

    /// Trusts `review` only if the on-disk definition is still the one that
    /// was shown.
    @discardableResult
    public static func approve(
        _ review: ProjectToolReview,
        projectURL: URL,
        manager: CustomToolManager = .shared,
        store: CustomToolTrustStore = .shared
    ) -> ProjectToolApprovalOutcome {
        guard let current = manager.untrustedProjectTools(for: projectURL)
            .first(where: { CustomToolTrustStore.fingerprint(of: $0) == review.id })
        else { return .changedSinceReview }
        store.trust(current)
        return .approved
    }
}
