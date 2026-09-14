import Foundation

/// Executor for specialized code and developer documentation searching.
public enum CodeSearchExecutor {
    /// High-reputation developer documentation domains prioritized for code queries.
    public static let developerDomains: [String] = [
        "docs.rs", "crates.io", "developer.apple.com", "swift.org", "github.com",
        "developer.mozilla.org", "pkg.go.dev", "pypi.org", "docs.python.org",
        "react.dev", "nextjs.org", "typescriptlang.org", "nodejs.org", "bun.sh"
    ]

    public static func execute(arguments: [String: String]) async throws -> String {
        let rawQuery = arguments["query"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !rawQuery.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 50,
                userInfo: [NSLocalizedDescriptionKey: "Missing required 'query' parameter for codesearch."]
            )
        }

        let framework = arguments["framework"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let provider = arguments["provider"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let tokensNum = Int(arguments["tokens_num"] ?? arguments["tokensNum"] ?? "5000") ?? 5000

        // Build augmented technical query
        var effectiveQuery = rawQuery
        if let fw = framework, !fw.isEmpty, !rawQuery.lowercased().contains(fw.lowercased()) {
            effectiveQuery = "\(fw) \(rawQuery)"
        }
        if !effectiveQuery.lowercased().contains("doc") && !effectiveQuery.lowercased().contains("api") {
            effectiveQuery += " documentation reference"
        }

        let requestedResults = min(10, max(3, tokensNum / 1000))

        let searchOutput = try await WebSearchExecutor.search(
            query: effectiveQuery,
            numResults: requestedResults,
            allowedDomains: nil,
            blockedDomains: nil,
            provider: provider ?? "auto"
        )

        if searchOutput.results.isEmpty {
            return "No technical documentation results found for query: `\(rawQuery)`."
        }

        var outputSections: [String] = []
        outputSections.append("### Code Search Results for `\(rawQuery)`\n")

        for (idx, item) in searchOutput.results.enumerated() {
            let snippetText = item.snippet ?? "(No snippet available)"
            outputSections.append("#### [\(idx + 1)] \(item.title)\nURL: \(item.url)\n\n\(snippetText)\n")
        }

        return outputSections.joined(separator: "\n")
    }
}
