import Foundation

/// Pure text summaries for the approval card of tools whose payload the
/// generic previews used to hide (multiedit's `edits`, http_request's method,
/// body, headers and auth). A card the user approves must show what will run.
enum ToolApprovalPreviewSummary {
    /// URL spellings http_request accepts, mirroring the executor's lookup
    /// (`endpoint` included) so the shown URL is the fetched URL.
    static let httpURLKeys = ["url", "uri", "Url", "URL", "href", "endpoint"]

    static func httpURL(_ arguments: [String: String]) -> String? {
        for key in httpURLKeys {
            if let value = arguments[key], !value.isEmpty { return value }
        }
        return nil
    }

    private static func clip(_ text: String, _ limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + "\n... (\(text.count - limit) more characters)"
    }

    /// One block per edit: path, replace_all flag, removed and added text.
    /// Returns nil when no edits parse so the caller falls back to the raw
    /// arguments instead of rendering an empty card.
    static func multiEditSummary(_ arguments: [String: String]) -> String? {
        guard let edits = try? MultiEditExecutor.parseEdits(from: arguments), !edits.isEmpty else {
            return nil
        }
        let files = Set(edits.map(\.filePath)).count
        var lines = ["\(edits.count) edit(s) across \(files) file(s), applied atomically"]
        for (index, edit) in edits.enumerated() {
            lines.append("")
            lines.append("[\(index + 1)] \(edit.filePath)\(edit.replaceAll ? " (replace all)" : "")")
            for line in clip(edit.oldString, 400).split(separator: "\n", omittingEmptySubsequences: false) {
                lines.append("- \(line)")
            }
            for line in clip(edit.newString, 400).split(separator: "\n", omittingEmptySubsequences: false) {
                lines.append("+ \(line)")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static let secretHeaderFragments = ["authorization", "cookie", "token", "key", "secret"]

    /// Method, URL, header names (secret values masked), auth indicator and
    /// the body. The auth token itself is never printed.
    static func httpRequestSummary(_ arguments: [String: String]) -> String {
        let method = (arguments["method"] ?? "GET").uppercased()
        var lines = ["\(method) \(httpURL(arguments) ?? "(no URL)")"]
        if let raw = arguments["headers"] {
            if let data = raw.data(using: .utf8),
               let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
                for name in dict.keys.sorted() {
                    let secret = secretHeaderFragments.contains { name.lowercased().contains($0) }
                    lines.append("Header: \(name): \(secret ? "[hidden]" : clip(dict[name] ?? "", 200))")
                }
            } else {
                lines.append("Headers: (unparseable) \(clip(raw, 200))")
            }
        }
        let authType = arguments["auth_type"] ?? arguments["authType"]
        let hasToken = (arguments["auth_token"] ?? arguments["authToken"] ?? arguments["token"]) != nil
        if hasToken || authType != nil {
            lines.append("Auth: \(authType ?? "token") [token hidden]")
        }
        if let body = arguments["body"] ?? arguments["data"] ?? arguments["payload"], !body.isEmpty {
            lines.append("")
            lines.append("Body:")
            lines.append(clip(body, 2000))
        }
        return lines.joined(separator: "\n")
    }
}
