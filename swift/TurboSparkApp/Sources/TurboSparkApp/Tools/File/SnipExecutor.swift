import Foundation

/// Executor for extracting bounded line snippets from files with line numbering and context.
public enum SnipExecutor {
    public static func execute(arguments: [String: String], rootURL: URL) throws -> String {
        guard let relPath = arguments["path"] ?? arguments["file_path"] ?? arguments["file"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 41,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument for Snip."]
            )
        }

        let targetURL = try AppToolRegistry.resolveSecurePath(relPath: relPath, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 41,
                userInfo: [NSLocalizedDescriptionKey: "File not found at '\(relPath)'."]
            )
        }

        let content = try String(contentsOf: targetURL, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")
        let totalLines = lines.count

        let startLine = max(1, Int(arguments["start_line"] ?? arguments["start"] ?? arguments["offset"] ?? "1") ?? 1)
        let endLine = min(totalLines, Int(arguments["end_line"] ?? arguments["end"] ?? arguments["limit"] ?? "\(totalLines)") ?? totalLines)

        guard startLine <= endLine else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 41,
                userInfo: [NSLocalizedDescriptionKey: "Invalid line range: start_line (\(startLine)) > end_line (\(endLine))."]
            )
        }

        let slice = lines[(startLine - 1)..<endLine]
        var output = "### Snippet: \(relPath) (lines \(startLine)-\(endLine) of \(totalLines))\n```\n"
        for (idx, line) in slice.enumerated() {
            let lineNum = startLine + idx
            output += "\(String(format: "%4d", lineNum)) | \(line)\n"
        }
        output += "```"
        return output
    }
}
