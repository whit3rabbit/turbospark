import Foundation

extension AppToolRegistry {
    /// Converts a glob (`**`, `*`, `?`, `[set]`, `{a,b}`) to an anchored regex
    /// over a `/`-separated relative path. A pattern with no `/` matches the
    /// basename at any depth, like ripgrep's `--glob`, so `*.swift` finds files
    /// in subdirectories.
    static func globRegex(_ glob: String) throws -> NSRegularExpression {
        var pattern = glob.hasPrefix("./") ? String(glob.dropFirst(2)) : glob
        let basenameOnly = !pattern.contains("/")
        if basenameOnly { pattern = "**/" + pattern }
        var out = "^"
        var braceDepth = 0
        let chars = Array(pattern)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            switch c {
            case "*":
                if i + 1 < chars.count, chars[i + 1] == "*" {
                    // `**/` matches zero or more whole directories.
                    if i + 2 < chars.count, chars[i + 2] == "/" {
                        out += "(?:.*/)?"
                        i += 3
                        continue
                    }
                    out += ".*"
                    i += 2
                    continue
                }
                out += "[^/]*"
            case "?": out += "[^/]"
            case "[":
                if let close = chars[(i + 1)...].firstIndex(of: "]"), close > i + 1 {
                    var set = String(chars[(i + 1)..<close])
                    if set.hasPrefix("!") { set = "^" + set.dropFirst() }
                    out += "[" + set + "]"
                    i = close
                } else {
                    out += "\\["
                }
            case "{": braceDepth += 1; out += "(?:"
            case "}":
                if braceDepth > 0 { braceDepth -= 1; out += ")" } else { out += "\\}" }
            case ",": out += braceDepth > 0 ? "|" : ","
            default:
                out += NSRegularExpression.escapedPattern(for: String(c))
            }
            i += 1
        }
        // Unbalanced braces would make the regex invalid; close them.
        out += String(repeating: ")", count: braceDepth) + "$"
        return try NSRegularExpression(pattern: out)
    }

    /// Recursive file listing filtered by a glob, under `relPath`. Mirrors
    /// `searchCode`'s containment and skip rules: hidden files, heavy
    /// directories and symlinks are skipped.
    static func globFiles(pattern: String, relPath: String, rootURL: URL, maxResults: Int = 200) throws -> String {
        guard !pattern.split(separator: "/").contains("..") else {
            throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "Glob pattern must not contain '..'; pass `path` instead."
            ])
        }
        let regex: NSRegularExpression
        do {
            regex = try globRegex(pattern)
        } catch {
            throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "Invalid glob pattern '\(pattern)'."
            ])
        }
        let baseURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: baseURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw NSError(domain: "TurboSparkTool", code: 10, userInfo: [NSLocalizedDescriptionKey: "Path is not a directory: \(relPath)"])
        }
        guard let enumerator = FileManager.default.enumerator(
            at: baseURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [NSLocalizedDescriptionKey: "Cannot search path."])
        }
        let basePrefix = baseURL.standardizedFileURL.path.hasSuffix("/")
            ? baseURL.standardizedFileURL.path : baseURL.standardizedFileURL.path + "/"
        let heavy: Set<String> = ["node_modules", "target", ".build", ".git"]
        var matches: [String] = []
        var visited = 0
        var truncated = false
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey])
            if values?.isDirectory == true {
                if heavy.contains(fileURL.lastPathComponent) { enumerator.skipDescendants() }
                continue
            }
            if values?.isSymbolicLink == true || values?.isRegularFile != true { continue }
            visited += 1
            if visited > 50_000 { truncated = true; break }
            let full = fileURL.standardizedFileURL.path
            let rel = full.hasPrefix(basePrefix) ? String(full.dropFirst(basePrefix.count)) : fileURL.lastPathComponent
            let range = NSRange(location: 0, length: (rel as NSString).length)
            if regex.firstMatch(in: rel, options: [], range: range) != nil {
                matches.append(rel)
                if matches.count >= maxResults { truncated = true; break }
            }
        }
        if matches.isEmpty { return "No files match '\(pattern)' under \(relPath)." }
        matches.sort()
        var lines = ["\(matches.count) file(s) match '\(pattern)' under \(relPath):"] + matches
        if truncated { lines.append("... results truncated; narrow the pattern or path.") }
        return lines.joined(separator: "\n")
    }
}
