import Foundation

/// Which project-relative paths a code search may return.
///
/// **THE INDEXED AND UNINDEXED SEARCHES MUST HIDE THE SAME FILES.** Syntext
/// indexes hidden files and `.git/` (its walker leaves exclusions to
/// `.gitignore`, which does not apply outside a git repository), while the
/// unindexed scan skips hidden entries and vendor directories. Search is
/// auto-allowed, so a credential file reachable through only one of the two
/// paths was readable with no prompt depending on whether the project was
/// opted into indexing.
enum GrepPathPolicy {
    /// Directory names a search never descends into, at any depth.
    static let excludedDirectoryNames: Set<String> = [".git", "node_modules", "target", ".build"]

    /// Whether `relativePath` may appear in search output.
    ///
    /// `scope` is the directory prefix (`src/`) or file the caller asked for,
    /// if any. Hidden components are checked only BELOW it, matching the
    /// unindexed scan, which starts its walk at the requested path and skips
    /// hidden entries under it; an explicit `.github/` search still works.
    static func allows(_ relativePath: String, scope: String? = nil) -> Bool {
        let components = relativePath.split(separator: "/").map(String.init)
        guard !components.isEmpty else { return false }
        if components.dropLast().contains(where: excludedDirectoryNames.contains) { return false }
        if components.last == ".git" { return false }

        let scopeDepth = scope.map { $0.split(separator: "/").count } ?? 0
        let isFileScope = scope.map { !$0.hasSuffix("/") } ?? false
        let checked: ArraySlice<String> = isFileScope ? [] : components.dropFirst(scopeDepth)
        if checked.contains(where: { $0.hasPrefix(".") }) { return false }

        // Both spellings: the classifier's directory rules (`/.ssh/`) need a
        // leading slash that a project-relative path does not have.
        return !ToolRiskClassifier.isSensitivePath(relativePath)
            && !ToolRiskClassifier.isSensitivePath("/" + relativePath)
    }

    /// Whether `relativePath` lies inside `scope`: under a directory prefix
    /// (`src/`), or equal to a file scope. A nil scope is the whole project.
    static func isInside(_ relativePath: String, scope: String?) -> Bool {
        guard let scope else { return true }
        return scope.hasSuffix("/") ? relativePath.hasPrefix(scope) : relativePath == scope
    }
}
