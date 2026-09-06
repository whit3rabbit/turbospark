import Foundation

/// Directory and file names every filesystem walk in the app skips.
///
/// ONE table so the attachment importer's folder expansion and the `@`
/// mention index cannot restate (and then drift from) each other. Both walk
/// the same trees with different purposes -- one collects attachable files,
/// one lists everything referable -- and the skip list is the part that
/// means the same thing to both.
enum FileSystemScanRules {
    /// Directories never descended into.
    static let skipDirectoryNames: Set<String> = [
        ".git", ".svn", ".hg", "node_modules", "target", ".build",
        ".next", "dist", "build", "venv", ".venv", "env",
        "Pods", "Carthage", "DerivedData", ".DS_Store", "__pycache__",
    ]

    /// Individual files never offered or attached.
    static let skipFileNames: Set<String> = [".DS_Store"]
}
