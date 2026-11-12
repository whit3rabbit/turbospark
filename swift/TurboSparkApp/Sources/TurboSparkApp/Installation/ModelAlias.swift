import Foundation

/// Local names for a repository installed from Hugging Face search.
///
/// The install path needs an alias, because it becomes the folder name and the
/// store key. Asking the user to invent one before they can even see whether
/// the model runs was the worst step of the old flow, so the app proposes one
/// and lets them edit it.
///
/// Pure and free of SwiftUI so it can be tested without the app.
enum ModelAlias {
    /// Long enough for `qwen3-coder-30b-a3b-instruct-q4_k_m`, short enough to
    /// read in a list row.
    static let maxLength = 48

    /// A safe, readable alias for `repo` (`owner/name` or `owner/name@rev`),
    /// optionally tagged with the quantization the user picked, and unique
    /// among `taken`.
    ///
    /// The result always satisfies `AppModel.isSafeInstallAlias`: ASCII
    /// letters, digits and `._-`, never `.` or `..`, never empty. Lower case,
    /// because two aliases that differ only by case collide on a
    /// case-insensitive volume.
    static func suggest(repo: String, quantLabel: String? = nil, taken: Set<String>) -> String {
        var name = repo.split(separator: "@", maxSplits: 1).first.map(String.init) ?? repo
        name = name.split(separator: "/").last.map(String.init) ?? name
        name = stripRedundantSuffix(name)

        var base = sanitize(name)
        if let quantLabel {
            let tag = sanitize(quantLabel)
            // Do not repeat what the repository name already says
            // (`...-4bit` with a `4bit` label).
            if !tag.isEmpty, !base.contains(tag) { base += "-" + tag }
        }
        if base.isEmpty { base = "model" }
        base = trimEdges(String(base.prefix(maxLength)))
        if base.isEmpty { base = "model" }

        let lowered = Set(taken.map { $0.lowercased() })
        var candidate = base
        var counter = 2
        while lowered.contains(candidate.lowercased()) {
            candidate = "\(base)-\(counter)"
            counter += 1
        }
        return candidate
    }

    /// Whether `alias` is already used, ignoring case.
    static func isTaken(_ alias: String, in taken: Set<String>) -> Bool {
        let lowered = alias.lowercased()
        return taken.contains { $0.lowercased() == lowered }
    }

    /// `-GGUF` and friends say nothing the file extension does not, and make
    /// every alias for a conversion look alike.
    private static func stripRedundantSuffix(_ name: String) -> String {
        var result = name
        for suffix in ["-gguf", "_gguf", ".gguf"] where result.lowercased().hasSuffix(suffix) {
            result.removeLast(suffix.count)
            break
        }
        return result
    }

    /// Lower-cases, maps every character outside `[a-z0-9._-]` to `-`,
    /// collapses runs of `-`, and trims separators from both ends.
    private static func sanitize(_ input: String) -> String {
        var out = ""
        var lastWasDash = false
        for scalar in input.lowercased().unicodeScalars {
            let ok = scalar.isASCII
                && (CharacterSet.alphanumerics.contains(scalar) || "._".unicodeScalars.contains(scalar))
            if ok {
                out.unicodeScalars.append(scalar)
                lastWasDash = false
            } else if !lastWasDash {
                out.append("-")
                lastWasDash = true
            }
        }
        return trimEdges(out)
    }

    private static func trimEdges(_ input: String) -> String {
        input.trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
    }
}
