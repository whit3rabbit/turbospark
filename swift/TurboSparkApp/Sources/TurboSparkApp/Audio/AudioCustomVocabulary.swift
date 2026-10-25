import Foundation
import TurboSpark

/// A custom vocabulary term with canonical spelling and optional phonetic misrecognitions.
///
/// Phase 1:
/// Enables custom vocabulary and jargon replacement to ensure domain-specific acronyms,
/// project names, and technical terms are accurately spelled in speech-to-text transcripts.
public struct AudioVocabularyTerm: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var term: String
    public var aliases: [String]
    public var category: String?
    public var createdAt: Date

    public init(
        id: String = UUID().uuidString,
        term: String,
        aliases: [String] = [],
        category: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.term = term
        self.aliases = aliases
        self.category = category
        self.createdAt = createdAt
    }
}

/// Applies custom vocabulary replacements and prompt biasing to transcripts.
public enum AudioCustomVocabulary {

    /// Normalizes text by replacing misrecognized phonetic variants or lowercase
    /// spellings with the user's canonical vocabulary terms.
    public static func apply(_ terms: [AudioVocabularyTerm], to text: String) -> String {
        guard !terms.isEmpty, !text.isEmpty else { return text }
        var result = text

        for item in terms {
            let canonical = item.term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !canonical.isEmpty else { continue }

            // Build list of targets to replace: aliases plus case-insensitive version of canonical
            var targets = item.aliases
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty && $0.caseInsensitiveCompare(canonical) != .orderedSame }

            // Also replace case-insensitive variations of canonical if it has uppercase letters
            if canonical.lowercased() != canonical {
                targets.append(canonical.lowercased())
            }

            // Sort targets by length descending to match longest phrases first (e.g. "turbo spark" before "turbo")
            targets.sort { $0.count > $1.count }

            for target in targets {
                // Escape special regex characters
                let escaped = NSRegularExpression.escapedPattern(for: target)
                // Match with word boundaries, handling multi-word spaces cleanly
                let pattern = "\\b" + escaped.replacingOccurrences(of: "\\ ", with: "\\s+") + "\\b"

                if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                    let range = NSRange(result.startIndex..<result.endIndex, in: result)
                    result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: canonical)
                }
            }
        }

        return result
    }

    /// Normalizes segments by applying custom vocabulary to each segment's text.
    static func apply(_ terms: [AudioVocabularyTerm], to segments: [AudioLibrarySegment]) -> [AudioLibrarySegment] {
        guard !terms.isEmpty else { return segments }
        return segments.map { segment in
            var updated = segment
            updated.text = apply(terms, to: segment.text)
            return updated
        }
    }

    /// Composes a concise glossary string for models that accept prompt biasing or prefix context.
    public static func promptContext(terms: [AudioVocabularyTerm], limit: Int = 30) -> String {
        let uniqueTerms = Array(terms.prefix(limit)).map(\.term)
        guard !uniqueTerms.isEmpty else { return "" }
        return uniqueTerms.joined(separator: ", ")
    }
}

/// Persistent storage for custom vocabulary terms under `~/Library/Application Support/TurboSpark/custom_vocabulary.json`.
public enum AudioCustomVocabularyStore {
    private static var fileURL: URL {
        AppStorageRoot.directory.appendingPathComponent("custom_vocabulary.json")
    }

    /// Default seeded vocabulary terms for the project and common Apple Silicon / AI tech.
    public static let defaultTerms: [AudioVocabularyTerm] = [
        AudioVocabularyTerm(term: "TurboSpark", aliases: ["turbo spark", "turbospark"], category: "Project"),
        AudioVocabularyTerm(term: "CoreAudio", aliases: ["core audio"], category: "Tech"),
        AudioVocabularyTerm(term: "CoreML", aliases: ["core ml", "core-ml"], category: "Tech"),
        AudioVocabularyTerm(term: "Metal", aliases: ["metal"], category: "Tech"),
        AudioVocabularyTerm(term: "FluidAudio", aliases: ["fluid audio"], category: "Project"),
        AudioVocabularyTerm(term: "GGUF", aliases: ["g guf", "g-guf"], category: "Tech"),
        AudioVocabularyTerm(term: "MLX", aliases: ["m l x", "m-l-x"], category: "Tech"),
        AudioVocabularyTerm(term: "Qwen", aliases: ["q wen", "kewen"], category: "Model"),
    ]

    public static func load() -> [AudioVocabularyTerm] {
        if let stored = AppJSONStore.load([AudioVocabularyTerm].self, from: fileURL, label: "Custom Vocabulary"), !stored.isEmpty {
            return stored
        }
        // First load: seed with default terms
        save(defaultTerms)
        return defaultTerms
    }

    public static func save(_ terms: [AudioVocabularyTerm]) {
        AppJSONStore.save(terms, to: fileURL, label: "Custom Vocabulary")
    }

    @discardableResult
    public static func add(term: String, aliases: [String] = [], category: String? = nil) -> AudioVocabularyTerm {
        var terms = load()
        let trimmedTerm = term.trimmingCharacters(in: .whitespacesAndNewlines)
        if let index = terms.firstIndex(where: { $0.term.caseInsensitiveCompare(trimmedTerm) == .orderedSame }) {
            // Merge aliases into existing term
            var existing = terms[index]
            var set = Set(existing.aliases)
            for a in aliases {
                let trimmedA = a.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedA.isEmpty { set.insert(trimmedA) }
            }
            existing.aliases = Array(set)
            if let category { existing.category = category }
            terms[index] = existing
            save(terms)
            return existing
        } else {
            let newTerm = AudioVocabularyTerm(
                term: trimmedTerm,
                aliases: aliases.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty },
                category: category
            )
            terms.append(newTerm)
            save(terms)
            return newTerm
        }
    }

    public static func delete(id: String) {
        var terms = load()
        terms.removeAll { $0.id == id }
        save(terms)
    }
}
