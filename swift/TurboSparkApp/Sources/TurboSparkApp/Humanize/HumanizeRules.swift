import Foundation

/// The user's rules for a rewrite. They work OUTSIDE the model.
///
/// The checkpoint was trained on one exact instruction (fingerprint-checked
/// in core), and the engine refuses every option except an output budget and
/// a seed, so no rule ever reaches the prompt. Replacements edit each part's
/// finished text; protected terms and words to avoid are checked against the
/// whole rewrite once it is done.
struct HumanizeRules: Codable, Equatable, Sendable {
    /// One term per line that must appear in the rewrite exactly as written.
    var protectedTerms = ""
    /// One word or phrase per line, flagged wherever it appears in the rewrite.
    var avoidTerms = ""
    /// One `old -> new` pair per line, applied to the finished rewrite.
    var replacements = ""
    /// Sends `seed`, so the same draft gives the same rewrite.
    var fixedSeed = false
    var seed: UInt64 = 42

    var isEmpty: Bool {
        Self.lines(protectedTerms).isEmpty && Self.lines(avoidTerms).isEmpty
            && replacementPairs.isEmpty && !fixedSeed
    }

    var replacementPairs: [(old: String, new: String)] {
        Self.lines(replacements).compactMap { line in
            guard let arrow = line.range(of: "->") else { return nil }
            let old = line[..<arrow.lowerBound].trimmingCharacters(in: .whitespaces)
            let new = line[arrow.upperBound...].trimmingCharacters(in: .whitespaces)
            return old.isEmpty ? nil : (old, new)
        }
    }

    /// Exact text, whole words, in line order.
    func applyReplacements(to text: String) -> String {
        replacementPairs.reduce(text) { text, pair in
            guard let regex = Self.regex(for: pair.old, caseInsensitive: false) else { return text }
            return regex.stringByReplacingMatches(
                in: text, range: NSRange(location: 0, length: (text as NSString).length),
                withTemplate: NSRegularExpression.escapedTemplate(for: pair.new))
        }
    }

    /// Protected terms match exactly, because a name that changed case did not
    /// survive. A term the draft never contained has nothing to protect.
    /// Words to avoid match regardless of case.
    func check(draft: String, rewrite: String) -> HumanizeRuleFindings {
        var findings = HumanizeRuleFindings()
        for term in Self.lines(protectedTerms)
        where !Self.ranges(of: term, in: draft, caseInsensitive: false).isEmpty
            && Self.ranges(of: term, in: rewrite, caseInsensitive: false).isEmpty {
            findings.missingProtected.append(term)
        }
        for term in Self.lines(avoidTerms) {
            let found = Self.ranges(of: term, in: rewrite, caseInsensitive: true)
            guard !found.isEmpty else { continue }
            findings.avoidFound.append(term)
            findings.avoidRanges += found
        }
        findings.avoidRanges.sort { $0.location < $1.location }
        return findings
    }

    static func lines(_ text: String) -> [String] {
        var seen = Set<String>()
        return text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    static func ranges(of term: String, in text: String, caseInsensitive: Bool) -> [NSRange] {
        guard let regex = regex(for: term, caseInsensitive: caseInsensitive) else { return [] }
        return regex.matches(in: text, range: NSRange(location: 0, length: (text as NSString).length)).map(\.range)
    }

    /// Whole words where the script separates words with spaces. Han and kana
    /// mark no word boundary, so a term containing them matches as written.
    private static func regex(for term: String, caseInsensitive: Bool) -> NSRegularExpression? {
        let escaped = NSRegularExpression.escapedPattern(for: term)
        let unbounded = term.range(of: #"[\p{Han}\p{Hiragana}\p{Katakana}]"#, options: .regularExpression) != nil
        let pattern = unbounded ? escaped : #"(?<![\p{L}\p{N}])"# + escaped + #"(?![\p{L}\p{N}])"#
        return try? NSRegularExpression(pattern: pattern, options: caseInsensitive ? [.caseInsensitive] : [])
    }
}

struct HumanizeRuleFindings: Equatable, Sendable {
    var missingProtected: [String] = []
    var avoidFound: [String] = []
    /// UTF-16 ranges of every avoided word in the rewrite, for highlighting.
    var avoidRanges: [NSRange] = []

    var isEmpty: Bool { missingProtected.isEmpty && avoidFound.isEmpty }
}

/// Rules persist per user. A value that no longer decodes (older, or edited
/// by hand) starts clean rather than failing the section.
struct HumanizeRulesStore {
    static let key = "TurboSpark.humanizeRules"
    let defaults: UserDefaults

    func load() -> HumanizeRules {
        guard let data = defaults.data(forKey: Self.key),
              let rules = try? JSONDecoder().decode(HumanizeRules.self, from: data) else { return HumanizeRules() }
        return rules
    }

    func save(_ rules: HumanizeRules) {
        if let data = try? JSONEncoder().encode(rules) { defaults.set(data, forKey: Self.key) }
    }
}

/// A display copy of core's `HUMANIZER_INSTRUCTION`, shown read-only so the
/// user can see what the model is told. It is never sent: the engine builds
/// the prompt itself. `HumanizeRulesTests` hashes it the way core's
/// fingerprint test does, so this copy cannot drift from the real one.
enum HumanizeInstruction {
    static let text = "Rewrite the text below so it reads like a person wrote it, not a language model.\n\nReorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\nthroat-clearing, and any sentence that only announces what comes next.\nPrefer the concrete word over the abstract one. It is fine to sound uneven.\n\nEvery fact, number, unit, date, name and quotation must survive unchanged."
    static let separator = "\n\n### Rewritten:\n\n"
    /// core's `PROMPT_FINGERPRINT`.
    static let fingerprint = "cc51d66b4c593fbe"
}
