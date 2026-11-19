import Foundation

/// Starting points for the rules lists. They exist because the model's own
/// instruction cannot change (the checkpoint was trained on one exact text and
/// the engine refuses any other), so style is steered outside the model, with
/// words to avoid and replacements that run on the finished rewrite.
///
/// A preset only ADDS lines. It never removes or reorders what the user typed,
/// and applying it twice changes nothing, so trying one is always safe.
/// The visible names live at the call site (`HumanizeAdvancedPanel`) so the
/// localization gates can see them; the term lists are data, shown verbatim.
enum HumanizePreset: CaseIterable, Sendable {
    case plainEnglish
    case business
    case casual

    var avoidTerms: [String] {
        switch self {
        case .plainEnglish:
            ["delve", "tapestry", "testament", "realm", "robust", "seamless", "leverage", "game-changer"]
        case .business:
            ["synergy", "circle back", "touch base", "low-hanging fruit", "move the needle", "going forward"]
        case .casual:
            ["nevertheless", "hence", "thus", "whilst", "henceforth"]
        }
    }

    /// `old -> new` pairs. Matching is exact-case, so a sentence-initial form
    /// is its own line.
    var replacements: [String] {
        switch self {
        case .plainEnglish:
            [
                "utilize -> use", "Utilize -> Use",
                "in order to -> to", "In order to -> To",
                "a number of -> several", "A number of -> Several",
                "due to the fact that -> because", "Due to the fact that -> Because",
            ]
        case .business:
            [
                "at this point in time -> now", "At this point in time -> Now",
                "with regard to -> about", "With regard to -> About",
                "prior to -> before", "Prior to -> Before",
                "in the event that -> if", "In the event that -> If",
            ]
        case .casual:
            [
                "Furthermore, -> Also,", "Moreover, -> Also,", "Additionally, -> Also,",
                "Consequently, -> So,", "Therefore, -> So,", "However, -> But",
            ]
        }
    }
}

extension HumanizeRules {
    /// This rule set with the preset's lines added after the user's own.
    /// Protected terms, the seed and every line already present are untouched.
    func adding(_ preset: HumanizePreset) -> HumanizeRules {
        var next = self
        next.avoidTerms = Self.appending(preset.avoidTerms, to: avoidTerms)
        next.replacements = Self.appending(preset.replacements, to: replacements)
        return next
    }

    /// True when every line of the preset is already in the lists.
    func contains(_ preset: HumanizePreset) -> Bool {
        let avoid = Set(Self.lines(avoidTerms)), replace = Set(Self.lines(replacements))
        return preset.avoidTerms.allSatisfy(avoid.contains) && preset.replacements.allSatisfy(replace.contains)
    }

    private static func appending(_ additions: [String], to text: String) -> String {
        let existing = Set(lines(text))
        let fresh = additions.filter { !existing.contains($0) }
        guard !fresh.isEmpty else { return text }
        let base = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return ([base].filter { !$0.isEmpty } + fresh).joined(separator: "\n")
    }
}
