import Foundation
import TurboSpark

/// Filtering and sorting for the Model Hub list.
///
/// Pure and free of SwiftUI so it can be unit tested: the previous version
/// lived inline in the view and could only be checked by looking at the app.
///
/// EVERY OPTION HERE IS DERIVED FROM DATA THE CARD ALSO DISPLAYS. The version
/// this replaces matched alias substrings, which drifts silently and was
/// already wrong: `.conversational` fell through to `break` and returned the
/// whole catalog, `museGlimmer` was listed under both "Reasoning" and "Dense",
/// and `gemma` was listed under "Coding" and "MoE" by hand. Format and
/// capability now come from `ModelFamilyVisuals`, the same table the badges
/// read, and fit comes from `ModelRecommendation.verdict`, which is measured.
struct ModelHubFilter: Equatable {
    /// Which half of the catalog to show.
    enum Tab: String, CaseIterable, Identifiable {
        case recommended = "Recommended"
        case discover = "Discover"
        case onDevice = "On Device"
        var id: String { rawValue }
    }

    /// Ordering of the resulting list.
    enum SortOption: String, CaseIterable, Identifiable {
        case recommended = "Best fit"
        case name = "Name"
        case size = "Size"
        var id: String { rawValue }
    }

    /// A dropdown value that is either "no filter" or one concrete choice.
    ///
    /// Modelled as a string rather than an enum because the concrete choices
    /// are read off the catalog at runtime; a fixed enum is exactly what went
    /// stale before.
    typealias Selection = String?

    var tab: Tab = .recommended
    var searchText: String = ""
    var format: Selection = nil
    var capability: Selection = nil
    var fit: Selection = nil
    var sort: SortOption = .recommended

    /// Whether anything other than the tab and sort is narrowing the list.
    var isNarrowed: Bool {
        !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || format != nil
            || capability != nil
            || fit != nil
    }

    /// Clears every narrowing filter, leaving the tab and sort alone.
    mutating func clearNarrowing() {
        searchText = ""
        format = nil
        capability = nil
        fit = nil
    }

    // MARK: - Option lists

    /// Format labels present in the catalog, in display order.
    static func formatOptions(for entries: [CatalogEntry]) -> [String] {
        sortedUnique(entries.map { visuals(for: $0).formatLabel })
    }

    /// Capability tags present in the catalog, in display order.
    static func capabilityOptions(for entries: [CatalogEntry]) -> [String] {
        sortedUnique(entries.flatMap { visuals(for: $0).capabilities })
    }

    /// Fit labels present, given whatever recommendations have loaded.
    ///
    /// Driven by what is actually in the map rather than by the full verdict
    /// enum: offering "Too large" on a machine where no row is refused is a
    /// filter that can only ever return nothing.
    static func fitOptions(
        for entries: [CatalogEntry],
        recommendations: [String: ModelRecommendation]
    ) -> [String] {
        let labels = entries.compactMap { recommendations[$0.alias].map { fitLabel($0.verdict) } }
        let present = Set(labels)
        return orderedFitLabels.filter { present.contains($0) }
    }

    /// Display label for a fit verdict, as this filter's menu groups by.
    /// The mapping lives in `ModelFitPresentation`; this is a thin forward so
    /// the badge and the filter cannot disagree about what a verdict is
    /// called (swift Gotcha 22).
    static func fitLabel(_ verdict: ModelRecommendation.FitVerdict) -> String {
        ModelFitPresentation.filterLabel(verdict)
    }

    private static let orderedFitLabels = [
        "Fits in memory", "Streams", "Tight fit", "Unknown", "Too large",
    ]

    private static func sortedUnique(_ values: [String]) -> [String] {
        Array(Set(values)).sorted()
    }

    private static func visuals(for entry: CatalogEntry) -> ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(alias: entry.alias, family: entry.family, name: entry.name)
    }

    // MARK: - Application

    /// Applies the tab, search, filters and sort to a catalog.
    func apply(
        to catalog: [CatalogEntry],
        installedAliases: Set<String>,
        recommendations: [String: ModelRecommendation]
    ) -> [CatalogEntry] {
        var list = catalog

        if tab == .onDevice {
            list = list.filter { installedAliases.contains($0.alias) || $0.installed }
        } else if tab == .recommended {
            list = list.filter { entry in
                guard let rec = recommendations[entry.alias] else { return false }
                return rec.runs && rec.verdict != .refused
            }
        }

        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !query.isEmpty {
            list = list.filter { entry in
                entry.alias.lowercased().contains(query)
                    || entry.name.lowercased().contains(query)
                    || entry.family.lowercased().contains(query)
                    || (entry.notes?.lowercased().contains(query) ?? false)
            }
        }

        if let format {
            list = list.filter { Self.visuals(for: $0).formatLabel == format }
        }

        if let capability {
            list = list.filter { Self.visuals(for: $0).capabilities.contains(capability) }
        }

        if let fit {
            list = list.filter { entry in
                guard let verdict = recommendations[entry.alias]?.verdict else { return false }
                return Self.fitLabel(verdict) == fit
            }
        }

        switch sort {
        case .recommended:
            list.sort { lhs, rhs in
                let lhsRec = recommendations[lhs.alias]
                let rhsRec = recommendations[rhs.alias]

                if tab == .recommended {
                    let lhsP = Self.recommendedPriority(entry: lhs, recommendation: lhsRec)
                    let rhsP = Self.recommendedPriority(entry: rhs, recommendation: rhsRec)
                    if lhsP.tier != rhsP.tier { return lhsP.tier < rhsP.tier }
                    if lhsP.rank != rhsP.rank { return lhsP.rank < rhsP.rank }
                    return lhsP.alias < rhsP.alias
                }

                let lhsRank = Self.verdictRank(lhsRec?.verdict)
                let rhsRank = Self.verdictRank(rhsRec?.verdict)
                if lhsRank != rhsRank { return lhsRank < rhsRank }
                return lhs.alias < rhs.alias
            }
        case .name:
            list.sort { $0.alias.localizedCaseInsensitiveCompare($1.alias) == .orderedAscending }
        case .size:
            list.sort { $0.downloadBytes < $1.downloadBytes }
        }

        return list
    }

    /// Priorities for the Recommended tab: optimal MLX, optimal MoE, optimal dense, then tight fits.
    static func recommendedPriority(
        entry: CatalogEntry,
        recommendation: ModelRecommendation?
    ) -> (tier: Int, rank: Int, alias: String) {
        let rank = verdictRank(recommendation?.verdict)
        let isOptimal = rank <= 1
        let visuals = visuals(for: entry)
        let isMlx = visuals.formatLabel.localizedCaseInsensitiveContains("mlx")
        let isMoe = visuals.capabilities.contains("MoE")
            || recommendation?.verdict == .streams
            || entry.name.localizedCaseInsensitiveContains("moe")
            || entry.family.localizedCaseInsensitiveContains("moe")

        let tier: Int
        if isOptimal && isMlx {
            tier = 0
        } else if isOptimal && isMoe {
            tier = 1
        } else if isOptimal {
            tier = 2
        } else if isMlx || isMoe {
            tier = 3
        } else {
            tier = 4
        }

        return (tier, rank, entry.alias)
    }

    /// Identifies models that use Apple Silicon native MLX formats or MoE architectures.
    static func isMlxOrMoe(
        entry: CatalogEntry,
        recommendation: ModelRecommendation? = nil
    ) -> Bool {
        let visuals = visuals(for: entry)
        return visuals.formatLabel.localizedCaseInsensitiveContains("mlx")
            || visuals.capabilities.contains("MoE")
            || recommendation?.verdict == .streams
            || entry.name.localizedCaseInsensitiveContains("moe")
            || entry.family.localizedCaseInsensitiveContains("moe")
    }

    /// Sort order for the "Best fit" option: what runs best comes first.
    static func verdictRank(_ verdict: ModelRecommendation.FitVerdict?) -> Int {
        switch verdict {
        case .resident: return 0
        case .streams: return 1
        case .tight: return 2
        case .unknown, .none: return 3
        case .refused: return 4
        }
    }
}
