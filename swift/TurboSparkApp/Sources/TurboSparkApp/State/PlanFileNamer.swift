import Foundation

/// Names the markdown file a finalized plan is written to.
///
/// Claude Code's own plan files are named this way (a topic slug plus a
/// two-word tag, `...-starry-melody.md`), and the reason it is worth copying
/// is that plans arrive in bursts: a timestamp names them all alike at a
/// glance, and a counter needs state nothing here keeps. Two words out of a
/// pair of lists give 1,024 combinations, which is not a uniqueness
/// guarantee and is not asked to be one -- [`uniqueFileName`] checks the
/// directory, and the check is what makes the name safe rather than the
/// arithmetic.
///
/// Pure, with an injectable generator, so the tests need no directory and no
/// clock.
public enum PlanFileNamer {
    public static let adjectives: [String] = [
        "amber", "brisk", "calm", "coral", "crisp", "dusky", "eager", "fable",
        "gentle", "golden", "hidden", "ivory", "jolly", "keen", "lucid", "misty",
        "noble", "opal", "patient", "quiet", "rapid", "starry", "tidal", "umber",
        "vivid", "warm", "wild", "yonder", "zephyr", "azure", "bright", "clever",
    ]

    public static let nouns: [String] = [
        "anchor", "beacon", "canyon", "delta", "ember", "forest", "glacier", "harbor",
        "island", "jetty", "kernel", "lantern", "melody", "nebula", "orchard", "prairie",
        "quarry", "ridge", "summit", "thicket", "upland", "valley", "willow", "xenon",
        "yarrow", "zenith", "basin", "cavern", "dunes", "estuary", "fjord", "grove",
    ]

    /// The two-word tag alone, for example `starry-melody`.
    public static func tag<G: RandomNumberGenerator>(using generator: inout G) -> String {
        let adjective = adjectives[Int(generator.next(upperBound: UInt64(adjectives.count)))]
        let noun = nouns[Int(generator.next(upperBound: UInt64(nouns.count)))]
        return "\(adjective)-\(noun)"
    }

    /// A file-system-safe slug of a plan's own title, or nil when the title
    /// carries nothing usable.
    ///
    /// Bounded at 48 characters because the tag and the extension follow it
    /// and a model-authored title has no length limit at all.
    public static func slug(_ title: String?) -> String? {
        guard let title else { return nil }
        let allowed = CharacterSet.alphanumerics
        let scalars = title.lowercased().unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        let collapsed = String(scalars)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        let trimmed = String(collapsed.prefix(48))
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? nil : trimmed
    }

    /// `<slug>-<tag>.md`, or `plan-<tag>.md` when there is no usable title.
    public static func fileName<G: RandomNumberGenerator>(
        title: String? = nil,
        using generator: inout G
    ) -> String {
        "\(slug(title) ?? "plan")-\(tag(using: &generator)).md"
    }

    /// The same, retried until the name is free in `directory`.
    ///
    /// Gives up after `attempts` tries and appends a UUID fragment rather
    /// than looping forever or returning a name that would overwrite a plan
    /// the user still has open. Overwriting is the one outcome worth this
    /// much care: the file is the artifact, and a silent overwrite loses a
    /// plan with nothing reporting it.
    public static func uniqueFileName<G: RandomNumberGenerator>(
        in directory: URL,
        title: String? = nil,
        attempts: Int = 8,
        using generator: inout G
    ) -> String {
        for _ in 0 ..< max(1, attempts) {
            let candidate = fileName(title: title, using: &generator)
            let url = directory.appendingPathComponent(candidate)
            if !FileManager.default.fileExists(atPath: url.path) { return candidate }
        }
        let fallback = UUID().uuidString.prefix(8).lowercased()
        return "\(slug(title) ?? "plan")-\(fallback).md"
    }
}
