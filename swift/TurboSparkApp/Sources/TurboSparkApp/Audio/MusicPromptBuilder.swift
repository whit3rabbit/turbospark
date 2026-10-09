import Foundation

/// A labelled group of prompt vocabulary (a genre family, a mood family).
struct MusicPromptGroup: Identifiable, Equatable, Sendable {
    let name: String
    let items: [String]
    var id: String { name }
}

/// Prompt vocabulary for the MiniMax Music style description.
///
/// Every phrase here is English text the music model reads, not UI copy, so it
/// is deliberately not localized. The families follow the model's published
/// prompt library; the lists are a vocabulary map, not a limit on what the
/// model accepts (the description field still takes any text).
enum MusicPromptCatalog {
    static let genres: [MusicPromptGroup] = [
        .init(name: "Pop", items: [
            "pop", "pop ballad", "dance-pop", "synth-pop", "electropop", "dream pop", "indie pop",
            "disco funk pop", "nu-disco"
        ]),
        .init(name: "Electronic", items: [
            "house", "trance", "techno", "dubstep", "hardstyle", "ambient electronic", "darkwave", "synthwave",
            "lo-fi chill"
        ]),
        .init(name: "R&B and soul", items: [
            "contemporary R&B", "neo-soul", "trap soul", "soul", "blues", "gospel"
        ]),
        .init(name: "Hip-hop", items: ["hip-hop", "trap", "drill", "lo-fi hip-hop", "melodic rap"]),
        .init(name: "Rock", items: [
            "pop rock", "indie rock", "alternative rock", "arena rock", "punk", "post-punk", "hard rock",
            "metalcore", "symphonic metal"
        ]),
        .init(name: "Folk and country", items: [
            "indie folk", "folk pop", "singer-songwriter", "country", "Americana", "bluegrass", "Celtic folk",
            "reggae"
        ]),
        .init(name: "Jazz and stage", items: [
            "vocal jazz", "jazz ballad", "lo-fi jazz", "big band swing", "bossa nova", "lounge", "musical theatre",
            "cabaret", "doo-wop"
        ]),
        .init(name: "Cinematic", items: [
            "cinematic orchestral", "epic trailer music", "film score", "cinematic pop ballad",
            "orchestral with modern percussion", "classical"
        ]),
        .init(name: "East Asian", items: [
            "Mandopop", "Cantopop", "J-pop", "K-pop", "J-rock", "city pop", "guofeng pop"
        ])
    ]

    static let moods: [MusicPromptGroup] = [
        .init(name: "Warm", items: ["romantic", "hopeful", "nostalgic", "comforting"]),
        .init(name: "Dark", items: ["tense", "lonely", "mysterious", "melancholic"]),
        .init(name: "Energetic", items: ["triumphant", "aggressive", "playful", "euphoric"]),
        .init(name: "Calm", items: ["dreamy", "intimate", "meditative", "atmospheric"])
    ]

    static let grooves = ["laid-back", "bouncing", "driving", "syncopated", "swinging", "steady", "loose", "punchy"]

    /// Each entry is instrument + performance, because the guide finds a bare
    /// instrument name gives the model less to act on than a role.
    static let instruments: [MusicPromptGroup] = [
        .init(name: "Keys and synth", items: [
            "sparse piano chords", "warm electric piano", "atmospheric synth pads", "pulsing synth arpeggios",
            "bright synth stabs"
        ]),
        .init(name: "Guitar", items: [
            "softly fingerpicked acoustic guitar", "strummed acoustic guitar",
            "clean electric guitar playing slow legato phrases", "distorted guitar playing a driving riff",
            "expressive slide guitar"
        ]),
        .init(name: "Bass", items: [
            "deep 808 sub-bass", "melodic bass guitar", "warm upright bass", "funky slap bass"
        ]),
        .init(name: "Drums and percussion", items: [
            "tight punchy drums", "brushed jazz drums", "four-on-the-floor kick", "restrained percussion",
            "heavy low percussion"
        ]),
        .init(name: "Strings and brass", items: [
            "low strings gradually building tension", "lush string section", "solo cello", "orchestral brass",
            "harp", "airy flute", "saxophone melody lines"
        ]),
        .init(name: "Global", items: ["plucked guzheng phrases", "expressive erhu melody", "bamboo flute"])
    ]

    static let voices = [
        "female alto", "female mezzo-soprano", "bright female soprano", "male baritone", "warm male tenor",
        "deep male vocal", "female and male duet"
    ]
    static let tones = ["airy", "warm", "raspy", "bright", "soft", "dark", "clear"]
    static let deliveries = ["intimate", "restrained", "conversational", "powerful", "rhythmic", "dramatic", "sing-rap"]
    static let techniques = [
        "breathy phrasing", "smooth legato", "subtle vibrato", "heavy autotune", "layered backing vocals",
        "close harmonies"
    ]
    static let emotions = ["vulnerable", "confident", "yearning", "detached", "playful", "frustrated"]

    static let production = [
        "clean modern finish", "warm vintage texture", "lo-fi tape saturation", "wide cinematic soundstage",
        "raw live-room feel", "polished radio-ready compression", "spacious mix with clear vocal separation",
        "controlled low end"
    ]
}

enum MusicTempo: String, CaseIterable, Identifiable, Sendable {
    case slow, mid, fast, accelerating
    var id: String { rawValue }

    /// Prompt wording. Qualitative on purpose: the guide warns against
    /// inventing precision, so an exact BPM is a separate opt-in.
    var phrase: String {
        switch self {
        case .slow: return "slow tempo"
        case .mid: return "mid-tempo"
        case .fast: return "fast, energetic tempo"
        case .accelerating: return "gradually accelerating tempo"
        }
    }
}

/// How the song should develop. One sentence each, because the guide's point
/// is contrast between sections, not a label.
enum MusicArc: String, CaseIterable, Identifiable, Sendable {
    case sparseToBig, slowBuild, steady, strippedBridge, fullArc, evolving
    var id: String { rawValue }

    var sentence: String {
        switch self {
        case .sparseToBig: return "Keep the verses sparse, then open into a wider, bigger chorus."
        case .slowBuild: return "Start quiet and build slowly toward a powerful climax, then resolve."
        case .steady: return "Hold steady, consistent energy throughout."
        case .strippedBridge: return "Strip the bridge back before a full, lifted final chorus."
        case .fullArc: return "Introduce quietly, build tension, release in the chorus, pull back in the bridge, then resolve."
        case .evolving: return "Evolve slowly with no hard section changes."
        }
    }
}

enum MusicVocalMode: Sendable { case vocal, instrumental }

/// Something worth telling the writer before they use the generated text.
enum MusicPromptWarning: Equatable, Sendable {
    /// Vocals requested but no voice named. The guide names this the main
    /// cause of a track drifting to instrumental.
    case noVoice
    /// Vocals requested, but the lyrics field is empty and will render instrumental.
    case vocalsWithoutLyrics
    /// Instrumental requested, but the lyrics field holds words.
    case instrumentalWithLyrics
    /// Instrumental requested without an instrument to carry the melody.
    case instrumentalNeedsLead
}

struct MusicPromptSelection: Equatable, Sendable {
    static let maxMoods = 3
    static let maxInstruments = 5
    static let maxTechniques = 2
    static let maxProduction = 2
    static let bpmRange = 40...220

    var genre: String?
    var influence: String?
    var scene = ""
    var moods: [String] = []
    var tempo: MusicTempo?
    var bpm: Int?
    var groove: String?
    var instruments: [String] = []
    var vocalMode: MusicVocalMode = .vocal
    var voice: String?
    var tone: String?
    var delivery: String?
    var techniques: [String] = []
    var emotion: String?
    var arc: MusicArc?
    var production: [String] = []

    /// Remove `value` if present, otherwise add it unless `limit` is reached.
    /// Order is kept: the first instrument picked is the one that leads.
    static func toggled(_ list: [String], _ value: String, limit: Int) -> [String] {
        if let index = list.firstIndex(of: value) {
            var next = list
            next.remove(at: index)
            return next
        }
        return list.count < limit ? list + [value] : list
    }
}

enum MusicPromptBuilder {
    /// The style description, in the guide's order: Style, Mood, Tempo and
    /// groove, Instruments, Vocals, Structure, Production. Empty choices
    /// contribute nothing, so a partial selection still reads as prose.
    static func caption(for selection: MusicPromptSelection) -> String {
        [
            style(selection), mood(selection), motion(selection), instruments(selection),
            vocals(selection), selection.arc?.sentence, production(selection)
        ]
        .compactMap { $0 }
        .joined(separator: " ")
    }

    static func warnings(for selection: MusicPromptSelection, lyricsAreInstrumental: Bool) -> [MusicPromptWarning] {
        var out: [MusicPromptWarning] = []
        switch selection.vocalMode {
        case .vocal:
            if selection.voice == nil { out.append(.noVoice) }
            if lyricsAreInstrumental { out.append(.vocalsWithoutLyrics) }
        case .instrumental:
            if selection.instruments.isEmpty { out.append(.instrumentalNeedsLead) }
            if !lyricsAreInstrumental { out.append(.instrumentalWithLyrics) }
        }
        return out
    }

    // MARK: - Sentences

    private static func style(_ s: MusicPromptSelection) -> String? {
        let scene = collapsed(s.scene)
        guard s.genre != nil || !scene.isEmpty else { return nil }
        var text = s.genre ?? "music"
        // An influence equal to the genre adds nothing; one without a genre
        // has nothing to modify.
        if s.genre != nil, let influence = s.influence, influence != s.genre {
            text += " with \(influence) influences"
        }
        if !scene.isEmpty { text += " for \(scene)" }
        return sentence(text)
    }

    private static func mood(_ s: MusicPromptSelection) -> String? {
        s.moods.isEmpty ? nil : sentence("\(list(s.moods)) mood")
    }

    private static func motion(_ s: MusicPromptSelection) -> String? {
        var text = s.tempo?.phrase
        if let bpm = s.bpm, MusicPromptSelection.bpmRange.contains(bpm) {
            text = text.map { "\($0) around \(bpm) BPM" } ?? "around \(bpm) BPM"
        }
        if let groove = s.groove {
            let phrase = "\(article(before: groove)) \(groove) groove"
            text = text.map { "\($0) with \(phrase)" } ?? phrase
        }
        return text.map(sentence)
    }

    private static func instruments(_ s: MusicPromptSelection) -> String? {
        guard let first = s.instruments.first else { return nil }
        let rest = Array(s.instruments.dropFirst())
        // Vocals lead a sung track, so the list is just a palette. With no
        // vocals the first pick is the lead and the rest support it, which
        // the guide recommends over letting every instrument compete.
        if s.vocalMode == .instrumental {
            return sentence(rest.isEmpty ? "led by \(first)" : "led by \(first), supported by \(list(rest))")
        }
        return sentence(list(s.instruments))
    }

    private static func vocals(_ s: MusicPromptSelection) -> String? {
        if s.vocalMode == .instrumental { return "Instrumental, no vocals." }
        var head = [s.tone, s.voice].compactMap { $0 }.joined(separator: " ")
        let hasDetail = s.delivery != nil || !s.techniques.isEmpty || s.emotion != nil
        guard !head.isEmpty || hasDetail else { return nil }
        head = head.isEmpty ? "vocals" : head + " vocals"
        var text = head
        if let delivery = s.delivery { text += " with \(delivery) delivery" }
        if !s.techniques.isEmpty { text += ", \(list(s.techniques))" }
        if let emotion = s.emotion { text += ", sounding \(emotion)" }
        return sentence(text)
    }

    private static func production(_ s: MusicPromptSelection) -> String? {
        s.production.isEmpty ? nil : sentence("production: \(list(s.production))")
    }

    // MARK: - Text helpers

    /// "a", "a and b", "a, b and c".
    private static func list(_ items: [String]) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        default: return items.dropLast().joined(separator: ", ") + " and " + items[items.count - 1]
        }
    }

    private static func article(before word: String) -> String {
        guard let first = word.lowercased().first else { return "a" }
        return "aeiou".contains(first) ? "an" : "a"
    }

    private static func collapsed(_ text: String) -> String {
        var out = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        while let last = out.last, ".!?,;:".contains(last) { out.removeLast() }
        return out
    }

    /// Capitalize the first letter only (acronyms like R&B keep their case)
    /// and end with a period.
    private static func sentence(_ text: String) -> String {
        guard let first = text.first else { return "" }
        return String(first).uppercased() + text.dropFirst() + "."
    }
}
