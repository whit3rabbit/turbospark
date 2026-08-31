import Foundation

/// A reason to send a command to the approval sheet. There is deliberately no
/// `approve` case.
///
/// **THE MODEL MAY ONLY ADD FRICTION.** `TerminalCommandClassifier.isAutoApprovable`
/// decides what runs, and this can only push a decision toward `.ask`. That is
/// not caution for its own sake, it is what the measurement supports: scored
/// against the two corpus lists in `TerminalRiskGateTests`, the model adds ZERO
/// true positives (the allowlist already refuses every evasion before the model
/// is consulted) and one to three false positives at every threshold. A type
/// with an `approve` case would invite someone to wire the losing direction.
public struct HazardVeto: Equatable, Sendable {
    public let reason: String
    public let hazard: Double
    public let obfuscation: Double
}

/// Scores for one command. Both heads run on one featurization.
public struct CommandScores: Equatable, Sendable {
    public let hazard: Double
    public let obfuscation: Double
}

/// The bundled linear permission-gate models.
///
/// **SHIPPED WITH THE VETO OFF, AND THAT IS THE MEASUREMENT TALKING.** Held out
/// by generator the hazard model scores 0.7010 against 0.9972 in-distribution,
/// and `python3 -m pytest tests/ -q` scores 1.000, so no threshold stops it
/// firing on a command `.auto` promises to run silently. Turning `vetoEnabled`
/// on today reddens `testOrdinaryDevelopmentCommandsStillRunUnprompted`.
///
/// What ships on by default is `advisoryReason`, which only ever decorates a
/// verdict that was ALREADY `.ask`, so it cannot change what runs. The rest is
/// plumbing kept ready so a better corpus is a weights swap rather than a port.
public enum CommandGate {

    /// Whether a veto may move an allowlisted command to `.ask`.
    /// Set from `MacAppSettings.commandAdvisoryVeto` at launch. Default false.
    nonisolated(unsafe) public static var vetoEnabled: Bool = false

    /// Probability at or above which the veto fires, when enabled.
    nonisolated(unsafe) public static var vetoThreshold: Double = 0.90

    // MARK: - Model

    struct Model {
        let charFeatures: Int
        let wordFeatures: Int
        let lexicalCount: Int
        let total: Int
        let names: [String]
        let intercepts: [Float]
        let weights: [[Float]]
        let wordPattern: NSRegularExpression

        var hazardIndex: Int? { names.firstIndex(of: "hazard") }
        var obfuscationIndex: Int? { names.firstIndex(of: "obfuscation") }
    }

    /// Loaded once, lazily. Swift static initialization is thread-safe, and the
    /// value is immutable afterwards, so `score` needs no lock.
    static let model: Model? = loadModel()

    /// The parity fixture, reached through the APP target's bundle.
    ///
    /// The test target ships no resources of its own, so `Bundle.module` inside
    /// a test resolves to the wrong bundle and finds nothing. Tests go through
    /// this accessor instead.
    static var oracleFixtureURL: URL? {
        Bundle.module.url(forResource: "permission-gate-oracle", withExtension: "json")
    }

    /// `permission-gate.bin` layout, little-endian:
    ///   "PGT1" | charFeatures | wordFeatures | lexicalCount | modelCount
    ///   then per model: name[16] | intercept:f32 | weights:f32[total]
    ///
    /// A flat binary rather than the trainer's JSON: 192 KiB against 713 KiB,
    /// and it loads with one read and no parsing.
    private static func loadModel() -> Model? {
        guard let url = Bundle.module.url(forResource: "permission-gate", withExtension: "bin"),
              let data = try? Data(contentsOf: url),
              data.count > 20
        else { return nil }

        var offset = 0
        func readUInt32() -> Int {
            defer { offset += 4 }
            return Int(data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
        }

        guard data.prefix(4) == Data("PGT1".utf8) else { return nil }
        offset = 4
        let charFeatures = readUInt32()
        let wordFeatures = readUInt32()
        let lexicalCount = readUInt32()
        let modelCount = readUInt32()
        let total = charFeatures + wordFeatures + lexicalCount
        guard total > 0, modelCount > 0, lexicalCount == CommandLexicalFeatures.count else {
            return nil
        }

        var names: [String] = []
        var intercepts: [Float] = []
        var weights: [[Float]] = []
        for _ in 0..<modelCount {
            guard offset + 16 + 4 + total * 4 <= data.count else { return nil }
            let raw = data.subdata(in: offset..<(offset + 16))
            offset += 16
            names.append(String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self))
            intercepts.append(data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: Float.self)
            })
            offset += 4
            var vector = [Float](repeating: 0, count: total)
            let lower = offset, upper = offset + total * 4
            vector.withUnsafeMutableBytes { destination in
                data.copyBytes(
                    to: destination.bindMemory(to: UInt8.self), from: lower..<upper)
            }
            offset += total * 4
            weights.append(vector)
        }

        // The pattern is stated here rather than read from the binary because a
        // mismatch must be a compile-visible edit, not a data surprise. The
        // `\b` anchors are load-bearing; see HashedFeatureVectorizer.
        guard let pattern = try? NSRegularExpression(pattern: "\\b[\\w./:-]+\\b") else {
            return nil
        }

        return Model(
            charFeatures: charFeatures, wordFeatures: wordFeatures,
            lexicalCount: lexicalCount, total: total, names: names,
            intercepts: intercepts, weights: weights, wordPattern: pattern)
    }

    // MARK: - Scoring

    /// Both head probabilities, or nil when the model is unavailable.
    ///
    /// Callable from any thread: the model is immutable and every buffer is
    /// local to the call.
    public static func score(_ command: String) -> CommandScores? {
        guard let model, !command.isEmpty else { return nil }
        let lowered = command.lowercased()

        var vector = HashedFeatureVectorizer.SparseVector()

        var charCounts: [Int: Float] = [:]
        HashedFeatureVectorizer.charNgrams(lowered, 3, 5) { gram in
            let index = HashedFeatureVectorizer.bucket(
                Array(gram), modulo: model.charFeatures)
            charCounts[index, default: 0] += 1
        }
        vector.addBlock(counts: charCounts, offset: 0)

        var wordCounts: [Int: Float] = [:]
        for gram in HashedFeatureVectorizer.wordNgrams(
            lowered, 1, 2, pattern: model.wordPattern)
        {
            let index = HashedFeatureVectorizer.bucket(
                Array(gram.utf8), modulo: model.wordFeatures)
            wordCounts[index, default: 0] += 1
        }
        vector.addBlock(counts: wordCounts, offset: model.charFeatures)

        // Lexical features are computed on the ORIGINAL text, not the lowered
        // copy, matching the trainer.
        vector.addRaw(
            CommandLexicalFeatures.compute(command),
            offset: model.charFeatures + model.wordFeatures)

        func probability(_ index: Int?) -> Double {
            guard let index else { return 0 }
            let z = vector.dot(model.weights[index], intercept: model.intercepts[index])
            return 1 / (1 + exp(-z))
        }
        return CommandScores(
            hazard: probability(model.hazardIndex),
            obfuscation: probability(model.obfuscationIndex))
    }

    // MARK: - The two entry points

    /// A veto, when the feature is enabled and the model is confident.
    ///
    /// Call this ONLY after `TerminalCommandClassifier.isAutoApprovable` has
    /// already said yes. Reached any earlier it would be deciding rather than
    /// advising, which is the arrangement this whole type exists to prevent.
    public static func veto(for command: String) -> HazardVeto? {
        guard vetoEnabled, let scores = score(command) else { return nil }
        if scores.hazard >= vetoThreshold {
            return HazardVeto(
                reason: String(format: "Command scored %.2f for hazard by the local classifier",
                               scores.hazard),
                hazard: scores.hazard, obfuscation: scores.obfuscation)
        }
        if scores.obfuscation >= vetoThreshold {
            return HazardVeto(
                reason: String(format: "Command scored %.2f for obfuscation by the local classifier",
                               scores.obfuscation),
                hazard: scores.hazard, obfuscation: scores.obfuscation)
        }
        return nil
    }

    /// A sentence for a verdict that is ALREADY `.ask`.
    ///
    /// Safe unconditionally: it changes what the approval sheet says and never
    /// what runs. Returns nil when the model has no strong opinion, so an
    /// unremarkable command does not gain a noise line.
    public static func advisoryReason(for command: String) -> String? {
        guard let scores = score(command) else { return nil }
        if scores.hazard >= 0.75 {
            return String(format: "local classifier: hazard %.2f", scores.hazard)
        }
        if scores.obfuscation >= 0.75 {
            return String(format: "local classifier: obfuscation %.2f", scores.obfuscation)
        }
        return nil
    }
}
