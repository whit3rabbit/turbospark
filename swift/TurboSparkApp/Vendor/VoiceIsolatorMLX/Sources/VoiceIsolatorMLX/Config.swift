import Foundation

/// Architecture of a checkpoint, read from its metadata.
public struct Config: Codable, Sendable, Equatable {
    public var width = 512
    public var blockTypes = ["transformer", "mamba", "transformer", "mamba",
                             "transformer", "mamba", "transformer"]
    public var numHeads = 8
    public var nFFT = 2048
    public var hopLength = 480
    public var sampleRate = 48000
    public var bandPlan: [[Int]] = [[100, 1000], [250, 4000], [500, 8000],
                                    [1000, 16000], [2000, 24000]]
    public var correctionBound: Double = 0.5
    public var chunkSeconds: Double = 8.0
    public var overlapSeconds: Double = 2.0
    public var exclusiveSelfAttention = true
    public var exclusiveAxes = "time"
    public var mambaState = 16
    public var mambaConv = 4
    public var mambaExpand = 2
    public var causal = false
    public var lookaheadFrames = 0
    public var lookaheadFramesChoices: [Int] = []
    public var dualMode = false
    public var modeFilm = false
    public var attentionWindowFrames = 0
    public var offlinePasses = 1

    enum CodingKeys: String, CodingKey {
        case width
        case blockTypes = "block_types"
        case numHeads = "num_heads"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
        case sampleRate = "sample_rate"
        case bandPlan = "band_plan"
        case correctionBound = "correction_bound"
        case chunkSeconds = "chunk_seconds"
        case overlapSeconds = "overlap_seconds"
        case exclusiveSelfAttention = "exclusive_self_attention"
        case exclusiveAxes = "exclusive_axes"
        case mambaState = "mamba_state"
        case mambaConv = "mamba_conv"
        case mambaExpand = "mamba_expand"
        case causal
        case lookaheadFrames = "lookahead_frames"
        case lookaheadFramesChoices = "lookahead_frames_choices"
        case dualMode = "dual_mode"
        case modeFilm = "mode_film"
        case attentionWindowFrames = "attention_window_frames"
        case offlinePasses = "offline_passes"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func get<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        let d = Config()
        width = try get(.width, d.width)
        blockTypes = try get(.blockTypes, d.blockTypes)
        numHeads = try get(.numHeads, d.numHeads)
        nFFT = try get(.nFFT, d.nFFT)
        hopLength = try get(.hopLength, d.hopLength)
        sampleRate = try get(.sampleRate, d.sampleRate)
        bandPlan = try get(.bandPlan, d.bandPlan)
        correctionBound = try get(.correctionBound, d.correctionBound)
        chunkSeconds = try get(.chunkSeconds, d.chunkSeconds)
        overlapSeconds = try get(.overlapSeconds, d.overlapSeconds)
        exclusiveSelfAttention = try get(.exclusiveSelfAttention, d.exclusiveSelfAttention)
        exclusiveAxes = try get(.exclusiveAxes, d.exclusiveAxes)
        mambaState = try get(.mambaState, d.mambaState)
        mambaConv = try get(.mambaConv, d.mambaConv)
        mambaExpand = try get(.mambaExpand, d.mambaExpand)
        causal = try get(.causal, d.causal)
        lookaheadFrames = try get(.lookaheadFrames, d.lookaheadFrames)
        lookaheadFramesChoices = try get(.lookaheadFramesChoices, d.lookaheadFramesChoices)
        dualMode = try get(.dualMode, d.dualMode)
        modeFilm = try get(.modeFilm, d.modeFilm)
        attentionWindowFrames = try get(.attentionWindowFrames, d.attentionWindowFrames)
        offlinePasses = try get(.offlinePasses, d.offlinePasses)
    }

    /// Parses the checkpoint's JSON config and refuses what this port does not
    /// implement.
    public static func parse(json: String) throws -> Config {
        guard let data = json.data(using: .utf8) else {
            throw VoiceIsolatorError.invalidCheckpoint("config is not UTF-8")
        }
        if let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let required: [String: Any?] = [
                "representation": "bands", "decoder_type": "complex_mask_residual",
            ]
            for (key, value) in required {
                if let got = raw[key], "\(got)" != "\(value!)" {
                    throw VoiceIsolatorError.invalidCheckpoint("\(key)=\(got) is not supported")
                }
            }
        }
        let config = try JSONDecoder().decode(Config.self, from: data)
        if config.dualMode || config.modeFilm {
            throw VoiceIsolatorError.invalidCheckpoint("dual_mode / mode_film are not supported")
        }
        if config.width % (2 * config.numHeads) != 0 {
            throw VoiceIsolatorError.invalidCheckpoint("width must be a multiple of 2 * num_heads")
        }
        return config
    }

    public func json() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(data: try! encoder.encode(self), encoding: .utf8)!
    }

    public var headDim: Int { width / numHeads }
    public var inner: Int { mambaExpand * width }
    public var dtRank: Int { (width + 15) / 16 }
    public var bins: Int { nFFT / 2 + 1 }
    public var exclusiveTime: Bool {
        exclusiveSelfAttention && (exclusiveAxes == "both" || exclusiveAxes == "time")
    }
    public var exclusiveBand: Bool {
        exclusiveSelfAttention && (exclusiveAxes == "both" || exclusiveAxes == "band")
    }

    /// Bin ranges [start, end) of the bands, covering every bin once.
    public func bandEdges() -> [(Int, Int)] {
        let hzPerBin = Double(sampleRate) / Double(nFFT)
        var edges: [(Int, Int)] = []
        var start = 0
        for pair in bandPlan {
            // Python's round(): half to even
            let width = max(1, Int((Double(pair[0]) / hzPerBin).rounded(.toNearestOrEven)))
            let upto = min(bins, Int((Double(pair[1]) / hzPerBin).rounded(.toNearestOrEven)))
            while start < upto {
                let end = min(start + width, upto)
                edges.append((start, end))
                start = end
            }
        }
        if start < bins { edges.append((start, bins)) }
        return edges
    }
}

/// Index tables that pack the ragged bands into dense, padded tensors.
public struct BandLayout: Sendable {
    public let edges: [(Int, Int)]
    public let inPad: Int
    public let outPad: Int
    /// [bands * inPad] indices into the interleaved (re, im) components with a
    /// trailing zero column at index 2 * bins.
    public let gather: [Int32]
    public let valid: [Float]
    public let counts: [Float]
    /// [4 * bins] indices of the valid decoder outputs in [bands * outPad].
    public let keep: [Int32]

    public init(config: Config, inMultiple: Int = 1) {
        edges = config.bandEdges()
        let widths = edges.map { $0.1 - $0.0 }
        let maxWidth = widths.max()!
        inPad = (2 * maxWidth + inMultiple - 1) / inMultiple * inMultiple
        outPad = 4 * maxWidth
        let bins = config.bins
        var gather = [Int32](repeating: Int32(2 * bins), count: edges.count * inPad)
        var valid = [Float](repeating: 0, count: edges.count * inPad)
        var keep: [Int32] = []
        for (band, (s, e)) in edges.enumerated() {
            for j in 0 ..< 2 * (e - s) {
                gather[band * inPad + j] = Int32(2 * s + j)
                valid[band * inPad + j] = 1
            }
            for j in 0 ..< 4 * (e - s) { keep.append(Int32(band * outPad + j)) }
        }
        self.gather = gather
        self.valid = valid
        self.counts = widths.map { Float(2 * $0) }
        self.keep = keep
    }
}

public enum VoiceIsolatorError: Error, CustomStringConvertible {
    case invalidCheckpoint(String)
    case invalidArgument(String)
    case audio(String)
    case download(String)

    public var description: String {
        switch self {
        case .invalidCheckpoint(let m): return "invalid checkpoint: \(m)"
        case .invalidArgument(let m): return "invalid argument: \(m)"
        case .audio(let m): return "audio: \(m)"
        case .download(let m): return "download: \(m)"
        }
    }
}
