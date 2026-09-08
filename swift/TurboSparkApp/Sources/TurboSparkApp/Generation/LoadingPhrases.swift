import Foundation

/// The witty streaming-status phrases, the qwen-code `loadingPhrases.ts`
/// parity: while a turn runs, the status footer cycles through these every
/// 15 seconds instead of repeating one static line.
///
/// Pure selection so the rotation is testable: the phrase is a function of
/// elapsed time only, no stored index, so a footer view can hold no state
/// and two footers always agree.
enum LoadingPhrases {
    /// Rotation period, matching qwen-code's 15s cycle.
    static let rotationInterval: TimeInterval = 15

    static let all: [String] = [
        "Consulting the tensors...",
        "Warming up the attention heads...",
        "Sharpening the logits...",
        "Herding tokens into place...",
        "Distilling an answer...",
        "Softmaxing the options...",
        "Reading between the embeddings...",
        "Untangling the context window...",
        "Greasing the expert router...",
        "Compiling a coherent reply...",
        "Decoding at the speed of thought...",
        "Aligning the KV cache...",
        "Sampling the multiverse...",
        "Polishing the prose...",
        "Following the gradient downhill...",
        "Counting parameters so you do not have to...",
        "Convincing the argmax...",
        "Rounding error, roundly ignored...",
    ]

    /// The phrase for an elapsed wall-clock stretch of the current turn.
    /// `turnStartDistance` exists so two turns started at different seconds
    /// do not always open on the same line; the streaming footer passes the
    /// turn's start time distance from a fixed epoch (0 in tests).
    static func phrase(
        elapsed: TimeInterval,
        turnStartDistance: TimeInterval = 0
    ) -> String {
        guard !all.isEmpty else { return "" }
        let tick = max(0, elapsed + turnStartDistance) / rotationInterval
        return all[Int(tick) % all.count]
    }
}
