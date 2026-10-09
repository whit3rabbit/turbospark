import Foundation
import TurboSpark

/// Counters describe measured work within a phase, never an estimated share
/// of total runtime. Depth repeats per frame; denoising and decoding repeat
/// per chunk. Neither may reset the parent progress bar.
struct AudioGenerationProgress {
    enum Phase: Int { case preparing, composing, rendering, saving }
    private(set) var phase: Phase = .preparing
    private(set) var completed = 0
    private(set) var total: Int?
    private(set) var detail: String?
    let startedAt: Date
    private(set) var updatedAt: Date

    init(now: Date = Date()) { startedAt = now; updatedAt = now }

    var fraction: Double? {
        guard let total, total > 0 else { return nil }
        return Double(completed) / Double(total)
    }

    var title: String {
        switch phase {
        case .preparing: return String(localized: "Encoding prompt", bundle: .module)
        case .composing: return String(localized: "Generating music", bundle: .module)
        case .rendering: return String(localized: "Rendering audio", bundle: .module)
        case .saving: return String(localized: "Saving audio", bundle: .module)
        }
    }

    var counter: String? {
        guard let total else { return nil }
        let label = phase == .composing
            ? String(localized: "Frames", bundle: .module)
            : String(localized: "Chunks", bundle: .module)
        return "\(label): \(completed) / \(total)"
    }

    mutating func receive(_ event: AudioProgress, now: Date = Date()) {
        guard phase != .saving else { return }
        switch event.stage {
        case "tokenized", "ar", "depth":
            guard phase.rawValue <= Phase.composing.rawValue else { return }
            phase = .composing
            if event.stage == "ar" {
                advance(event.completed, total: event.total)
                detail = nil
            } else if event.stage == "depth", let count = event.total, count > 0 {
                // These are completed inner steps, not completed audio frames.
                let label = String(localized: "Steps", bundle: .module)
                detail = "\(label): \(max(0, min(count, event.completed))) / \(count)"
            }
        case "flow_chunk":
            if phase != .rendering { completed = 0; total = nil }
            phase = .rendering
            advance(event.completed, total: event.total)
            detail = nil
        case "euler", "vocoder":
            guard phase == .rendering else { return }
            if let count = event.total, count > 0 {
                let label = event.stage == "euler"
                    ? String(localized: "Denoising", bundle: .module)
                    : String(localized: "Decoding", bundle: .module)
                detail = "\(label): \(max(0, min(count, event.completed))) / \(count)"
            }
        default: return
        }
        updatedAt = now
    }

    mutating func saving(now: Date = Date()) {
        phase = .saving; completed = 0; total = nil; detail = nil; updatedAt = now
    }

    private mutating func advance(_ count: Int, total newTotal: Int?) {
        guard let newTotal, newTotal > 0 else { return }
        total = max(total ?? 0, newTotal)
        completed = max(completed, min(total ?? newTotal, max(0, count)))
    }
}

extension AudioWorkspaceController {
    func receiveProgress(_ update: AudioProgress, runID: UUID, token: UUID) {
        guard valid(token), isBusy, activeRunID == runID else { return }
        if var generation = generationProgress {
            generation.receive(update)
            generationProgress = generation
            status = generation.title
            progress = generation.fraction
        } else {
            status = update.stage
            progress = update.total.flatMap { $0 > 0 ? Double(max(0, min($0, update.completed))) / Double($0) : nil }
        }
    }

    static func validateGeneratedAudio(_ audio: AudioPCM?) throws -> AudioPCM {
        guard let audio, audio.format.interleaved, (8_000...192_000).contains(audio.format.sampleRate),
              (1...8).contains(audio.format.channels), !audio.samples.isEmpty,
              audio.samples.count % Int(audio.format.channels) == 0,
              audio.samples.allSatisfy(\.isFinite) else {
            throw AudioWorkspaceError.message(String(localized: "There is no audio to play or export.", bundle: .module))
        }
        return audio
    }
}
