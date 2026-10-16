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
        case .preparing: return AppStrings.localized("Encoding prompt")
        case .composing: return AppStrings.localized("Generating music")
        case .rendering: return AppStrings.localized("Rendering audio")
        case .saving: return AppStrings.localized("Saving audio")
        }
    }

    var counter: String? {
        guard let total else { return nil }
        let label = phase == .composing
            ? AppStrings.localized("Frames")
            : AppStrings.localized("Chunks")
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
                let label = AppStrings.localized("Steps")
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
                    ? AppStrings.localized("Denoising")
                    : AppStrings.localized("Decoding")
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

/// The stages of a speech run, for the Voice Cloning progress card. The
/// engines visit them in different orders (Qwen3 loads its speech model
/// after preparing the reference), so a stage is done once it was visited
/// and another is active, not because it sorts earlier.
struct AudioSpeechRunProgress: Equatable, Sendable {
    enum Stage: Int, CaseIterable, Sendable { case waiting, loadingModel, preparingReference, generating, saving }
    enum StageState: Equatable, Sendable { case pending, active, done }
    private(set) var stage: Stage = .waiting
    private(set) var reached: Set<Stage> = [.waiting]
    let startedAt: Date

    init(now: Date = Date()) { startedAt = now }

    /// Late engine events after saving began cannot move the card back.
    mutating func enter(_ next: Stage) {
        guard stage != .saving else { return }
        stage = next; reached.insert(next)
    }

    func state(of candidate: Stage) -> StageState {
        if candidate == stage { return .active }
        return reached.contains(candidate) ? .done : .pending
    }

    /// The stage an engine progress event names, from the Swift and Rust
    /// speech providers' stage strings.
    static func stage(forProgress name: String) -> Stage? {
        switch name {
        case "Preparing reference audio", "reference": return .preparingReference
        case "Loading speech model", "verifying": return .loadingModel
        case "Generating speech", "diffusion": return .generating
        default: return nil
        }
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
            if let stage = AudioSpeechRunProgress.stage(forProgress: update.stage) { speechRun?.enter(stage) }
            switch update.stage {
            case "Preparing reference audio": status = AppStrings.localized("Preparing reference audio")
            case "Loading speech model": status = AppStrings.localized("Loading speech model")
            case "Generating speech", "diffusion": status = AppStrings.localized("Generating speech")
            case "reference": status = AppStrings.localized("Preparing reference audio")
            case "verifying": status = AppStrings.localized("Verifying")
            default: status = update.stage
            }
            progress = update.total.flatMap { $0 > 0 ? Double(max(0, min($0, update.completed))) / Double($0) : nil }
        }
    }

    static func validateGeneratedAudio(_ audio: AudioPCM?) throws -> AudioPCM {
        guard let audio, audio.format.interleaved, (8_000...192_000).contains(audio.format.sampleRate),
              (1...8).contains(audio.format.channels), !audio.samples.isEmpty,
              audio.samples.count % Int(audio.format.channels) == 0,
              audio.samples.allSatisfy(\.isFinite) else {
            throw AudioWorkspaceError.message(AppStrings.localized("There is no audio to play or export."))
        }
        return audio
    }
}
