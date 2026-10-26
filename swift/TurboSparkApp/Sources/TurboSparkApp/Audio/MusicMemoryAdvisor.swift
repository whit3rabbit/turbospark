import Foundation
import Metal
import TurboSpark

/// Memory and context size advisor for the Music generation section.
/// Evaluates unified host and GPU working set limits to prevent out-of-memory
/// conditions while maximizing context frames when hardware supports it.
enum MusicMemoryAdvisor {
    /// Reference constants for MiniMax Music 3 and general music text-to-audio models.
    static let defaultContextSize = 10_240
    static let defaultMaxFrames = 9_000
    static let defaultFrameRate = 25.0
    static let defaultMaxDuration = 360.0

    /// Base memory working reserve required by the flow decoder and vocoder.
    static let baseWorkingReserveBytes: UInt64 = 2 * 1024 * 1024 * 1024 // 2 GiB
    /// Estimated memory growth per second of generated audio.
    static let memoryPerSecondBytes: UInt64 = 64 * 1024 * 1024 // 64 MiB/s
    /// Typical MiniMax Music 3 resident weight footprint (~9.2 GiB).
    static let defaultModelWeightBytes: UInt64 = 9_865_000_000

    struct Profile: Sendable, Equatable {
        let physicalMemoryBytes: UInt64
        let budgetBytes: UInt64
        let modelResidentBytes: UInt64
        let maxModelContextTokens: Int
        let maxModelFrames: Int
        let frameRate: Double
        let maxSupportedFrames: Int
        let maxSupportedSeconds: Double
        let canSupportMaxContext: Bool
        let tierSummary: String
    }

    /// Evaluates system hardware memory against the active or given music model.
    static func evaluate(for audioProfile: AudioProfile?) -> Profile {
        let physical = ProcessInfo.processInfo.physicalMemory
        let metalDevice = MTLCreateSystemDefaultDevice()
        let workingSet = metalDevice?.recommendedMaxWorkingSetSize ?? (physical * 3 / 4)
        let budget = min(physical * 3 / 4, workingSet)

        let modelBytes: UInt64
        if let resident = audioProfile?.residentMemoryBytes, resident > 0 {
            modelBytes = resident
        } else {
            modelBytes = defaultModelWeightBytes
        }

        let maxTokens = audioProfile?.capabilities.maxContextTokens ?? defaultContextSize
        let maxModelFrames = audioProfile?.capabilities.maxAudioFrames ?? defaultMaxFrames
        let frameRate = Double(audioProfile?.capabilities.frameRateHz ?? Int(defaultFrameRate))
        let maxModelDuration = Double(audioProfile?.capabilities.maxInputSeconds ?? Int(defaultMaxDuration))

        // Full context needs: model weights + base overhead (2 GiB) + duration scaling
        let maxDurationNeed = baseWorkingReserveBytes.saturatingAdd(
            UInt64(maxModelDuration.rounded(.up)) * memoryPerSecondBytes
        )
        let totalMaxContextNeed = modelBytes.saturatingAdd(maxDurationNeed)

        let canSupportMax = budget >= totalMaxContextNeed

        let maxSupportedSeconds: Double
        let maxSupportedFrames: Int

        if canSupportMax {
            maxSupportedSeconds = maxModelDuration
            maxSupportedFrames = maxModelFrames
        } else {
            // Constrained memory: compute safe budget headroom
            let reserved = modelBytes.saturatingAdd(baseWorkingReserveBytes)
            if budget > reserved {
                let available = budget - reserved
                let rawSeconds = Double(available) / Double(memoryPerSecondBytes)
                let clamped = min(maxModelDuration, max(15.0, rawSeconds))
                maxSupportedSeconds = clamped
                maxSupportedFrames = min(maxModelFrames, max(375, Int(clamped * frameRate)))
            } else {
                // Minimum fallback for tightly constrained environments
                maxSupportedSeconds = 30.0
                maxSupportedFrames = min(maxModelFrames, Int(30.0 * frameRate))
            }
        }

        let gibBudget = Double(budget) / (1024 * 1024 * 1024)
        let physicalGib = Double(physical) / (1024 * 1024 * 1024)
        let tierSummary: String
        if canSupportMax {
            tierSummary = String(
                format: AppStrings.localized("%.0f GB Unified Memory - Full context supported (up to %d frames / %.0fs)"),
                physicalGib, maxModelFrames, maxModelDuration
            )
        } else {
            tierSummary = String(
                format: AppStrings.localized("%.0f GB Unified Memory (%.0f GB budget) - Constrained to %d frames (%.0fs)"),
                physicalGib, gibBudget, maxSupportedFrames, maxSupportedSeconds
            )
        }

        return Profile(
            physicalMemoryBytes: physical,
            budgetBytes: budget,
            modelResidentBytes: modelBytes,
            maxModelContextTokens: maxTokens,
            maxModelFrames: maxModelFrames,
            frameRate: frameRate,
            maxSupportedFrames: maxSupportedFrames,
            maxSupportedSeconds: maxSupportedSeconds,
            canSupportMaxContext: canSupportMax,
            tierSummary: tierSummary
        )
    }

    /// Approximate prompt token count for description and lyrics.
    static func estimatePromptTokens(caption: String, lyrics: String) -> Int {
        let chars = caption.count + lyrics.count
        guard chars > 0 else { return 0 }
        // Qwen tokenizer averages ~3.5 characters per token across natural English / formatting
        return max(1, Int((Double(chars) / 3.5).rounded(.up)))
    }

    /// Validates a music recipe against the hardware memory profile and context constraints.
    static func validate(recipe: AudioRecipe, profile: AudioProfile?) -> String? {
        let mem = evaluate(for: profile)
        let duration = recipe.durationSeconds
        guard duration.isFinite, duration > 0 else {
            return AppStrings.localized("Duration must be greater than 0 seconds.")
        }

        let frames = Int((duration * mem.frameRate).rounded())

        if frames > mem.maxSupportedFrames {
            return String(
                format: AppStrings.localized("Duration requires %d frames (%.0fs), exceeding the safe system memory limit of %d frames (%.0fs)."),
                frames, duration, mem.maxSupportedFrames, mem.maxSupportedSeconds
            )
        }

        if frames > mem.maxModelFrames {
            return String(
                format: AppStrings.localized("Duration requires %d frames, exceeding the model frame cap of %d frames."),
                frames, mem.maxModelFrames
            )
        }

        let promptTokens = estimatePromptTokens(caption: recipe.caption, lyrics: recipe.lyrics)
        let totalTokens = promptTokens + frames
        let contextLimit = recipe.music?.contextSize ?? mem.maxModelContextTokens
        if totalTokens > contextLimit {
            return String(
                format: AppStrings.localized("Prompt (%d tokens) and music (%d frames) total %d, exceeding context size %d."),
                promptTokens, frames, totalTokens, contextLimit
            )
        }

        return nil
    }
}

private extension UInt64 {
    func saturatingAdd(_ other: UInt64) -> UInt64 {
        let (res, overflow) = addingReportingOverflow(other)
        return overflow ? UInt64.max : res
    }
}
