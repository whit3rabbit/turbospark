import Foundation
import MLX
import QwenImage

/// Per-stage wall-clock accumulation over the pipeline progress events,
/// following the ZImageMLXBenchmark bucket contract so the two harnesses
/// report comparable numbers.
struct StageBreakdown {
    private var currentStage: String?
    private var stageStartedAt = Date()
    private(set) var totals: [String: Double] = [:]

    static let bucketByStage = [
        "Loading model": "load_weights_s",
        "Loading text encoder": "load_weights_s",
        "Loading transformer": "load_weights_s",
        "Loading VAE": "load_weights_s",
        "Encoding text": "text_encode_s",
        "Denoising": "denoise_s",
        "Decoding": "vae_decode_s",
        "Saving": "png_encode_s",
    ]

    mutating func observe(stage: String) {
        guard stage != currentStage else { return }
        if let previous = currentStage {
            let bucket = Self.bucketByStage[previous] ?? "other_s"
            totals[bucket, default: 0] += Date().timeIntervalSince(stageStartedAt)
        }
        currentStage = stage
        stageStartedAt = Date()
    }

    mutating func finish() {
        observe(stage: "\0done")
    }
}

/// Peak `phys_footprint` watermark, sampled on every progress event.
enum PeakMemory {
    static func currentFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPointer, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return info.phys_footprint
    }

    /// The kernel-maintained lifetime peak of the physical footprint. A sampled
    /// value misses transients that rise and fall between samples (the VAE
    /// decode window allocates float32 activations and returns them to the OS
    /// before the next sample), so the frozen peak must come from here.
    static func lifetimePeakFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPointer, &count)
            }
        }
        guard result == KERN_SUCCESS, info.ledger_phys_footprint_peak > 0 else { return nil }
        return UInt64(info.ledger_phys_footprint_peak)
    }
}

/// Polls the process footprint every 20 ms and keeps the maximum seen while
/// each pipeline stage was current. Lazy MLX weight materialization and VAE
/// activations rise and fall between progress events, so event-time sampling
/// misses the stage that sets the peak.
final class StagePeakSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var stage = "start"
    private var maxima: [String: UInt64] = [:]
    private var running = true

    func setStage(_ name: String) {
        lock.withLock { stage = name }
    }

    func start() {
        let thread = Thread { [self] in
            while lock.withLock({ running }) {
                if let footprint = PeakMemory.currentFootprintBytes() {
                    lock.withLock {
                        maxima[stage] = max(maxima[stage] ?? 0, footprint)
                    }
                }
                Thread.sleep(forTimeInterval: 0.02)
            }
        }
        thread.start()
    }

    func stop() -> [(String, UInt64)] {
        lock.withLock { running = false }
        return lock.withLock { maxima.sorted { $0.key < $1.key }.map { ($0.key, $0.value) } }
    }
}

func emit(_ line: String) {
    print(line)
    fflush(stdout)
}

@main
struct QwenImageMLXBenchmark {
    static func main() async throws {
        var positional: [String] = []
        var steps = 40
        var prompt = "A tiny red cabin beside a frozen lake at blue hour, pine forest, stars, cinematic lighting, wide composition, no text."
        var seed: UInt64 = 42
        var width = 1024
        var height = 1024
        var residency = QwenImageResidency.warm
        var vaeDtype = QwenImageVAEPrecision.float32

        var iterator = CommandLine.arguments.makeIterator()
        _ = iterator.next()
        while let argument = iterator.next() {
            switch argument {
            case "--steps":
                guard let value = iterator.next(), let parsed = Int(value), parsed > 0 else {
                    fatalError("--steps requires a positive integer")
                }
                steps = parsed
            case "--prompt":
                guard let value = iterator.next(), !value.isEmpty else {
                    fatalError("--prompt requires non-empty text")
                }
                prompt = value
            case "--seed":
                guard let value = iterator.next(), let parsed = UInt64(value) else {
                    fatalError("--seed requires an unsigned integer")
                }
                seed = parsed
            case "--width":
                guard let value = iterator.next(), let parsed = Int(value), parsed > 0 else {
                    fatalError("--width requires a positive integer")
                }
                width = parsed
            case "--height":
                guard let value = iterator.next(), let parsed = Int(value), parsed > 0 else {
                    fatalError("--height requires a positive integer")
                }
                height = parsed
            case "--vae-dtype":
                switch iterator.next() {
                case "float32": vaeDtype = .float32
                case "bfloat16": vaeDtype = .bfloat16
                default: fatalError("--vae-dtype requires float32 or bfloat16")
                }
            case "--residency":
                switch iterator.next() {
                case "warm": residency = .warm
                case "staged": residency = .staged
                default: fatalError("--residency requires warm or staged")
                }
            default:
                positional.append(argument)
            }
        }

        guard positional.count == 1 || positional.count == 2 else {
            fatalError(
                "usage: QwenImageMLXBenchmark <mlx-model-directory> [output.png] [--steps N] [--prompt TEXT] [--seed N] [--width N] [--height N] [--residency warm|staged] [--vae-dtype float32|bfloat16]"
            )
        }

        let model = positional[0]
        let outputURL = positional.count == 2
            ? URL(fileURLWithPath: positional[1])
            : FileManager.default.temporaryDirectory.appendingPathComponent("qwen-image-mlx-benchmark.png")
        let pipeline = QwenImage21Pipeline(residency: residency, vaePrecision: vaeDtype)
        let startedAt = Date()
        var breakdown = StageBreakdown()
        var peakMemoryBytes: UInt64 = 0
        let request = QwenImageGenerationRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            seed: seed,
            outputPath: outputURL,
            model: model
        )

        let sampler = StagePeakSampler()
        sampler.start()
        let png = try await pipeline.generateToMemory(request) { progress in
            sampler.setStage(progress.stage.rawValue)
            let elapsed = Date().timeIntervalSince(startedAt)
            // The live footprint at each event shows which stage owns the peak;
            // the lifetime peak below stays the number to quote.
            let live = PeakMemory.currentFootprintBytes().map {
                String(format: "%.2f", Double($0) / 1_073_741_824.0)
            } ?? "n/a"
            emit("PROGRESS elapsed_s=\(String(format: "%.2f", elapsed)) stage=\(progress.stage.rawValue) step=\(progress.stepIndex)/\(progress.totalSteps) footprint_gib=\(live)")
            breakdown.observe(stage: progress.stage.rawValue)
            if let footprint = PeakMemory.currentFootprintBytes() {
                peakMemoryBytes = max(peakMemoryBytes, footprint)
            }
        }
        breakdown.finish()
        let stagePeaks = sampler.stop()
        if let footprint = PeakMemory.currentFootprintBytes() {
            peakMemoryBytes = max(peakMemoryBytes, footprint)
        }
        if let lifetimePeak = PeakMemory.lifetimePeakFootprintBytes() {
            peakMemoryBytes = max(peakMemoryBytes, lifetimePeak)
        }
        try png.write(to: outputURL)
        let elapsed = Date().timeIntervalSince(startedAt)
        emit(
            "RESULT elapsed_s=\(String(format: "%.2f", elapsed)) steps=\(steps) width=\(width) height=\(height) residency=\(residency) vae_dtype=\(vaeDtype) png_bytes=\(png.count) path=\(outputURL.path)"
        )
        let breakdownFields =
            breakdown.totals.sorted { $0.key < $1.key }
            .map { "\($0.key)=\(String(format: "%.2f", $0.value))" }
            .joined(separator: " ")
        emit("BREAKDOWN \(breakdownFields)")
        for (stage, bytes) in stagePeaks {
            emit("STAGE_PEAK stage=\(stage) gib=\(String(format: "%.2f", Double(bytes) / 1_073_741_824.0))")
        }
        emit("PEAK_MEM bytes=\(peakMemoryBytes) gib=\(String(format: "%.2f", Double(peakMemoryBytes) / 1_073_741_824.0))")
    }
}
