import Foundation
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
            default:
                positional.append(argument)
            }
        }

        guard positional.count == 1 || positional.count == 2 else {
            fatalError(
                "usage: QwenImageMLXBenchmark <mlx-model-directory> [output.png] [--steps N] [--prompt TEXT] [--seed N] [--width N] [--height N]"
            )
        }

        let model = positional[0]
        let outputURL = positional.count == 2
            ? URL(fileURLWithPath: positional[1])
            : FileManager.default.temporaryDirectory.appendingPathComponent("qwen-image-mlx-benchmark.png")
        let pipeline = QwenImage21Pipeline()
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

        let png = try await pipeline.generateToMemory(request) { progress in
            let elapsed = Date().timeIntervalSince(startedAt)
            emit("PROGRESS elapsed_s=\(String(format: "%.2f", elapsed)) stage=\(progress.stage.rawValue) step=\(progress.stepIndex)/\(progress.totalSteps)")
            breakdown.observe(stage: progress.stage.rawValue)
            if let footprint = PeakMemory.currentFootprintBytes() {
                peakMemoryBytes = max(peakMemoryBytes, footprint)
            }
        }
        breakdown.finish()
        if let footprint = PeakMemory.currentFootprintBytes() {
            peakMemoryBytes = max(peakMemoryBytes, footprint)
        }
        try png.write(to: outputURL)
        let elapsed = Date().timeIntervalSince(startedAt)
        emit(
            "RESULT elapsed_s=\(String(format: "%.2f", elapsed)) steps=\(steps) width=\(width) height=\(height) png_bytes=\(png.count) path=\(outputURL.path)"
        )
        let breakdownFields =
            breakdown.totals.sorted { $0.key < $1.key }
            .map { "\($0.key)=\(String(format: "%.2f", $0.value))" }
            .joined(separator: " ")
        emit("BREAKDOWN \(breakdownFields)")
        emit("PEAK_MEM bytes=\(peakMemoryBytes) gib=\(String(format: "%.2f", Double(peakMemoryBytes) / 1_073_741_824.0))")
    }
}
