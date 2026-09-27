import Foundation
import ZImage

@main
struct ZImageMLXBenchmark {
    static func main() async throws {
        var positional: [String] = []
        var steps = 9
        var guidance: Float = 0

        var iterator = CommandLine.arguments.makeIterator()
        _ = iterator.next()
        while let argument = iterator.next() {
            switch argument {
            case "--steps":
                guard let value = iterator.next(), let parsed = Int(value), parsed > 0 else {
                    fatalError("--steps requires a positive integer")
                }
                steps = parsed
            case "--guidance":
                guard let value = iterator.next(), let parsed = Float(value), parsed >= 0 else {
                    fatalError("--guidance requires a non-negative number")
                }
                guidance = parsed
            default:
                positional.append(argument)
            }
        }

        guard positional.count == 1 || positional.count == 2 else {
            fatalError(
                "usage: ZImageMLXBenchmark <mlx-model-directory> [output.png] [--steps N] [--guidance F]")
        }

        let model = positional[0]
        let outputURL = positional.count == 2
            ? URL(fileURLWithPath: positional[1])
            : FileManager.default.temporaryDirectory.appendingPathComponent("zimage-mlx-benchmark.png")
        let pipeline = ZImagePipeline()
        let startedAt = Date()
        let request = ZImageGenerationRequest(
            prompt: "A tiny red cabin beside a frozen lake at blue hour, pine forest, stars, cinematic lighting, wide composition, no text.",
            width: 1024,
            height: 1024,
            steps: steps,
            guidanceScale: guidance,
            seed: 42,
            outputPath: outputURL,
            model: model,
            runtimeOptions: ZImageRuntimeOptions(residencyPolicy: .warm)
        )

        let png = try await pipeline.generateToMemory(request) { progress in
            let elapsed = Date().timeIntervalSince(startedAt)
            emit("PROGRESS elapsed_s=\(String(format: "%.2f", elapsed)) stage=\(progress.stage.rawValue) step=\(progress.stepIndex)/\(progress.totalSteps)")
        }
        try png.write(to: outputURL)
        let elapsed = Date().timeIntervalSince(startedAt)
        emit(
            "RESULT elapsed_s=\(String(format: "%.2f", elapsed)) steps=\(steps) guidance=\(guidance) png_bytes=\(png.count) path=\(outputURL.path)"
        )
    }

    private static func emit(_ message: String) {
        FileHandle.standardOutput.write(Data((message + "\n").utf8))
    }
}
