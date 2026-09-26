import Foundation
import ZImage

@main
struct ZImageMLXBenchmark {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else {
            fatalError("usage: ZImageMLXBenchmark <mlx-model-directory> [output.png]")
        }

        let model = CommandLine.arguments[1]
        let outputURL = CommandLine.arguments.count == 3
            ? URL(fileURLWithPath: CommandLine.arguments[2])
            : FileManager.default.temporaryDirectory.appendingPathComponent("zimage-mlx-benchmark.png")
        let pipeline = ZImagePipeline()
        let startedAt = Date()
        let request = ZImageGenerationRequest(
            prompt: "A tiny red cabin beside a frozen lake at blue hour, pine forest, stars, cinematic lighting, wide composition, no text.",
            width: 1024,
            height: 1024,
            steps: 9,
            guidanceScale: 0,
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
        emit("RESULT elapsed_s=\(String(format: "%.2f", elapsed)) png_bytes=\(png.count) path=\(outputURL.path)")
    }

    private static func emit(_ message: String) {
        FileHandle.standardOutput.write(Data((message + "\n").utf8))
    }
}
