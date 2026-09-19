import Foundation
import TurboSpark

/// Each request keeps the native batch-one envelope. Seeds advance explicitly
/// so choosing several images produces variations rather than identical copies.
enum ImageGenerationSequence {
    static func requests(options: ImageGenerateOptions, count: Int) -> [ImageGenerateOptions] {
        (0..<min(max(count, 1), 4)).map { offset in
            ImageGenerateOptions(
                prompt: options.prompt, seed: options.seed &+ UInt64(offset),
                width: options.width, height: options.height, steps: options.steps)
        }
    }

    @MainActor
    static func run(
        _ requests: [ImageGenerateOptions],
        generate: (Int, ImageGenerateOptions) async throws -> Void
    ) async throws {
        for (index, request) in requests.enumerated() {
            try Task.checkCancellation()
            try await generate(index, request)
        }
    }
}
