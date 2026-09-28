import Foundation
import QwenImage
import TurboSpark

/// Adapts the vendored Qwen-Image-2.1 Swift + MLX pipeline to the app's
/// image stream. Unlike the Z-Image adapter there is no snapshot staging
/// step: the MLX community tree carries its quantization inside each
/// component config, so the retained source directory loads directly.
final class QwenImageGenerationSession: ImageGenerationSession, @unchecked Sendable {
    private let pipeline = QwenImage21Pipeline()
    private let modelSpec: String
    private let modelID: String
    private let revision: String
    private let quantization: String
    private let taskLock = NSLock()
    private var activeTask: Task<Void, Never>?

    init(model: ImageInstalledModel) throws {
        modelID = model.modelID
        revision = model.revision
        quantization = model.quantization
        guard let sourcePath = model.sourcePath,
              FileManager.default.fileExists(atPath: sourcePath) else {
            throw QwenImageSessionError.missingRetainedSource(model.alias)
        }
        modelSpec = sourcePath
    }

    func cancel() {
        taskLock.lock()
        let task = activeTask
        taskLock.unlock()
        task?.cancel()
    }

    func generate(
        _ options: ImageGenerateOptions
    ) -> AsyncThrowingStream<ImageGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let pipeline = self.pipeline
            let modelSpec = self.modelSpec
            let modelID = self.modelID
            let revision = self.revision
            let quantization = self.quantization
            let task = Task.detached(priority: .userInitiated) {
                do {
                    let request = QwenImageGenerationRequest(
                        prompt: options.prompt,
                        width: Int(options.width),
                        height: Int(options.height),
                        steps: Int(options.steps),
                        trueCfgScale: 1,
                        seed: options.seed,
                        model: modelSpec
                    )
                    let png = try await pipeline.generateToMemory(request) { progress in
                        let stage: String
                        switch progress.stage.rawValue {
                        case "Encoding text": stage = "text_encoder"
                        case "Denoising": stage = "transformer"
                        case "Decoding": stage = "vae_decoder"
                        case "Saving": stage = "png_encode"
                        default: stage = "loading_model"
                        }
                        continuation.yield(.stage(
                            name: stage,
                            completed: progress.stepIndex,
                            total: progress.totalSteps
                        ))
                    }
                    try Task.checkCancellation()
                    let steps = options.steps
                    let metadata = GeneratedImageMetadata(
                        prompt: options.prompt,
                        seed: options.seed,
                        width: options.width,
                        height: options.height,
                        batch: 1,
                        schedulerSteps: steps,
                        transformerForwards: steps,
                        guidanceScale: 1,
                        modelID: modelID,
                        modelRevision: revision,
                        componentRevisions: ["source": revision],
                        quantization: quantization,
                        scheduler: GeneratedImageSchedulerMetadata(
                            numTrainTimesteps: 0,
                            shift: 0,
                            timesteps: [],
                            sigmas: [],
                            evaluationCount: steps,
                            guidancePolicy: "mlx-qwen-image-2.1"
                        ),
                        noiseProvenance: "mlx-qwen-image-seeded",
                        engineRevision: "QwenImage.swift"
                    )
                    continuation.yield(.finished(ImageGenerationResult(png: png, metadata: metadata)))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                self.clearActiveTask()
            }
            self.storeActiveTask(task)
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
        }
    }

    private func storeActiveTask(_ task: Task<Void, Never>) {
        taskLock.lock()
        activeTask = task
        taskLock.unlock()
    }

    private func clearActiveTask() {
        taskLock.lock()
        activeTask = nil
        taskLock.unlock()
    }
}

private enum QwenImageSessionError: LocalizedError {
    case missingRetainedSource(String)

    var errorDescription: String? {
        switch self {
        case .missingRetainedSource(let alias):
            return """
                The '\(alias)' install predates source retention and cannot run. \
                Delete and reinstall it from the model list.
                """
        }
    }
}
