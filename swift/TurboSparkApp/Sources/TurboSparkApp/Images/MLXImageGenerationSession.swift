import Foundation
import TurboSpark
import ZImage

protocol ImageGenerationSession: AnyObject, Sendable {
    func cancel()
    func generate(_ options: ImageGenerateOptions) -> AsyncThrowingStream<ImageGenerationEvent, Error>
}

extension TurboSparkImageSession: ImageGenerationSession {}

/// Adapts the upstream Swift + MLX Z-Image pipeline to the app's image stream.
/// The app-installed packed image remains available for native fallback, while
/// the retained source tree lets MLX load the original safetensors directly.
final class MLXImageGenerationSession: ImageGenerationSession, @unchecked Sendable {
    private let pipeline = ZImagePipeline()
    private let modelSpec: String
    private let modelID: String
    private let revision: String
    private let quantization: String
    private let taskLock = NSLock()
    private var activeTask: Task<Void, Never>?

    init(model: ImageInstalledModel) {
        modelID = model.modelID
        revision = model.revision
        quantization = model.quantization
        if let sourcePath = model.sourcePath,
           FileManager.default.fileExists(atPath: sourcePath) {
            modelSpec = sourcePath
        } else {
            // Existing packed installs predate source retention. The upstream
            // resolver can fetch their exact catalog revision into the Hub cache.
            modelSpec = "\(model.modelID):\(model.revision)"
        }
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
                    let request = ZImageGenerationRequest(
                        prompt: options.prompt,
                        width: Int(options.width),
                        height: Int(options.height),
                        steps: Int(options.steps),
                        guidanceScale: 0,
                        seed: options.seed,
                        model: modelSpec,
                        runtimeOptions: ZImageRuntimeOptions(residencyPolicy: .warm)
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
                        guidanceScale: 0,
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
                            guidancePolicy: "mlx-z-image-turbo"
                        ),
                        noiseProvenance: "mlx-z-image-seeded",
                        engineRevision: "Z-Image.swift"
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
