import Foundation
import QwenImage
import TurboSpark

/// Adapts the vendored Qwen-Image-2.1 Swift + MLX pipeline to the app's
/// image stream. Unlike the Z-Image adapter there is no snapshot staging
/// step: each component config carries its own quantization block and the
/// weights loader infers the per-layer width from tensor shapes, so the
/// retained source tree loads directly.
final class QwenImageGenerationSession: ImageGenerationSession, @unchecked Sendable {
    private let pipeline: QwenImage21Pipeline
    private let modelSpec: String
    private let modelID: String
    private let revision: String
    private let quantization: String
    private let producer = ImageProducerSlot()

    /// `residency` defaults to `.staged`: the text encoder is 9.4 GB for the
    /// Turbo checkpoint and is needed only until the prompt is encoded.
    ///
    /// The VAE decodes in bfloat16, the dtype the VAE is stored in and the
    /// Diffusers reference decodes in. Measured at 1024 x 1024 it cuts the
    /// lifetime peak from 20.5 to 11.2 GiB (the decode, not the denoiser,
    /// set the float32 peak) for a 50.9 dB PSNR difference from the float32
    /// image. See docs/QWEN_IMAGE_21_TURBO.md.
    init(model: ImageInstalledModel, residency: QwenImageResidency = .staged) throws {
        modelID = model.modelID
        revision = model.revision
        quantization = model.quantization
        guard let sourcePath = model.sourcePath,
              FileManager.default.fileExists(atPath: sourcePath) else {
            throw QwenImageSessionError.missingRetainedSource(model.alias)
        }
        modelSpec = sourcePath
        pipeline = QwenImage21Pipeline(residency: residency, vaePrecision: .bfloat16)
    }

    func cancel() {
        producer.cancel()
    }

    func waitUntilIdle() async {
        await producer.waitUntilIdle()
    }

    func unload() async {
        cancel()
        await producer.waitUntilIdle()
        pipeline.unloadModel()
    }

    // Backstop for any path that just drops the session: by deinit nothing
    // can be running on the pipeline, and `unloadModel` clears the MLX cache.
    deinit {
        pipeline.unloadModel()
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
            let token = UUID()
            let task = Task.detached(priority: .userInitiated) {
                do {
                    // Hold the native permit through producer completion, including
                    // cancellation drain, so served audio/text cannot overlap MLX.
                    var permit: NativeHeavyWorkPermit?
                    while permit == nil {
                        try Task.checkCancellation()
                        permit = try NativeHeavyWorkPermit.tryAcquire()
                        if permit == nil { try await Task.sleep(for: .milliseconds(50)) }
                    }
                    defer { permit?.release() }
                    // The pipeline floors each side to a multiple of 32, but
                    // the server admits multiples of 16. Reject instead of
                    // returning a smaller image than the metadata claims.
                    let multiple = Int(QwenImageModelMetadata.dimensionMultiple)
                    guard Int(options.width) % multiple == 0, Int(options.height) % multiple == 0 else {
                        throw QwenImageSessionError.dimensionsNotSupported(
                            Int(options.width), Int(options.height), multiple)
                    }
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
                        switch progress.stage {
                        case .loadingModel: stage = "loading_model"
                        case .loadingTextEncoder: stage = "loading_text_encoder"
                        case .encodingText: stage = "text_encoder"
                        case .loadingTransformer: stage = "loading_transformer"
                        case .loadingVAE: stage = "loading_vae"
                        case .denoising: stage = "transformer"
                        case .decoding: stage = "vae_decoder"
                        case .saving: stage = "png_encode"
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
                self.producer.clear(token: token)
            }
            self.producer.store(task, token: token)
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
        }
    }
}

enum QwenImageSessionError: LocalizedError, Equatable {
    case missingRetainedSource(String)
    case dimensionsNotSupported(Int, Int, Int)

    var errorDescription: String? {
        switch self {
        case .dimensionsNotSupported(let width, let height, let multiple):
            return "Qwen-Image needs each side to be a multiple of \(multiple) pixels; \(width)x\(height) is not."
        case .missingRetainedSource(let alias):
            return """
                The '\(alias)' install predates source retention and cannot run. \
                Delete and reinstall it from the model list.
                """
        }
    }
}
