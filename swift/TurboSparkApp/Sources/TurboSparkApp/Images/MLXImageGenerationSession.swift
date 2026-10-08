import Foundation
import TurboSpark
import ZImage

protocol ImageGenerationSession: AnyObject, Sendable {
    func cancel()
    func generate(_ options: ImageGenerateOptions) -> AsyncThrowingStream<ImageGenerationEvent, Error>
    /// Returns once the detached producer behind the last `generate` has
    /// actually stopped. Cancelling a stream finishes the CONSUMER at once,
    /// but the producer keeps running until its next cancellation point
    /// (model load, text encode and VAE decode have none), so the job permit
    /// must not be released before this returns.
    func waitUntilIdle() async
    /// Cancels, waits for the producer to stop, then releases the pipeline's
    /// weights AND the allocator cache behind them. Dropping the session
    /// alone deallocates the weight buffers into the MLX buffer cache, where
    /// they stay resident (10 to 20 GB at `.warm` residency) until some later
    /// cache clear, so the memory is not returned to the system.
    func unload() async
}

extension ImageGenerationSession {
    func waitUntilIdle() async {}
    func unload() async { cancel() }
}

/// The session's one running producer, with an identity check on clear.
/// A previous run's late `clear` must not null the NEXT run's task (that
/// would make it uncancellable), and `waitUntilIdle` must see the newest.
final class ImageProducerSlot: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var token: UUID?

    func store(_ task: Task<Void, Never>, token: UUID) {
        lock.lock()
        self.task = task
        self.token = token
        lock.unlock()
    }

    /// Clears only if `token` still owns the slot.
    func clear(token: UUID) {
        lock.lock()
        if self.token == token {
            task = nil
            self.token = nil
        }
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        let current = task
        lock.unlock()
        current?.cancel()
    }

    func waitUntilIdle() async {
        // A producer that finishes clears its own slot, so loop until the
        // slot is empty or holds a task that has already completed.
        while true {
            lock.lock()
            let current = task
            lock.unlock()
            guard let current else { return }
            await current.value
            lock.lock()
            let same = task == current
            if same { task = nil; token = nil }
            lock.unlock()
            if same { return }
        }
    }
}

/// Adapts the upstream Swift + MLX Z-Image pipeline to the app's image stream.
/// The retained source tree lets MLX load the original safetensors directly.
final class MLXImageGenerationSession: ImageGenerationSession, @unchecked Sendable {
    private let pipeline = ZImagePipeline()
    private let modelSpec: String
    private let sourcePath: String?
    private let modelID: String
    private let revision: String
    private let quantization: String
    private let producer = ImageProducerSlot()

    init(model: ImageInstalledModel) {
        modelID = model.modelID
        revision = model.revision
        quantization = model.quantization
        if let sourcePath = model.sourcePath,
           FileManager.default.fileExists(atPath: sourcePath) {
            self.sourcePath = sourcePath
            modelSpec = sourcePath
        } else {
            self.sourcePath = nil
            // Existing packed installs predate source retention. The upstream
            // resolver can fetch their exact catalog revision into the Hub cache.
            modelSpec = "\(model.modelID):\(model.revision)"
        }
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
            let fallbackModelSpec = self.modelSpec
            let modelID = self.modelID
            let revision = self.revision
            let quantization = self.quantization
            let sourcePath = self.sourcePath
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
                    let modelSpec = try sourcePath.map {
                        try MLXImageModelSnapshot.prepare(
                            sourcePath: $0,
                            modelID: modelID,
                            revision: revision
                        )
                    } ?? fallbackModelSpec
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
                        switch progress.stage {
                        case .downloadingModel: stage = "downloading_model"
                        case .loadingModel: stage = "loading_model"
                        case .loadingTokenizer: stage = "loading_tokenizer"
                        case .loadingTextEncoder: stage = "loading_text_encoder"
                        case .loadingTransformer: stage = "loading_transformer"
                        case .loadingVAE: stage = "loading_vae"
                        case .encodingText, .loadingLoRA: stage = "text_encoder"
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
                self.producer.clear(token: token)
            }
            self.producer.store(task, token: token)
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
        }
    }
}

private enum MLXImageModelSnapshot {
    private static let lock = NSLock()

    static func prepare(sourcePath: String, modelID: String, revision: String) throws -> String {
        let fm = FileManager.default
        let source = URL(fileURLWithPath: sourcePath).resolvingSymlinksInPath().standardizedFileURL
        let quantizeConfig = source.appendingPathComponent("quantize_config.json")
        let quantizationManifest = source.appendingPathComponent("quantization.json")

        if fm.fileExists(atPath: quantizationManifest.path) {
            return source.path
        }
        // Curated MLX exports may declare quantization in component configs
        // instead of a root manifest. Without a manifest the vendor loader
        // applies packed tensors to dense layers and crashes during denoising.
        let configs = [
            quantizeConfig,
            source.appendingPathComponent("transformer/config.json"),
            source.appendingPathComponent("text_encoder/config.json")
        ]
        var resolved: (bits: Int, groupSize: Int)?
        for config in configs where fm.fileExists(atPath: config.path) {
            let data = try Data(contentsOf: config)
            let document = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let quantization = document?["quantization"] as? [String: Any] else { continue }
            guard let bits = quantization["bits"] as? Int,
                  let groupSize = quantization["group_size"] as? Int,
                  (2...8).contains(bits), groupSize > 0 else {
                throw MLXImageModelSnapshotError.invalidQuantizationConfig(config.path)
            }
            if let resolved, (resolved.bits != bits || resolved.groupSize != groupSize) {
                throw MLXImageModelSnapshotError.invalidQuantizationConfig(config.path)
            }
            resolved = (bits, groupSize)
        }
        guard let resolved else { return source.path }

        for component in ["transformer", "text_encoder", "vae", "tokenizer"] {
            var isDirectory: ObjCBool = false
            let componentPath = source.appendingPathComponent(component, isDirectory: true).path
            guard fm.fileExists(atPath: componentPath, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw MLXImageModelSnapshotError.missingComponent(component, source.path)
            }
        }

        lock.lock()
        defer { lock.unlock() }

        guard let cache = fm.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw MLXImageModelSnapshotError.cacheUnavailable
        }
        let snapshot = cache
            .appendingPathComponent("TurboSpark", isDirectory: true)
            .appendingPathComponent("ImageMLX", isDirectory: true)
            .appendingPathComponent("\(safeComponent(modelID))-\(safeComponent(revision))", isDirectory: true)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)

        for component in ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"] {
            let sourceComponent = source.appendingPathComponent(component, isDirectory: true)
            guard fm.fileExists(atPath: sourceComponent.path) else { continue }
            try stageDirectory(
                sourceComponent,
                to: snapshot.appendingPathComponent(component, isDirectory: true),
                fileManager: fm
            )
        }
        try link(
            source.appendingPathComponent("model_index.json").resolvingSymlinksInPath(),
            to: snapshot.appendingPathComponent("model_index.json"),
            fileManager: fm
        )

        let manifest: [String: Any] = [
            "model_id": modelID,
            "revision": revision,
            "group_size": resolved.groupSize,
            "bits": resolved.bits,
            "mode": "affine",
            "layers": []
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try manifestData.write(to: snapshot.appendingPathComponent("quantization.json"), options: .atomic)
        return snapshot.path
    }

    private static func safeComponent(_ value: String) -> String {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        let result = value.unicodeScalars.map { allowed.contains($0) ? String($0) : "-" }.joined()
        return result.isEmpty ? "model" : result
    }

    private static func stageDirectory(_ source: URL, to destination: URL, fileManager: FileManager) throws {
        if let attributes = try? fileManager.attributesOfItem(atPath: destination.path) {
            if attributes[.type] as? FileAttributeType == .typeSymbolicLink {
                try fileManager.removeItem(at: destination)
            } else if attributes[.type] as? FileAttributeType != .typeDirectory {
                throw MLXImageModelSnapshotError.cachePathOccupied(destination.path)
            }
        }
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        // The upstream weight loader enumerates these directories, so keep them real and link each file.
        for child in try fileManager.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) {
            let resolvedChild = child.resolvingSymlinksInPath().standardizedFileURL
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: resolvedChild.path, isDirectory: &isDirectory) else {
                throw MLXImageModelSnapshotError.missingPath(child.path)
            }
            let destinationChild = destination.appendingPathComponent(child.lastPathComponent)
            if isDirectory.boolValue {
                try stageDirectory(resolvedChild, to: destinationChild, fileManager: fileManager)
            } else {
                try link(resolvedChild, to: destinationChild, fileManager: fileManager)
            }
        }
    }

    private static func link(_ source: URL, to destination: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: source.path) else {
            throw MLXImageModelSnapshotError.missingPath(source.path)
        }
        if let attributes = try? fileManager.attributesOfItem(atPath: destination.path) {
            let resolvedSource = source.resolvingSymlinksInPath().standardizedFileURL
            let resolvedDestination = destination.resolvingSymlinksInPath().standardizedFileURL
            if resolvedSource == resolvedDestination {
                return
            }
            guard attributes[.type] as? FileAttributeType == .typeSymbolicLink else {
                throw MLXImageModelSnapshotError.cachePathOccupied(destination.path)
            }
            try fileManager.removeItem(at: destination)
        }
        try fileManager.createSymbolicLink(at: destination, withDestinationURL: source)
    }
}

private enum MLXImageModelSnapshotError: LocalizedError {
    case cacheUnavailable
    case cachePathOccupied(String)
    case invalidQuantizationConfig(String)
    case missingComponent(String, String)
    case missingPath(String)

    var errorDescription: String? {
        switch self {
        case .cacheUnavailable:
            return "Could not locate the app cache for the MLX image model."
        case .cachePathOccupied(let path):
            return "The MLX model cache path is occupied by a non-symlink: \(path)"
        case .invalidQuantizationConfig(let path):
            return "Could not read the quantization settings from \(path)"
        case .missingComponent(let component, let path):
            return "The MLX model is missing its \(component) component: \(path)"
        case .missingPath(let path):
            return "The MLX model file or component is missing: \(path)"
        }
    }
}
