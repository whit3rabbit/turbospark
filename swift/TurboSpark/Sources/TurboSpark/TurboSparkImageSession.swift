import CTurboSpark
import Foundation

/// A serialized native image-generation session.
///
/// The image backend has an explicit byte-buffer ownership contract at the C
/// boundary. This wrapper copies the PNG into `Data` before returning the C
/// buffer, so Swift callers never retain Rust-owned memory.
public final class TurboSparkImageSession: @unchecked Sendable {
    private static let queueKey = DispatchSpecificKey<Void>()

    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle
    private let queue: DispatchQueue

    public init(modelPath: String) async throws {
        let modelPath = NSString(string: modelPath).expandingTildeInPath
        let queue = DispatchQueue(label: "com.turbospark.image-session", qos: .userInitiated)
        queue.setSpecific(key: Self.queueKey, value: ())
        let handle: OpaquePointer = try await withCheckedThrowingContinuation { continuation in
            queue.async {
                var out: OpaquePointer?
                let status = modelPath.withCString { path in
                    ts_image_session_open(path, &out)
                }
                guard status == 0, let out else {
                    continuation.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                continuation.resume(returning: out)
            }
        }
        self.handle = Handle(raw: handle)
        self.queue = queue
    }

    deinit {
        // The C ABI requires close after generation leaves the native queue.
        // Shutdown requests cancellation and then releases this object, so
        // serialize the close behind any queued or in-flight generation.
        if DispatchQueue.getSpecific(key: Self.queueKey) != nil {
            ts_image_session_close(handle.raw)
        } else {
            queue.sync {
                ts_image_session_close(handle.raw)
            }
        }
    }

    /// Stop the current image job without waiting for the worker queue.
    public func cancel() {
        ts_image_session_cancel(handle.raw)
    }

    /// Generates one image and streams stage progress.
    public func generate(
        _ options: ImageGenerateOptions
    ) -> AsyncThrowingStream<ImageGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let optionsJSON: String
            do {
                let data = try JSONEncoder().encode(options)
                guard let json = String(data: data, encoding: .utf8) else {
                    throw TurboSparkError(code: .json, message: "image options were not UTF-8")
                }
                optionsJSON = json
            } catch {
                continuation.finish(throwing: error)
                return
            }

            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }

            queue.async { [self] in
                let box = ImageStreamBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<ImageStreamBox>.fromOpaque(userdata).release() }

                var png: UnsafeMutablePointer<UInt8>?
                var pngLength = 0
                var metadataJSON: UnsafeMutablePointer<CChar>?
                let status = optionsJSON.withCString { options in
                    ts_image_generate(
                        handle.raw,
                        options,
                        imageStreamCallback,
                        userdata,
                        &png,
                        &pngLength,
                        &metadataJSON
                    )
                }
                defer {
                    if let png { ts_image_buffer_free(png, pngLength) }
                    if let metadataJSON { ts_string_free(metadataJSON) }
                }
                guard status == 0 else {
                    continuation.finish(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                guard let metadataJSON else {
                    continuation.finish(throwing: TurboSparkError(
                        code: .unknown,
                        message: "image generation returned no metadata"
                    ))
                    return
                }
                do {
                    let info = try decode(
                        ImageGenerationWireResult.self,
                        from: String(cString: metadataJSON)
                    )
                    switch info.status {
                    case "cancelled":
                        continuation.yield(.cancelled)
                        continuation.finish()
                    case "completed":
                        guard let metadata = info.metadata, let png else {
                            throw TurboSparkError(code: .json, message: "completed image had no PNG")
                        }
                        let result = ImageGenerationResult(
                            png: Data(bytes: png, count: pngLength),
                            metadata: metadata
                        )
                        continuation.yield(.finished(result))
                        continuation.finish()
                    default:
                        throw TurboSparkError(
                            code: .generate,
                            message: info.error ?? "unknown image generation status"
                        )
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

private struct ImageGenerationWireResult: Decodable {
    let status: String
    let error: String?
    let metadata: GeneratedImageMetadata?
}

private final class ImageStreamBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<ImageGenerationEvent, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<ImageGenerationEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

private let imageStreamCallback: TsImageEventCallback = {
    userdata, kind, text, length, completed, total in
    guard let userdata else { return }
    let box = Unmanaged<ImageStreamBox>.fromOpaque(userdata).takeUnretainedValue()
    let name: String
    if let text, length > 0 {
        let bytes = UnsafeBufferPointer(start: text, count: length).map { UInt8(bitPattern: $0) }
        name = String(decoding: bytes, as: UTF8.self)
    } else {
        name = ""
    }
    if kind == TS_IMAGE_EVENT_STAGE {
        box.continuation.yield(.stage(
            name: name,
            completed: Int(completed),
            total: Int(total)
        ))
    }
}
