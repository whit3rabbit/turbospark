import CTurboSpark
import Foundation

/// Standalone embedding model inference and vector similarity calculation.
public enum TurboSparkEmbedding {
    /// Computes dense vector embeddings for an array of input strings using a local
    /// encoder model.
    ///
    /// - Parameters:
    ///   - texts: The text strings to encode.
    ///   - modelPath: Path to an encoder model directory containing `config.json`,
    ///     `model.safetensors`, and `tokenizer.json`, or a catalog alias.
    /// - Returns: An array of normalized float embedding vectors, one per input string.
    public static func encode(texts: [String], modelPath: String) async throws -> [[Float]] {
        try await Task.detached {
            let data = try JSONEncoder().encode(texts)
            guard let textsJSON = String(data: data, encoding: .utf8) else {
                throw TurboSparkError(code: .json, message: "failed to serialize texts to JSON")
            }

            let json = try takeString { out in
                modelPath.withCString { mPtr in
                    textsJSON.withCString { tPtr in
                        ts_embedding_encode_json(mPtr, tPtr, out)
                    }
                }
            }

            guard let jsonData = json.data(using: .utf8) else {
                throw TurboSparkError(code: .json, message: "invalid UTF-8 in embedding result JSON")
            }
            return try JSONDecoder().decode([[Float]].self, from: jsonData)
        }.value
    }

    /// Computes the cosine similarity between two float vectors of equal length.
    /// Both vectors are L2-normalized first, so arbitrary nonzero inputs give a
    /// true cosine in [-1, 1]; a zero vector yields 0.0.
    /// Returns 0.0 if either vector is empty or if lengths do not match.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty, a.count == b.count else { return 0.0 }
        return a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                guard let aPtr = aBuf.baseAddress, let bPtr = bBuf.baseAddress else { return 0.0 }
                return ts_cosine_similarity(aPtr, bPtr, a.count)
            }
        }
    }

    /// Ranks an array of candidate document strings by semantic cosine similarity against a query.
    ///
    /// - Parameters:
    ///   - query: The query string to compare against.
    ///   - documents: The candidate text strings to rank.
    ///   - modelPath: Path or alias of the encoder model.
    /// - Returns: An array of tuples sorted in descending order of similarity score.
    public static func rank(
        query: String,
        documents: [String],
        modelPath: String
    ) async throws -> [(index: Int, text: String, score: Float)] {
        guard !documents.isEmpty else { return [] }
        var allTexts = [query]
        allTexts.append(contentsOf: documents)
        let embeddings = try await encode(texts: allTexts, modelPath: modelPath)
        guard embeddings.count == allTexts.count, let queryVector = embeddings.first else {
            return []
        }

        var results: [(index: Int, text: String, score: Float)] = []
        for (i, doc) in documents.enumerated() {
            let docVector = embeddings[i + 1]
            let score = cosineSimilarity(queryVector, docVector)
            results.append((index: i, text: doc, score: score))
        }
        results.sort { $0.score > $1.score }
        return results
    }

    /// Computes embeddings for texts (with an optional task prompt and
    /// Matryoshka size), image files and audio files in one call. Task,
    /// images and audio need a model with those prompts and towers
    /// (EmbeddingGemma 2 on Core ML, macOS 15+); other models throw.
    public static func encode(request: EmbeddingRequest, modelPath: String) async throws -> EmbeddingResult {
        try await Task.detached {
            let data = try JSONEncoder().encode(request)
            guard let requestJSON = String(data: data, encoding: .utf8) else {
                throw TurboSparkError(code: .json, message: "failed to serialize embedding request")
            }
            let json = try takeString { out in
                modelPath.withCString { mPtr in
                    requestJSON.withCString { rPtr in
                        ts_embedding_encode_request_json(mPtr, rPtr, out)
                    }
                }
            }
            guard let jsonData = json.data(using: .utf8) else {
                throw TurboSparkError(code: .json, message: "invalid UTF-8 in embedding result JSON")
            }
            return try JSONDecoder().decode(EmbeddingResult.self, from: jsonData)
        }.value
    }

    /// Computes a single normalized embedding vector for an input text.
    public static func encode(text: String, modelPath: String) async throws -> [Float] {
        let batch = try await encode(texts: [text], modelPath: modelPath)
        guard let first = batch.first else {
            throw TurboSparkError(code: .generate, message: "empty embedding result")
        }
        return first
    }

    /// Ranks documents by cosine similarity to a query, returning at most `k` highest scoring candidates.
    public static func topK(
        query: String,
        documents: [String],
        k: Int,
        modelPath: String
    ) async throws -> [(index: Int, text: String, score: Float)] {
        let ranked = try await rank(query: query, documents: documents, modelPath: modelPath)
        return Array(ranked.prefix(max(0, k)))
    }
}

/// One `TurboSparkEmbedding.encode(request:modelPath:)` call. Every field is
/// optional; an empty request returns empty lists.
public struct EmbeddingRequest: Encodable, Sendable, Equatable {
    public var texts: [String]
    /// A trained task prompt such as `search_query` or `document`.
    public var task: String?
    /// A Matryoshka size the model supports (768, 512, 256 or 128 for
    /// EmbeddingGemma 2).
    public var dimensions: Int?
    public var imagePaths: [String]
    /// Soft tokens per image (70, 140 or 280 for EmbeddingGemma 2).
    public var imageBudget: Int?
    /// WAV files; each is embedded as consecutive 10 s windows.
    public var audioPaths: [String]

    public init(
        texts: [String] = [], task: String? = nil, dimensions: Int? = nil,
        imagePaths: [String] = [], imageBudget: Int? = nil, audioPaths: [String] = []
    ) {
        self.texts = texts; self.task = task; self.dimensions = dimensions
        self.imagePaths = imagePaths; self.imageBudget = imageBudget; self.audioPaths = audioPaths
    }

    enum CodingKeys: String, CodingKey {
        case texts, task, dimensions
        case imagePaths = "image_paths"
        case imageBudget = "image_budget"
        case audioPaths = "audio_paths"
    }
}

/// One embedded window of an audio file.
public struct AudioWindowEmbedding: Decodable, Sendable, Equatable {
    public let startSeconds: Double
    public let durationSeconds: Double
    public let embedding: [Float]

    enum CodingKeys: String, CodingKey {
        case startSeconds = "start_seconds"
        case durationSeconds = "duration_seconds"
        case embedding
    }
}

/// Embeddings in request order: one per text, one per image, and one
/// window list per audio file.
public struct EmbeddingResult: Decodable, Sendable, Equatable {
    public let texts: [[Float]]
    public let images: [[Float]]
    public let audio: [[AudioWindowEmbedding]]
}

/// One pinned embedding model the app can download (`ts_embedding_catalog_json`).
public struct EmbeddingProfile: Decodable, Sendable, Equatable, Identifiable {
    public var id: String { alias }
    public let alias: String
    public let displayName: String
    public let family: String
    public let backend: String
    public let repository: String
    public let revision: String
    public let license: String
    /// Inputs embedded into one space: `text`, `image`, `audio`.
    public let modalities: [String]
    public let dimensions: Int
    public let matryoshkaDims: [Int]
    /// `verified`, `runs` or `caveat`. Evidence, not intent: `verified`
    /// means the tests named in `gates` ran a real install of this pin.
    public let status: String
    public let gates: [String]
    public let downloadBytes: UInt64
    /// Set once an install verifies against the current pin. The alias
    /// itself also works as an embedding model path.
    public let installedPath: String?
    /// `current`, `stale` (an older pin or damaged; Download replaces it),
    /// `foreign` (Download refuses it) or `absent`.
    public let installState: String
    /// Why there is no verified install, in words.
    public let installProblem: String?
    public let canRun: Bool
    public let unavailableReason: String?

    public var isVerified: Bool { status == "verified" }

    enum CodingKeys: String, CodingKey {
        case alias, family, backend, repository, revision, license, modalities, dimensions
        case status, gates
        case displayName = "display_name"
        case matryoshkaDims = "matryoshka_dims"
        case downloadBytes = "download_bytes"
        case installedPath = "installed_path"
        case installState = "install_state"
        case installProblem = "install_problem"
        case canRun = "can_run"
        case unavailableReason = "unavailable_reason"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        alias = try c.decode(String.self, forKey: .alias)
        displayName = try c.decode(String.self, forKey: .displayName)
        family = try c.decode(String.self, forKey: .family)
        backend = try c.decode(String.self, forKey: .backend)
        repository = try c.decode(String.self, forKey: .repository)
        revision = try c.decode(String.self, forKey: .revision)
        license = try c.decode(String.self, forKey: .license)
        modalities = try c.decode([String].self, forKey: .modalities)
        dimensions = try c.decode(Int.self, forKey: .dimensions)
        matryoshkaDims = try c.decode([Int].self, forKey: .matryoshkaDims)
        // A row without a status was never shown as tested.
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? "runs"
        gates = try c.decodeIfPresent([String].self, forKey: .gates) ?? []
        downloadBytes = try c.decode(UInt64.self, forKey: .downloadBytes)
        installedPath = try c.decodeIfPresent(String.self, forKey: .installedPath)
        installState = try c.decodeIfPresent(String.self, forKey: .installState)
            ?? (installedPath == nil ? "absent" : "current")
        installProblem = try c.decodeIfPresent(String.self, forKey: .installProblem)
        canRun = try c.decode(Bool.self, forKey: .canRun)
        unavailableReason = try c.decodeIfPresent(String.self, forKey: .unavailableReason)
    }
}

/// The verdict for one embedding model argument (`ts_embedding_probe_json`),
/// computed without loading the model. Every embedding open and server
/// attach applies the same check.
public struct EmbeddingProbe: Decodable, Sendable, Equatable {
    public let input: String
    public let resolvedPath: String?
    public let runnable: Bool
    /// `embeddinggemma2`, `bert`, `xlm-roberta`, or the refused family.
    public let family: String?
    /// `coreml` or `encoder`.
    public let backend: String?
    /// The verified install of a catalog profile whose status is verified.
    public let verified: Bool
    public let profileAlias: String?
    public let refusedBecause: String?

    enum CodingKeys: String, CodingKey {
        case input, runnable, family, backend, verified
        case resolvedPath = "resolved_path"
        case profileAlias = "profile_alias"
        case refusedBecause = "refused_because"
    }
}

/// Pinned embedding models: listing and verified install.
public enum EmbeddingCatalog {
    public static func profiles() throws -> [EmbeddingProfile] {
        let json = try takeString { ts_embedding_catalog_json($0) }
        return try JSONDecoder().decode([EmbeddingProfile].self, from: Data(json.utf8))
    }

    /// The family verdict for a pinned alias or a model directory, without
    /// loading it. Cheap (a few file reads); safe to call per keystroke
    /// after a debounce, but off the main actor.
    public static func probe(_ modelPath: String) throws -> EmbeddingProbe {
        let path = NSString(string: modelPath).expandingTildeInPath
        let json = try takeString { out in
            path.withCString { ts_embedding_probe_json($0, out) }
        }
        return try JSONDecoder().decode(EmbeddingProbe.self, from: Data(json.utf8))
    }

    /// Blocking download and install; call off the main actor. `revision`
    /// is the pin the caller showed (`EmbeddingProfile.revision`): Rust
    /// refuses any other, so a retry never adopts a different pin. Cancel
    /// with `TurboSparkCatalog.cancelInstall()`. Returns the install path.
    public static func install(
        alias: String,
        revision: String,
        progress: @escaping @Sendable (AudioDownloadProgress) -> Void = { _ in }
    ) throws -> String {
        try runPinnedInstall(progress: progress) { callback, context, out in
            alias.withCString { a in
                revision.withCString { r in ts_embedding_install(a, r, callback, context, out) }
            }
        }
    }
}

/// Holds a pinned install's Swift progress closure for the C callback.
final class PinnedDownloadBox {
    let callback: @Sendable (AudioDownloadProgress) -> Void
    init(_ callback: @escaping @Sendable (AudioDownloadProgress) -> Void) { self.callback = callback }
}

/// Runs one pinned-install C call (`ts_embedding_install`,
/// `ts_ocr_install`) with byte progress forwarded to `progress`, and returns
/// the install path from its `{"alias", "path"}` result.
func runPinnedInstall(
    progress: @escaping @Sendable (AudioDownloadProgress) -> Void,
    _ call: (
        TsInstallCallback, UnsafeMutableRawPointer, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
    ) -> Int32
) throws -> String {
    let box = PinnedDownloadBox(progress)
    let context = Unmanaged.passUnretained(box).toOpaque()
    defer { withExtendedLifetime(box) {} }
    let callback: TsInstallCallback = { userdata, _, path, length, completed, total in
        guard let userdata else { return }
        let box = Unmanaged<PinnedDownloadBox>.fromOpaque(userdata).takeUnretainedValue()
        let name = path.map {
            String(decoding: UnsafeBufferPointer(
                start: UnsafeRawPointer($0).assumingMemoryBound(to: UInt8.self), count: length),
                as: UTF8.self)
        }
        box.callback(AudioDownloadProgress(path: name, completedBytes: completed, totalBytes: total))
    }
    let json = try takeString { out in call(callback, context, out) }
    struct Installed: Decodable { let path: String }
    return try JSONDecoder().decode(Installed.self, from: Data(json.utf8)).path
}
