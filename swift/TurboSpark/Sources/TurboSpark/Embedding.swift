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
