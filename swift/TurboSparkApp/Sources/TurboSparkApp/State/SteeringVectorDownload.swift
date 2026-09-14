import CryptoKit
import Foundation
import TurboSpark

/// A control vector source entered by the user or supplied by a future
/// curated catalog. Only Hugging Face paths are accepted here so a pasted
/// value cannot turn the app into a general-purpose network downloader.
struct SteeringVectorSource: Equatable, Sendable {
    let repo: String
    let file: String
    let revision: String

    var normalizedRepo: String {
        ModelProbeGating.sanitizeRepo(repo)
    }

    var normalizedRevision: String {
        revision.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var normalizedFile: String {
        file.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var identity: String {
        normalizedRepo + "@" + normalizedRevision + "/" + normalizedFile
    }

    var validationError: String? {
        let repoParts = normalizedRepo.split(separator: "/", omittingEmptySubsequences: true)
        guard repoParts.count == 2,
              repoParts.allSatisfy({ isSafePathPart(String($0)) })
        else {
            return "Enter a Hugging Face repository as owner/name."
        }
        guard !normalizedRevision.isEmpty, isSafePathPart(normalizedRevision) else {
            return "Revision must be a branch name or commit without path separators."
        }
        let fileParts = normalizedFile.split(separator: "/", omittingEmptySubsequences: true)
        guard !fileParts.isEmpty,
              fileParts.allSatisfy({ isSafePathPart(String($0)) }),
              normalizedFile.lowercased().hasSuffix(".gguf")
        else {
            return "The vector file must be a .gguf file."
        }
        return nil
    }

    var downloadURL: URL? {
        guard validationError == nil else { return nil }
        let encodedRepo = normalizedRepo.split(separator: "/").map { encodePathPart(String($0)) }.joined(separator: "/")
        let encodedFile = normalizedFile.split(separator: "/").map { encodePathPart(String($0)) }.joined(separator: "/")
        let encodedRevision = encodePathPart(normalizedRevision)
        return URL(string: "https://huggingface.co/" + encodedRepo + "/resolve/"
            + encodedRevision + "/" + encodedFile)
    }

    private func isSafePathPart(_ part: String) -> Bool {
        guard !part.isEmpty, part != ".", part != ".." else { return false }
        return !part.contains("/") && !part.contains("\\") && !part.contains("?") && !part.contains("#")
    }

    private func encodePathPart(_ part: String) -> String {
        part.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? part
    }
}

struct DownloadedSteeringVector: Equatable, Sendable {
    let path: String
    let info: ControlVectorInfo
}

enum SteeringVectorDownloadError: LocalizedError {
    case invalidSource(String)
    case invalidResponse
    case httpStatus(Int)
    case tooLarge
    case shapeMismatch(String)

    var errorDescription: String? {
        switch self {
        case let .invalidSource(message): return message
        case .invalidResponse: return "Hugging Face returned an invalid download response."
        case let .httpStatus(status):
            if status == 401 || status == 403 {
                return "Hugging Face denied this file. Sign in under Settings > Models if it is gated."
            }
            if status == 404 { return "That repository, revision, or .gguf file was not found." }
            return "Hugging Face returned HTTP " + String(status) + "."
        case .tooLarge:
            return "This file is larger than the 32 MB control-vector limit."
        case let .shapeMismatch(message): return message
        }
    }
}

/// Downloads and validates a small control-vector sidecar.
enum SteeringVectorDownloader {
    static let maxBytes = 32 * 1024 * 1024

    static func managedPath(for source: SteeringVectorSource) -> URL {
        let digest = SHA256.hash(data: Data(source.identity.utf8))
            .prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
        let fileName = source.normalizedFile.split(separator: "/").last.map(String.init) ?? "vector.gguf"
        return AppStorageRoot.subdirectory("steering-vectors")
            .appendingPathComponent(digest + "-" + fileName)
    }

    static func download(
        source: SteeringVectorSource,
        expectedHidden: Int?,
        expectedLayers: Int?
    ) async throws -> DownloadedSteeringVector {
        guard let url = source.downloadURL else {
            throw SteeringVectorDownloadError.invalidSource(
                source.validationError ?? "Invalid Hugging Face vector source.")
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 120
        request.setValue("TurboSparkApp/steering-vector", forHTTPHeaderField: "User-Agent")
        do {
            if let token = try TurboSparkCatalog.getHfToken(), !token.isEmpty {
                request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
            }
        } catch {
            // A missing or unreadable optional token must not prevent public
            // vectors from downloading. Hugging Face will report a gated
            // repository through its response status below.
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw SteeringVectorDownloadError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw SteeringVectorDownloadError.httpStatus(http.statusCode)
        }
        if let contentLength = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int.init),
           contentLength > maxBytes {
            throw SteeringVectorDownloadError.tooLarge
        }
        guard data.count <= maxBytes else { throw SteeringVectorDownloadError.tooLarge }

        let directory = AppStorageRoot.subdirectory("steering-vectors")
        let temporary = directory.appendingPathComponent("." + UUID().uuidString + ".partial")
        try data.write(to: temporary, options: .atomic)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let info: ControlVectorInfo
        do {
            info = try TurboSparkCatalog.controlVectorInfo(path: temporary.path)
        } catch {
            throw SteeringVectorDownloadError.invalidSource(
                "The downloaded file is not a control vector this engine can read: "
                    + error.localizedDescription)
        }
        if let expectedHidden, info.hidden != expectedHidden {
            throw SteeringVectorDownloadError.shapeMismatch(
                "This vector is " + String(info.hidden) + " wide, but this model is "
                    + String(expectedHidden) + " wide.")
        }
        if let expectedLayers, info.spannedLayers > expectedLayers {
            throw SteeringVectorDownloadError.shapeMismatch(
                "This vector spans " + String(info.spannedLayers)
                    + " layers, but this model has " + String(expectedLayers) + ".")
        }

        let destination = managedPath(for: source)
        if FileManager.default.fileExists(atPath: destination.path) {
            try? FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
        return DownloadedSteeringVector(path: destination.path, info: info)
    }
}
