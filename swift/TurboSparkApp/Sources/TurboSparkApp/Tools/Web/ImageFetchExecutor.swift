import Foundation

/// The `image_fetch` tool: downloads one raster image into the project.
///
/// WebFetch refuses image content types, and image bytes must not enter the
/// model's context, so this is the only path that turns an `image_search` row
/// into a file. It shares WebFetch's transport (destination validator,
/// per-hop redirect revalidation, rebinding check) and adds what a download
/// to disk needs: a byte cap, a magic-byte check (the Content-Type header is
/// not trusted), and a destination that stays inside the project root.
public enum ImageFetchExecutor {
    public static let maxImageBytes = 10 * 1024 * 1024
    public static let defaultTimeoutSeconds = 30
    public static let maxTimeoutSeconds = 120

    public enum ImageKind: String, Sendable {
        case png, jpeg, gif, webp

        /// Extensions a saved file may carry for this kind.
        var extensions: [String] {
            switch self {
            case .png: return ["png"]
            case .jpeg: return ["jpg", "jpeg"]
            case .gif: return ["gif"]
            case .webp: return ["webp"]
            }
        }

        /// SVG is deliberately absent: it is XML that can carry script.
        static func sniff(_ data: Data) -> ImageKind? {
            let b = [UInt8](data.prefix(12))
            if b.count >= 8, Array(b[0..<8]) == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] { return .png }
            if b.count >= 3, Array(b[0..<3]) == [0xFF, 0xD8, 0xFF] { return .jpeg }
            if b.count >= 6, Array(b[0..<4]) == [0x47, 0x49, 0x46, 0x38], b[4] == 0x37 || b[4] == 0x39,
               b[5] == 0x61 { return .gif }
            if b.count >= 12, Array(b[0..<4]) == [0x52, 0x49, 0x46, 0x46],
               Array(b[8..<12]) == [0x57, 0x45, 0x42, 0x50] { return .webp }
            return nil
        }
    }

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "TurboSparkImageFetch", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// The destination under the project root, validated before any network
    /// request is made so a bad path never costs a download.
    static func resolveDestination(path: String, rootURL: URL) throws -> URL {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw error(10, "image_fetch needs a 'path' to save the image to.") }
        let target = try AppToolRegistry.resolveSecurePath(relPath: trimmed, rootURL: rootURL)
        guard target.path != rootURL.standardizedFileURL.resolvingSymlinksInPath().path else {
            throw error(11, "The destination must be a file path inside the project.")
        }
        if ToolRiskClassifier.isSensitivePath(trimmed) || ToolRiskClassifier.isSensitivePath(target.path) {
            throw error(12, "Refusing to write an image to a sensitive path: '\(trimmed)'.")
        }
        let ext = target.pathExtension.lowercased()
        guard ImageKind.allCases.contains(where: { $0.extensions.contains(ext) }) else {
            throw error(13, "The destination must end in .png, .jpg, .jpeg, .gif or .webp: '\(trimmed)'.")
        }
        if FileManager.default.fileExists(atPath: target.path) {
            throw error(14, "'\(trimmed)' already exists. image_fetch never overwrites; choose a new path.")
        }
        return target
    }

    /// Fetches `url` and writes it to `path` (relative to `rootURL`).
    /// Returns a short report for the model; it never contains image bytes.
    public static func fetch(
        url urlString: String,
        path: String,
        sourcePage: String? = nil,
        rootURL: URL,
        timeout: Int? = nil,
        customSession: URLSession? = nil,
        destinationValidator injectedValidator: ((URL) throws -> Void)? = nil,
        connectedAddressValidator injectedConnected: ((String) throws -> Void)? = nil
    ) async throws -> String {
        let destinationValidator = injectedValidator ?? { try HttpRequestDestinationValidator.validate($0) }
        let connectedValidator: ((String) throws -> Void)? = injectedConnected
            ?? (injectedValidator == nil ? HttpRequestExecutor.refuseConnectedPrivateAddress : nil)

        let target = try resolveDestination(path: path, rootURL: rootURL)

        let trimmedURL = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmedURL), let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https", let host = url.host, !host.isEmpty
        else {
            throw error(1, "URL must be an http:// or https:// image URL: '\(urlString)'")
        }
        guard url.user == nil, url.password == nil else {
            throw error(1, "Image URLs with embedded credentials are refused.")
        }
        try destinationValidator(url)

        let seconds = min(maxTimeoutSeconds, max(1, timeout ?? defaultTimeoutSeconds))
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = TimeInterval(seconds)
        request.setValue(WebSearchExecutor.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("image/png,image/jpeg,image/gif,image/webp", forHTTPHeaderField: "Accept")
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = TimeInterval(seconds)
        config.timeoutIntervalForResource = TimeInterval(seconds)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await HttpRequestExecutor.performRequest(
                request, configuration: config, customSession: customSession,
                validate: destinationValidator, validateConnectedAddress: connectedValidator,
                maxBytes: maxImageBytes)
        } catch let failure as NSError where failure.domain == "TurboSparkHttpRequest"
            && failure.localizedDescription.hasPrefix("Response too large") {
            throw error(2, "The image is larger than the \(maxImageBytes / (1024 * 1024)) MiB limit.")
        }
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw error(3, "Image download failed with HTTP \(status).")
        }
        guard data.count <= maxImageBytes else {
            throw error(2, "The image is larger than the \(maxImageBytes / (1024 * 1024)) MiB limit.")
        }
        guard !data.isEmpty, let kind = ImageKind.sniff(data) else {
            throw error(4, "The response is not a PNG, JPEG, GIF or WebP image, so nothing was saved.")
        }
        let ext = target.pathExtension.lowercased()
        guard kind.extensions.contains(ext) else {
            throw error(5, "The image is \(kind.rawValue) but the path ends in .\(ext); use .\(kind.extensions[0]).")
        }

        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        // `.withoutOverwriting` closes the race between the existence check
        // above and the write.
        try data.write(to: target, options: [.withoutOverwriting])

        var lines = [
            "Saved \(kind.rawValue) image (\(data.count) bytes) to \(path.trimmingCharacters(in: .whitespacesAndNewlines)).",
            "Downloaded from: \(http.url?.absoluteString ?? trimmedURL)",
        ]
        if let page = sourcePage?.trimmingCharacters(in: .whitespacesAndNewlines), !page.isEmpty {
            lines.append("Source page (cite this): \(page)")
        } else {
            lines.append("Cite the source page from the image_search row where this image is used.")
        }
        return lines.joined(separator: "\n")
    }
}

extension ImageFetchExecutor.ImageKind: CaseIterable {}
