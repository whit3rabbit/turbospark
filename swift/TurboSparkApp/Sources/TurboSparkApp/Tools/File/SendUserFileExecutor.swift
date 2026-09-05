import Foundation
import AppKit

/// Executor for sending, revealing, or presenting a generated/saved file to the user.
public enum SendUserFileExecutor {
    public static var onFileSent: (@Sendable (UUID?, URL, String?) -> Void)?

    public static func execute(arguments: [String: String], rootURL: URL, chatID: UUID? = nil) throws -> String {
        guard let relPath = arguments["path"] ?? arguments["file_path"] ?? arguments["file"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 42,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument for SendUserFile."]
            )
        }

        let targetURL = try AppToolRegistry.resolveSecurePath(relPath: relPath, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 42,
                userInfo: [NSLocalizedDescriptionKey: "File not found at '\(relPath)'."]
            )
        }

        let message = arguments["message"] ?? arguments["description"] ?? arguments["title"]
        onFileSent?(chatID, targetURL, message)

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: targetURL.path)[.size] as? Int64) ?? 0
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useAll]
        formatter.countStyle = .file
        let sizeStr = formatter.string(fromByteCount: fileSize)

        var result = "File '\(relPath)' (\(sizeStr)) has been presented to the user."
        if let msg = message, !msg.isEmpty {
            result += "\nNote: \(msg)"
        }
        return result
    }
}
