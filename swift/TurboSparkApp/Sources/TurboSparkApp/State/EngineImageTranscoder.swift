import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Makes an attachment readable by the engine's image decoder.
///
/// `crates/vision-io` enables the `jpeg` and `png` decoders only, and applies
/// no EXIF orientation. The attachment picker accepts more than that (gif,
/// heic, tiff, bmp, webp), and a phone JPEG is routinely stored sideways with
/// an orientation tag. Handing those to the engine as-is either fails the turn
/// with a decode error or sends the model a rotated picture, which it answers
/// fluently and wrongly.
///
/// So the engine is given a PNG for anything it cannot read faithfully. The
/// stored attachment is never modified; managed derivatives live beside the
/// profile's decrypted attachments and are removed with their source or when
/// that cache is purged.
enum EngineImageTranscoder {
    /// Used only for non-managed paths when there is no unlocked profile.
    /// This process-unique temporary directory is purged on app lifecycle
    /// boundaries and is never placed under persistent Application Support.
    private static let transientCacheDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("TurboSpark-ImageTranscoder-\(UUID().uuidString)", isDirectory: true)

    /// A protected per-profile cache when possible, with a process-private
    /// temporary cache for standalone paths that have no vault session.
    static var defaultCacheDirectory: URL {
        ManagedAssetStore.shared.imageTranscoderCacheDirectory() ?? transientCacheDirectory
    }

    /// The path to hand the engine for `path`: the same path when it is
    /// already faithful, a cached PNG when not, and the original path again if
    /// anything fails, so the engine reports its own (named) decode error
    /// rather than this layer inventing one.
    static func enginePath(
        for path: String,
        cacheDirectory: URL? = nil,
        storedReference: String? = nil
    ) -> String {
        // Remote URLs and empty (unresolvable) paths are not ours to rewrite.
        guard path.hasPrefix("/"),
            let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
            let type = CGImageSourceGetType(source) as String?
        else { return path }

        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? Int) ?? 1
        guard needsTranscode(type: type, orientation: orientation) else { return path }

        let cache = cacheDirectory ?? defaultCacheDirectory
        let name = cacheName(for: path, storedReference: storedReference)
        let output = cache.appendingPathComponent(name + ".png")

        let width = (properties?[kCGImagePropertyPixelWidth] as? Int) ?? 0
        let height = (properties?[kCGImagePropertyPixelHeight] as? Int) ?? 0
        // Never upscale: the cap is the source's own long edge.
        let longEdge = max(width, height) > 0 ? max(width, height) : 8192
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: longEdge,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        else { return path }

        do {
            try FileManager.default.createDirectory(
                at: cache, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cache.path)
            if FileManager.default.fileExists(atPath: output.path) {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o600], ofItemAtPath: output.path)
                return output.path
            }
            // Written beside the destination and moved in, so a concurrent
            // reader never sees a half-written PNG. Create it with private
            // permissions before ImageIO writes any pixels.
            let staging = cache.appendingPathComponent(".\(UUID().uuidString).png")
            defer { try? FileManager.default.removeItem(at: staging) }
            guard FileManager.default.createFile(
                atPath: staging.path, contents: nil, attributes: [.posixPermissions: 0o600])
            else { return path }
            guard
                let destination = CGImageDestinationCreateWithURL(
                    staging as CFURL, UTType.png.identifier as CFString, 1, nil)
            else {
                try? FileManager.default.removeItem(at: staging)
                return path
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                try? FileManager.default.removeItem(at: staging)
                return path
            }
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: staging.path)
            if FileManager.default.fileExists(atPath: output.path) {
                try? FileManager.default.removeItem(at: staging)
            } else {
                try FileManager.default.moveItem(at: staging, to: output)
            }
            return output.path
        } catch {
            // A failed write must not strand a partially decoded attachment.
            return path
        }
    }

    /// Clears converted images when a chat is deleted. They are disposable
    /// and are rebuilt from their still-encrypted sources if another chat
    /// needs them later.
    static func purgeCache() {
        ManagedAssetStore.shared.purgeImageTranscoderCache()
        purgeTransientCache()
    }

    static func purgeTransientCache() {
        try? FileManager.default.removeItem(at: transientCacheDirectory)
    }

    /// Removes plaintext PNGs created by versions that cached under
    /// Application Support. Run for all known profiles at launch and for the
    /// selected profile whenever its decrypted cache is purged.
    static func purgeLegacyPersistentCaches() {
        purgeLegacyPersistentCache(profileID: UserProfileStore.defaultProfileID)
        for profile in UserProfileStore.loadRegistry().profiles {
            purgeLegacyPersistentCache(profileID: profile.id)
        }
    }

    static func purgeLegacyPersistentCache(profileID: String) {
        let root = UserProfileStore.storeDirectory(
            profileID: profileID, machineRoot: AppStorageRoot.machineRoot) ?? AppStorageRoot.machineRoot
        try? FileManager.default.removeItem(
            at: root.appendingPathComponent("EngineImageCache", isDirectory: true))
    }

    /// PNG is read as-is (the engine's decoder ignores a PNG's rare
    /// orientation chunk; transcoding would not change that). JPEG is read
    /// as-is unless it carries a non-identity orientation. Everything else is
    /// converted.
    static func needsTranscode(type: String, orientation: Int) -> Bool {
        switch type {
        case UTType.png.identifier: return false
        case UTType.jpeg.identifier: return orientation > 1
        default: return true
        }
    }

    private static func cacheName(for path: String, storedReference: String?) -> String {
        if let id = storedReference.flatMap(ManagedAssetStore.assetID(from:)) { return id }
        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "\(path)|\(size)|\(modified)"
        return SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }
            .joined()
    }
}
