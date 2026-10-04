import Foundation
import TurboSpark

public struct AppToolMediaHistoryProjection: Equatable, Sendable {
    public let images: [ChatImage]
    public let omittedReferenceCount: Int

    public init(images: [ChatImage], omittedReferenceCount: Int) {
        self.images = images
        self.omittedReferenceCount = omittedReferenceCount
    }
}

/// Resolves persisted media references only for providers that accept image-bearing tool results.
public enum AppToolMediaHistoryAdapter {
    public static func project(
        _ references: [AppToolMediaReference]?,
        capability: AppToolMediaCapability,
        materialize: (String) throws -> URL?
    ) -> AppToolMediaHistoryProjection {
        guard let references, !references.isEmpty else {
            return AppToolMediaHistoryProjection(images: [], omittedReferenceCount: 0)
        }
        guard capability.supportsImageBearingToolResults else {
            return AppToolMediaHistoryProjection(images: [], omittedReferenceCount: references.count)
        }

        var images: [ChatImage] = []
        for reference in references {
            let fitsPixelLimits: Bool = {
                guard reference.pixelWidth > 0, reference.pixelHeight > 0 else { return false }
                return reference.pixelWidth <= AppToolMediaPolicy.applicationMaximumPixelDimension
                    && reference.pixelHeight <= AppToolMediaPolicy.applicationMaximumPixelDimension
                    && reference.pixelWidth <= capability.maximumPixelCount / reference.pixelHeight
            }()
            guard reference.mimeType == "image/png",
                  reference.byteCount > 0,
                  reference.byteCount <= Int64(min(
                    capability.maximumEncodedByteCount,
                    AppToolMediaPolicy.applicationMaximumEncodedByteCount)),
                  fitsPixelLimits,
                  ManagedAssetStore.assetID(from: reference.assetReference) != nil,
                  let url = try? materialize(reference.assetReference)
            else {
                continue
            }
            images.append(.path(url.path))
        }
        return AppToolMediaHistoryProjection(
            images: images,
            omittedReferenceCount: references.count - images.count)
    }
}
