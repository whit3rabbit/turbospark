import AppKit
import QuickLookThumbnailing
import SwiftUI

/// Path-keyed thumbnail cache behind the attachment chips and the Files list.
///
/// This is the app's FIRST cached, off-main image path. Every other image site
/// here is a synchronous full-resolution `NSImage(contentsOf:)` inside a view
/// body, which is affordable for a single preview pane and is not affordable in
/// a list: `swift/CLAUDE.md` Gotcha 16 has the transcript re-evaluating per
/// streamed token, so an uncached decode in a row is a decode per token.
/// `ModelLogoAssetCache` is the only precedent and is bundle-resource-only.
///
/// QuickLookThumbnailing rather than `CGImageSource` because one code path then
/// covers an image, a PDF's first page, and anything else a future caller
/// wants, at the cost of being asynchronous.
@MainActor
final class AttachmentThumbnailStore {
    static let shared = AttachmentThumbnailStore()

    private let cache = NSCache<NSString, NSImage>()
    // NSCache cannot hold nil, so a file the generator cannot render needs its
    // own record or every appearance of that row retries it.
    private var failedKeys: Set<String> = []

    private init() {
        cache.countLimit = 256
    }

    /// The cache identity of one rendering.
    ///
    /// A pure function so the invalidation rule is testable without a view
    /// (`swift/CLAUDE.md` Gotcha 26). The modification date is in the key
    /// because a rewritten file at the same path is a different picture, and
    /// the pixel size is in it because the chip asks for 16 where the Files
    /// row asks for 28 and neither may serve the other's answer.
    /// `nonisolated` because it is pure. Inheriting the type's `@MainActor`
    /// would make it implicitly async at every call site, including a
    /// synchronous test, which is the whole reason it was split out.
    nonisolated static func cacheKey(
        path: String, modified: Date?, pixelSize: Int
    ) -> String {
        let stamp = modified.map { String($0.timeIntervalSince1970) } ?? "none"
        return "\(path)|\(stamp)|\(pixelSize)"
    }

    /// A thumbnail for `url`, or nil when the file cannot produce one.
    ///
    /// Nil is a normal answer, not an error: every caller renders an SF Symbol
    /// until this resolves and keeps it if the answer is nil, so a failure here
    /// looks exactly like the app did before thumbnails existed.
    func thumbnail(for url: URL, pixelSize: CGFloat) async -> NSImage? {
        let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            .contentModificationDate
        let key = Self.cacheKey(
            path: url.path, modified: modified, pixelSize: Int(pixelSize))

        if let cached = cache.object(forKey: key as NSString) { return cached }
        if failedKeys.contains(key) { return nil }

        let scale = NSScreen.main?.backingScaleFactor ?? 2
        // **NOT `.all`, which includes `.icon`.** With the icon type enabled
        // the generator never fails: it answers a generic 32x32 document glyph
        // for a file it cannot render, and for a file that does not exist at
        // all -- measured, not assumed. That is strictly worse than what it
        // would replace, because the caller's fallback is a THEMED SF Symbol
        // that tracks the accent color and the type scale. Asking only for a
        // real thumbnail makes nil mean nil.
        let request = QLThumbnailGenerator.Request(
            fileAt: url,
            size: CGSize(width: pixelSize, height: pixelSize),
            scale: scale,
            representationTypes: [.thumbnail, .lowQualityThumbnail])

        let box: ThumbnailBox = await withCheckedContinuation { continuation in
            QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { rep, _ in
                continuation.resume(returning: ThumbnailBox(image: rep?.nsImage))
            }
        }

        guard let image = box.image else {
            failedKeys.insert(key)
            return nil
        }
        cache.setObject(image, forKey: key as NSString)
        return image
    }
}

/// Carries an `NSImage` (not `Sendable`) back across the generator's completion
/// handler. Sound because the value is created inside the handler and read once
/// on the main actor, with no other reference to it anywhere.
private struct ThumbnailBox: @unchecked Sendable {
    let image: NSImage?
}

/// An SF Symbol that upgrades itself to a real thumbnail when one is available.
///
/// The symbol renders IMMEDIATELY and is what remains if the generator fails or
/// `url` is nil, so adopting this view cannot regress a surface: the worst case
/// is exactly the icon that was there before.
@MainActor
struct AttachmentThumbnailView: View {
    let url: URL?
    let pixelSize: CGFloat
    let fallbackSymbol: String
    var symbolStep: AppFontStep = .callout
    var symbolTint: Color = .secondary

    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
            } else {
                Image(systemName: fallbackSymbol)
                    .themedFont(symbolStep)
                    .foregroundStyle(symbolTint)
            }
        }
        .frame(width: pixelSize, height: pixelSize)
        .task(id: url?.path) { await load() }
    }

    private func load() async {
        guard let url else {
            image = nil
            return
        }
        let loaded = await AttachmentThumbnailStore.shared.thumbnail(
            for: url, pixelSize: pixelSize)
        guard !Task.isCancelled else { return }
        // A crossfade rather than a slide, which is what reduce-motion asks
        // for in the first place, so this needs no accessibility guard.
        withAnimation(.easeIn(duration: 0.15)) { image = loaded }
    }
}
