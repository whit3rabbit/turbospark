import AppKit
import SwiftUI

/// Vendored ghost glyph, standing in for `Image(systemName: "ghost")`.
///
/// `"ghost"` is not, and has never been, a real SF Symbol name: it renders
/// as a blank glyph rather than an error, which is what made ghost mode's
/// toolbar button, sidebar badge and composer banner icon all silently
/// invisible. `ghost.svg` (Lucide, ISC licensed, see `NOTICE`) is loaded
/// once and cached, matching `TaskProgressAssetCache`'s pattern in this same
/// directory.
public final class GhostGlyphAssetCache {
    public static let shared = GhostGlyphAssetCache()
    private var cachedImage: NSImage?

    private init() {
        cachedImage = Self.load()
    }

    public var image: NSImage? {
        if cachedImage == nil {
            cachedImage = Self.load()
        }
        return cachedImage
    }

    private static func load() -> NSImage? {
        // SwiftPM's `.process("Resources")` rule flattens every subdirectory
        // to the bundle root (confirmed against the built app: `ghost.svg`
        // lands beside the `Logos/` brand SVGs, not under `Icons/`), matching
        // `TaskProgressAssetCache`'s lookup one file over, which passes no
        // `subdirectory:` either.
        Bundle.module.url(forResource: "ghost", withExtension: "svg")
            .flatMap { NSImage(contentsOf: $0) }
    }
}

/// A tintable ghost glyph, sized like a system symbol via an explicit point
/// size rather than an ambient font modifier (a custom image cannot inherit
/// one). Falls back to a real SF Symbol if the bundled asset fails to load,
/// so a load failure degrades to a visible icon instead of repeating the
/// bug this type exists to fix.
///
/// Colored via an explicit `color:` rather than the ambient foreground
/// style: `Image(nsImage:).renderingMode(.template)` did not pick up
/// `.foregroundStyle()` reliably for this SVG-backed image (confirmed
/// against a real build: the toolbar toggle stayed the inactive gray after
/// entering ghost mode). Masking a filled rectangle with the glyph's own
/// alpha channel tints correctly regardless of that ambiguity.
public struct GhostGlyph: View {
    public let size: CGFloat
    public let color: Color

    public init(size: CGFloat, color: Color = .primary) {
        self.size = size
        self.color = color
    }

    public var body: some View {
        if let image = GhostGlyphAssetCache.shared.image {
            color
                .frame(width: size, height: size)
                .mask(
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                )
        } else {
            Image(systemName: "eye.slash")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .foregroundStyle(color)
                .frame(width: size, height: size)
        }
    }
}
