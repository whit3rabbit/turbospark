import AppKit
import os

/// Composites a diagonal "DEV" ribbon onto the Dock icon of an unpackaged
/// build, so a build run from `swift run` or a bare debug binary is not
/// mistaken for the installed release.
///
/// This is a visual aid only. It does not change the bundle id or the data
/// directories, so it does not isolate a dev build's state from a release.
///
/// "Packaged" means `scripts/make-app-bundle.sh` wrote `TSReleasePackage`
/// into Info.plist. A bundle built without the key is badged, which is the
/// safe direction: a missing key never hides the badge on a dev build. The
/// ribbon is a drawn image, so the localization gates do not apply to "DEV".
public enum DevIconBadger {
    /// Info.plist key written (as `<true/>`) by `make-app-bundle.sh`.
    public static let releaseMarkerKey = "TSReleasePackage"

    private static let log = Logger(subsystem: "com.whit3rabbit.turbospark", category: "DevIconBadger")
    nonisolated(unsafe) private static var didLogFailure = false

    /// The policy: badge unless the release marker is present and true.
    public static func shouldBadge(releaseMarker: Any?) -> Bool {
        (releaseMarker as? Bool) != true
    }

    public static func shouldBadge(bundle: Bundle = .main) -> Bool {
        shouldBadge(releaseMarker: bundle.object(forInfoDictionaryKey: releaseMarkerKey))
    }

    /// The icon to show: the badged copy when the policy says so and
    /// compositing succeeds, otherwise `source` unchanged. Logs one line per
    /// process on a compositing failure and never throws.
    @MainActor
    public static func resolvedIcon(_ source: NSImage, bundle: Bundle = .main) -> NSImage {
        guard shouldBadge(bundle: bundle) else { return source }
        if let badged = badged(source) { return badged }
        if !didLogFailure {
            didLogFailure = true
            log.error("Dev icon badge failed; keeping the original icon")
        }
        return source
    }

    /// Draws `source` and a ribbon across its lower-right corner. Returns nil
    /// when the source has no pixels or no bitmap context can be made.
    public static func badged(_ source: NSImage) -> NSImage? {
        let width = Int(source.size.width.rounded())
        let height = Int(source.size.height.rounded())
        guard width > 0, height > 0,
              let rep = NSBitmapImageRep(
                  bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
                  bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                  colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let context = NSGraphicsContext(bitmapImageRep: rep)
        else { return nil }

        let w = CGFloat(width)
        let h = CGFloat(height)
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = context
        source.draw(in: NSRect(x: 0, y: 0, width: w, height: h))

        // Source-atop keeps the ribbon inside the icon's own opaque shape.
        context.compositingOperation = .sourceAtop
        let transform = NSAffineTransform()
        transform.translateX(by: w * 0.78, yBy: h * 0.22)
        transform.rotate(byDegrees: 45)
        transform.concat()

        let band = NSRect(x: -w * 0.7, y: -h * 0.09, width: w * 1.4, height: h * 0.18)
        NSColor(srgbRed: 0.04, green: 0.42, blue: 0.95, alpha: 1).setFill()
        band.fill()

        let label = NSAttributedString(string: "DEV", attributes: [
            .font: NSFont.systemFont(ofSize: h * 0.13, weight: .heavy),
            .foregroundColor: NSColor.white,
            .kern: h * 0.01,
        ])
        let size = label.size()
        label.draw(at: NSPoint(x: -size.width / 2, y: -size.height / 2))

        let image = NSImage(size: source.size)
        image.addRepresentation(rep)
        return image
    }
}
