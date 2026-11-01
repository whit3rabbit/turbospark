import Foundation

/// The screenshot the model last saw for one app, in the pixel size it saw.
///
/// Mapping goes through the image the model received, not the capture: the
/// capture is Retina-sized and the media policy may downscale it again, but
/// both keep the window's aspect, so `windowFrame / imageSize` is the scale.
public struct ComputerUseScreenshotFrame: Equatable, Sendable {
    public let screenshotID: Int
    public let bundleID: String
    public let windowFrame: ComputerUseRect
    public let imagePixelWidth: Int
    public let imagePixelHeight: Int

    public init(
        screenshotID: Int,
        bundleID: String,
        windowFrame: ComputerUseRect,
        imagePixelWidth: Int,
        imagePixelHeight: Int
    ) {
        self.screenshotID = screenshotID
        self.bundleID = bundleID
        self.windowFrame = windowFrame
        self.imagePixelWidth = imagePixelWidth
        self.imagePixelHeight = imagePixelHeight
    }

    /// Screen point for image pixel (`x`, `y`), or nil outside the image.
    public func screenPoint(x: Double, y: Double) -> (x: Double, y: Double)? {
        guard imagePixelWidth > 0, imagePixelHeight > 0,
              x.isFinite, y.isFinite,
              x >= 0, y >= 0,
              x < Double(imagePixelWidth), y < Double(imagePixelHeight)
        else { return nil }
        let scaleX = windowFrame.width / Double(imagePixelWidth)
        let scaleY = windowFrame.height / Double(imagePixelHeight)
        return (windowFrame.x + x * scaleX, windowFrame.y + y * scaleY)
    }
}

/// Grants and screenshot frames for one chat. Grants never outlive the
/// session: the app drops it on chat switch, model switch, or kill switch.
public actor ComputerUseSession {
    private var grants: [String: ComputerUseAppTier] = [:]
    private var frames: [String: ComputerUseScreenshotFrame] = [:]
    private var nextScreenshotID = 1

    public init() {}

    public func tier(for bundleID: String) -> ComputerUseAppTier? {
        grants[bundleID]
    }

    public var grantedApps: [String: ComputerUseAppTier] { grants }

    /// Records approved apps at their policy tier. A refused app is never
    /// stored, whatever the caller approved.
    public func grant(_ bundleIDs: [String]) -> [String: ComputerUseAppTier] {
        var added: [String: ComputerUseAppTier] = [:]
        for id in bundleIDs {
            let tier = ComputerUseAppPolicy.tier(forBundleID: id)
            guard tier != .refused else { continue }
            grants[id] = tier
            added[id] = tier
        }
        return added
    }

    public func revokeAll() {
        grants.removeAll()
        frames.removeAll()
    }

    public func recordScreenshot(
        bundleID: String,
        windowFrame: ComputerUseRect,
        imagePixelWidth: Int,
        imagePixelHeight: Int
    ) -> ComputerUseScreenshotFrame {
        let frame = ComputerUseScreenshotFrame(
            screenshotID: nextScreenshotID,
            bundleID: bundleID,
            windowFrame: windowFrame,
            imagePixelWidth: imagePixelWidth,
            imagePixelHeight: imagePixelHeight)
        nextScreenshotID += 1
        frames[bundleID] = frame
        return frame
    }

    /// The latest frame for `bundleID`, only if it is screenshot `id`. An
    /// older id means the model is aiming at a picture that no longer matches.
    public func frame(for bundleID: String, screenshotID id: Int) -> ComputerUseScreenshotFrame? {
        guard let frame = frames[bundleID], frame.screenshotID == id else { return nil }
        return frame
    }
}
