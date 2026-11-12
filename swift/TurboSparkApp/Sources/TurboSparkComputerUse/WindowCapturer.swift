import ComputerUseKit
import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

/// Captures one app's frontmost normal window, never the whole display, so a
/// screenshot cannot include other apps the user did not grant.
@MainActor
final class WindowCapturer {
    func capture(pid: pid_t) async throws -> ComputerUseCapture {
        // The caller checked CGPreflightScreenCaptureAccess first: a
        // SCShareableContent query prompts when access is missing.
        guard let windowID = InputSynthesizer.frontWindow(pid: pid)?.id else {
            throw HelperError(.failed, "The app has no visible window.")
        }
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
            throw HelperError(.failed, "The app's window is not capturable.")
        }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let configuration = SCStreamConfiguration()
        let scale = CGFloat(filter.pointPixelScale)
        configuration.width = max(1, Int(window.frame.width * scale))
        configuration.height = max(1, Int(window.frame.height * scale))
        configuration.showsCursor = false
        configuration.ignoreShadowsSingleWindow = true
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: configuration)
        guard let png = Self.encodePNG(image) else { throw HelperError(.failed, "Could not encode the screenshot.") }
        let frame = window.frame
        return ComputerUseCapture(
            pngData: png,
            windowFrame: ComputerUseRect(x: frame.minX, y: frame.minY, width: frame.width, height: frame.height))
    }

    private static func encodePNG(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? data as Data : nil
    }
}
