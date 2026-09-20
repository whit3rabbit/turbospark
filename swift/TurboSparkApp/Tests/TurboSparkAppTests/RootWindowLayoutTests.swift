import AppKit
import SwiftUI
import XCTest
@testable import TurboSparkApp

@MainActor
final class RootWindowLayoutTests: XCTestCase {
    private func withWindow(
        size: CGSize,
        check: (AppModel, NSWindow, NSView) async throws -> Void
    ) async throws {
        let model = AppModel()
        model.stopCronScheduler()
        model.activeSection = .files
        model.installed = []
        model.selected = nil
        model.modelDownloads = []
        model.isDownloadManagerExpanded = false
        model.activeToast = nil
        let suite = "RootWindowLayoutTests-" + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.set(true, forKey: "TurboSpark.chatSidebarVisible")
        defaults.set(true, forKey: "TurboSpark.inspectorVisible")
        let previousShutdown = AppShutdownCoordinator.shared.onTerminate
        let previousMotion = AppearanceManager.shared.reduceMotion
        AppearanceManager.shared.reduceMotion = .on
        let host = NSHostingView(rootView: RootView(model: model)
            .defaultAppStorage(defaults).environment(\.colorScheme, .dark))
        host.appearance = NSAppearance(named: .darkAqua)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        defer {
            window.close()
            AppearanceManager.shared.reduceMotion = previousMotion
            AppShutdownCoordinator.shared.onTerminate = previousShutdown
            defaults.removePersistentDomain(forName: suite)
        }
        try await settle(host)
        try await check(model, window, host)
    }

    private func settle(_ host: NSView) async throws {
        for _ in 0..<3 {
            host.layoutSubtreeIfNeeded()
            if let warm = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
                host.cacheDisplay(in: host.bounds, to: warm)
            }
            try await Task.sleep(for: .milliseconds(60))
        }
    }

    func testDownloadsDoNotPaintOverTheInspector() async throws {
        try await withWindow(size: CGSize(width: 1320, height: 800)) { model, _, host in
            for expanded in [false, true] {
                model.isDownloadManagerExpanded = expanded
                try await settle(host)
                let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
                host.cacheDisplay(in: host.bounds, to: bitmap)
                // This region contains the inspector's form, above the open
                // download card. An opaque overlay turns it into a solid strip.
                let scale = CGFloat(bitmap.pixelsWide) / host.bounds.width
                let left = Int((host.bounds.width - 280) * scale)
                let right = Int((host.bounds.width - 30) * scale)
                let top = Int(110 * scale)
                let bottom = Int(300 * scale)
                let reference = try XCTUnwrap(bitmap.colorAt(x: left, y: top)?.usingColorSpace(.deviceRGB))
                var visibleDetails = 0
                for y in stride(from: top, to: bottom, by: 3) {
                    for x in stride(from: left, to: right, by: 3) {
                        let color = try XCTUnwrap(bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                        let difference = abs(color.redComponent - reference.redComponent)
                            + abs(color.greenComponent - reference.greenComponent)
                            + abs(color.blueComponent - reference.blueComponent)
                        if difference > 0.15 { visibleDetails += 1 }
                    }
                }
                XCTAssertGreaterThan(visibleDetails, 50, "Downloads expanded=\(expanded) hid the inspector")
            }
        }
    }

    func testNativeWindowFloorProtectsRestoredFramesAndTracksVisiblePanes() async throws {
        try await withWindow(size: CGSize(width: 600, height: 360)) { _, window, host in
            XCTAssertEqual(window.contentMinSize, CGSize(width: 1222, height: 640))
            XCTAssertGreaterThanOrEqual(host.bounds.width, 1222)
            XCTAssertGreaterThanOrEqual(host.bounds.height, 640)
            let expandedSize = window.frame.size

            NotificationCenter.default.post(name: .toggleInspector, object: nil)
            try await settle(host)
            XCTAssertEqual(window.contentMinSize, CGSize(width: 901, height: 640))
            XCTAssertEqual(window.frame.size, expandedSize, "closing a pane must not shrink the user's window")

            NotificationCenter.default.post(name: .toggleChatSidebar, object: nil)
            try await settle(host)
            XCTAssertEqual(window.contentMinSize, CGSize(width: 693, height: 640))
        }
    }
}
