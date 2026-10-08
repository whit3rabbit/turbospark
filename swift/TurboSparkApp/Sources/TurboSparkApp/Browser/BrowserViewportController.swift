import AppKit
import Combine
import SwiftUI
import WebKit

public enum BrowserViewportMode: Equatable, Sendable {
    case responsive
    case fixed
}

public struct BrowserViewportLayout: Equatable, Sendable {
    public let size: CGSize
    public let pageZoom: CGFloat

    public init(size: CGSize, pageZoom: CGFloat) {
        self.size = size
        self.pageZoom = pageZoom
    }
}

@MainActor
public final class BrowserViewportController: ObservableObject {
    public static let widthRange = 320...3840
    public static let heightRange = 240...2160
    public static let zoomRange = 0.5...3.0

    @Published public private(set) var mode: BrowserViewportMode = .responsive
    @Published public private(set) var preference: BrowserViewportPreference

    /// The scroll view's content offset in fixed mode (zero when responsive).
    /// The element picker's hit layer is overlaid on the scroll view's
    /// viewport, not on the page, so its coordinates must add this offset.
    @Published public private(set) var scrollOffset: CGPoint = .zero

    private let writePreference: (BrowserViewportPreference) -> Void

    public init(
        readPreference: () -> BrowserViewportPreference = {
            MacAppSettingsFileStore.load().browser.viewportPreference
        },
        writePreference: @escaping (BrowserViewportPreference) -> Void = { preference in
            var settings = MacAppSettingsFileStore.load()
            settings.browser.viewportPreference = preference
            MacAppSettingsFileStore.save(settings)
        }
    ) {
        self.preference = Self.normalized(readPreference())
        self.writePreference = writePreference
    }

    public var isFixed: Bool {
        mode == .fixed
    }

    public func setScrollOffset(_ offset: CGPoint) {
        guard offset != scrollOffset else { return }
        scrollOffset = offset
    }

    /// Maps a point in the hit layer (viewport coordinates) to page CSS
    /// pixels: add the scroll offset, then undo the page zoom.
    public nonisolated static func cssPoint(viewPoint: CGPoint, scrollOffset: CGPoint, scale: CGFloat) -> CGPoint {
        let s = max(scale, 0.01)
        return CGPoint(x: (viewPoint.x + scrollOffset.x) / s, y: (viewPoint.y + scrollOffset.y) / s)
    }

    public func setMode(_ mode: BrowserViewportMode) {
        self.mode = mode
    }

    public func updatePreference(width: Int? = nil, height: Int? = nil, zoom: Double? = nil) {
        let next = Self.normalized(
            BrowserViewportPreference(
                width: width ?? preference.width,
                height: height ?? preference.height,
                zoom: zoom ?? preference.zoom
            )
        )
        guard next != preference else { return }
        preference = next
        writePreference(next)
    }

    public func layout(in containerSize: CGSize) -> BrowserViewportLayout {
        guard isFixed else {
            return BrowserViewportLayout(
                size: CGSize(width: max(0, containerSize.width), height: max(0, containerSize.height)),
                pageZoom: 1
            )
        }
        return BrowserViewportLayout(
            size: CGSize(width: preference.width, height: preference.height),
            pageZoom: CGFloat(preference.zoom)
        )
    }

    /// Changes the live view geometry and page zoom without navigating or reloading it.
    public func apply(to webView: WKWebView, containerSize: CGSize) {
        let layout = layout(in: containerSize)
        webView.pageZoom = layout.pageZoom
        webView.setFrameSize(layout.size)
    }

    private static func normalized(_ preference: BrowserViewportPreference) -> BrowserViewportPreference {
        let zoom = min(max(preference.zoom, zoomRange.lowerBound), zoomRange.upperBound)
        return BrowserViewportPreference(
            width: min(max(preference.width, widthRange.lowerBound), widthRange.upperBound),
            height: min(max(preference.height, heightRange.lowerBound), heightRange.upperBound),
            zoom: (zoom * 100).rounded() / 100
        )
    }
}

struct BrowserViewportWebView: View {
    @ObservedObject var controller: BrowserViewportController
    let webView: WKWebView

    var body: some View {
        GeometryReader { geometry in
            BrowserViewportHost(
                controller: controller,
                webView: webView,
                containerSize: geometry.size
            )
        }
    }
}

@MainActor
private struct BrowserViewportHost: NSViewRepresentable {
    @ObservedObject var controller: BrowserViewportController
    let webView: WKWebView
    let containerSize: CGSize

    func makeCoordinator() -> Coordinator { Coordinator() }

    @MainActor
    final class Coordinator {
        var observer: NSObjectProtocol?
        deinit { if let observer { NotificationCenter.default.removeObserver(observer) } }
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.contentView.postsBoundsChangedNotifications = true
        let controller = self.controller
        context.coordinator.observer = NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification,
            object: scrollView.contentView,
            queue: .main
        ) { [weak scrollView] _ in
            MainActor.assumeIsolated {
                guard let scrollView else { return }
                controller.setScrollOffset(
                    controller.isFixed ? scrollView.contentView.bounds.origin : .zero)
            }
        }
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        scrollView.autohidesScrollers = true

        let documentView = BrowserViewportDocumentView(frame: .zero)
        documentView.addSubview(webView)
        scrollView.documentView = documentView
        update(scrollView, documentView: documentView)
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        let documentView: BrowserViewportDocumentView
        if let current = scrollView.documentView as? BrowserViewportDocumentView {
            documentView = current
        } else {
            documentView = BrowserViewportDocumentView(frame: .zero)
            scrollView.documentView = documentView
        }
        if webView.superview !== documentView {
            webView.removeFromSuperview()
            documentView.addSubview(webView)
        }
        update(scrollView, documentView: documentView)
    }

    private func update(_ scrollView: NSScrollView, documentView: BrowserViewportDocumentView) {
        let layout = controller.layout(in: containerSize)
        let fixed = controller.isFixed
        scrollView.hasHorizontalScroller = fixed
        scrollView.hasVerticalScroller = fixed
        documentView.setFrameSize(layout.size)
        controller.apply(to: webView, containerSize: containerSize)
    }
}

@MainActor
private final class BrowserViewportDocumentView: NSView {
    override var isFlipped: Bool { true }

    override func resizeSubviews(withOldSize oldSize: NSSize) {
        super.resizeSubviews(withOldSize: oldSize)
        subviews.first?.frame = bounds
    }
}
