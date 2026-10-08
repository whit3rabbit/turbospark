import AppKit
import SwiftUI

/// Enforce the pane minimum on the actual window as well as the SwiftUI
/// layout. Restored frames and profile unlock can otherwise retain a smaller
/// window and clip a root view that refuses the proposed size.
struct MainWindowMinimumSize: NSViewRepresentable {
    var size: CGSize

    func makeNSView(context: Context) -> ConstraintView {
        ConstraintView(size: size)
    }

    func updateNSView(_ view: ConstraintView, context: Context) {
        view.minimumSize = size
        view.applyMinimumSize()
    }

    /// A minimum larger than the screen's visible area would push the window
    /// off screen and, through contentMinSize, stop the user from shrinking it
    /// back. The layout degrades inside the smaller window instead.
    static func clamped(_ size: CGSize, toScreen screen: CGSize?) -> CGSize {
        guard let screen, screen.width > 0, screen.height > 0 else { return size }
        return CGSize(
            width: min(size.width, screen.width),
            height: min(size.height, screen.height))
    }

    final class ConstraintView: NSView {
        var minimumSize: CGSize

        init(size: CGSize) {
            minimumSize = size
            super.init(frame: .zero)
        }

        required init?(coder: NSCoder) { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            applyMinimumSize()
        }

        func applyMinimumSize() {
            guard let window else { return }
            let minimumSize = MainWindowMinimumSize.clamped(
                minimumSize, toScreen: window.screen?.visibleFrame.size)
            window.contentMinSize = minimumSize
            let current = window.contentRect(forFrameRect: window.frame).size
            let required = CGSize(
                width: max(current.width, minimumSize.width),
                height: max(current.height, minimumSize.height))
            guard required != current else { return }
            var frame = window.frameRect(forContentRect: NSRect(origin: .zero, size: required))
            frame.origin = CGPoint(x: window.frame.minX, y: window.frame.maxY - frame.height)
            window.setFrame(frame, display: true)
        }
    }
}
