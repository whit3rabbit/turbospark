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
