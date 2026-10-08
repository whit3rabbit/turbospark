import AppKit
import SwiftUI

/// View modifier that applies a pointer cursor (pointing hand) on hover when enabled in appearance settings.
public struct AppPointerCursorModifier: ViewModifier {
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    /// Whether this view currently owns a pushed cursor. onHover(false) never
    /// fires when the view disappears while hovered (or the preference flips),
    /// so the pop must be tied to our own push, not to the hover callback.
    @State private var didPush = false

    public init() {}

    public func body(content: Content) -> some View {
        content
            .onHover { isHovered in
                if isHovered && appearanceManager.usePointerCursors {
                    if !didPush {
                        NSCursor.pointingHand.push()
                        didPush = true
                    }
                } else {
                    popIfPushed()
                }
            }
            .onChange(of: appearanceManager.usePointerCursors) { _, enabled in
                if !enabled { popIfPushed() }
            }
            .onDisappear { popIfPushed() }
    }

    private func popIfPushed() {
        guard didPush else { return }
        NSCursor.pop()
        didPush = false
    }
}

public extension View {
    /// Applies an interactive pointer hand cursor on hover if the pointer cursor preference is active.
    func appPointerCursor() -> some View {
        modifier(AppPointerCursorModifier())
    }
}
