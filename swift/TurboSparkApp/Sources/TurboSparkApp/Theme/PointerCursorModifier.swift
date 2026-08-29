import AppKit
import SwiftUI

/// View modifier that applies a pointer cursor (pointing hand) on hover when enabled in appearance settings.
public struct AppPointerCursorModifier: ViewModifier {
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    public init() {}

    public func body(content: Content) -> some View {
        if appearanceManager.usePointerCursors {
            content
                .onHover { isHovered in
                    if isHovered {
                        NSCursor.pointingHand.push()
                    } else {
                        NSCursor.pop()
                    }
                }
        } else {
            content
        }
    }
}

public extension View {
    /// Applies an interactive pointer hand cursor on hover if the pointer cursor preference is active.
    func appPointerCursor() -> some View {
        modifier(AppPointerCursorModifier())
    }
}
