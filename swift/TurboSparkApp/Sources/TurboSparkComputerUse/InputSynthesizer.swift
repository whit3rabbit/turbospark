import AppKit
import ComputerUseKit
import CoreGraphics

/// Posts mouse, keyboard, and scroll events. Every event carries
/// `syntheticEventMarker` so the kill switch can tell it from the user.
@MainActor
final class InputSynthesizer {
    struct WindowInfo {
        let id: CGWindowID
        let pid: pid_t
        let bounds: CGRect
        let layer: Int
    }

    private let source = CGEventSource(stateID: .privateState)

    func bringToFront(_ app: NSRunningApplication) async throws {
        if NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier { return }
        app.activate()
        for _ in 0..<20 {
            try await Task.sleep(nanoseconds: 50_000_000)
            if NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier { return }
        }
        throw HelperError(.notFrontmost, "\(app.localizedName ?? "The app") did not come to the front.")
    }

    func click(at point: CGPoint, pid: pid_t, button: ComputerUseMouseButton, count: Int) throws {
        try requireTopWindow(at: point, pid: pid)
        let (down, up, mouseButton): (CGEventType, CGEventType, CGMouseButton) = button == .left
            ? (.leftMouseDown, .leftMouseUp, .left)
            : (.rightMouseDown, .rightMouseUp, .right)
        post(CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: mouseButton))
        for click in 1...count {
            for type in [down, up] {
                let event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: point, mouseButton: mouseButton)
                event?.setIntegerValueField(.mouseEventClickState, value: Int64(click))
                post(event)
            }
        }
    }

    func type(_ text: String, app: NSRunningApplication) throws {
        let units = Array(text.utf16)
        var index = 0
        // Events carry at most 20 UTF-16 units of Unicode text.
        while index < units.count {
            try requireFrontmost(app)
            let chunk = Array(units[index..<min(index + 20, units.count)])
            for keyDown in [true, false] {
                let event = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: keyDown)
                event?.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: chunk)
                post(event)
            }
            index += 20
        }
    }

    func press(_ combo: ComputerUseKeyCombo, app: NSRunningApplication) throws {
        try requireFrontmost(app)
        var flags: CGEventFlags = []
        if combo.modifiers.contains(.command) { flags.insert(.maskCommand) }
        if combo.modifiers.contains(.shift) { flags.insert(.maskShift) }
        if combo.modifiers.contains(.option) { flags.insert(.maskAlternate) }
        if combo.modifiers.contains(.control) { flags.insert(.maskControl) }
        for keyDown in [true, false] {
            let event = CGEvent(keyboardEventSource: source, virtualKey: combo.keyCode, keyDown: keyDown)
            event?.flags = flags
            post(event)
        }
    }

    func scroll(at point: CGPoint, pid: pid_t, direction: ComputerUseScrollDirection, lines: Int) throws {
        try requireTopWindow(at: point, pid: pid)
        let amount = Int32(lines)
        let (vertical, horizontal): (Int32, Int32) = switch direction {
        case .up: (amount, 0)
        case .down: (-amount, 0)
        case .left: (0, amount)
        case .right: (0, -amount)
        }
        // Scroll goes to the window under the cursor, so the cursor moves.
        post(CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left))
        let event = CGEvent(
            scrollWheelEvent2Source: source, units: .line, wheelCount: 2,
            wheel1: vertical, wheel2: horizontal, wheel3: 0)
        event?.location = point
        post(event)
    }

    func frontWindowCenter(pid: pid_t) throws -> CGPoint {
        guard let window = Self.frontWindow(pid: pid) else {
            throw HelperError(.failed, "The app has no visible window.")
        }
        return CGPoint(x: window.bounds.midX, y: window.bounds.midY)
    }

    /// On-screen windows, front to back.
    static func windows() -> [WindowInfo] {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return [] }
        return list.compactMap { entry in
            guard let id = entry[kCGWindowNumber as String] as? CGWindowID,
                  let pid = entry[kCGWindowOwnerPID as String] as? pid_t,
                  let boundsDictionary = entry[kCGWindowBounds as String] as? NSDictionary,
                  let bounds = CGRect(dictionaryRepresentation: boundsDictionary),
                  let layer = entry[kCGWindowLayer as String] as? Int
            else { return nil }
            let alpha = entry[kCGWindowAlpha as String] as? Double ?? 1
            guard alpha > 0 else { return nil }
            return WindowInfo(id: id, pid: pid, bounds: bounds, layer: layer)
        }
    }

    static func frontWindow(pid: pid_t) -> WindowInfo? {
        windows().first { $0.pid == pid && $0.layer == 0 && $0.bounds.width > 40 && $0.bounds.height > 40 }
    }

    /// The topmost window under `point` must belong to the target app, so a
    /// notification or another app's window on top never takes the click.
    private func requireTopWindow(at point: CGPoint, pid: pid_t) throws {
        let ownPID = ProcessInfo.processInfo.processIdentifier
        guard let top = Self.windows().first(where: { $0.pid != ownPID && $0.bounds.contains(point) }),
              top.pid == pid
        else {
            throw HelperError(.outsideWindow, "That point is not on the app's window. Take a new screenshot.")
        }
    }

    private func requireFrontmost(_ app: NSRunningApplication) throws {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier else {
            throw HelperError(.notFrontmost, "\(app.localizedName ?? "The app") is no longer frontmost. Input stopped.")
        }
    }

    private func post(_ event: CGEvent?) {
        guard let event else { return }
        event.setIntegerValueField(.eventSourceUserData, value: ComputerUseHelperLayout.syntheticEventMarker)
        event.post(tap: .cghidEventTap)
    }
}
