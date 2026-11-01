import AppKit
import ApplicationServices

/// Renders an app's front window as a bounded text tree with element refs.
///
/// Refs live only until the next snapshot of the same app, so a ref always
/// points at an element the model saw in its latest tree. Secure text field
/// values are never read.
@MainActor
final class AXSnapshotter {
    static let maximumDepth = 30
    static let maximumNodes = 500
    static let maximumCharacters = 30_000

    private var references: [String: [String: AXUIElement]] = [:]

    /// Roles worth a ref even without a label.
    private static let actionableRoles: Set<String> = [
        "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton",
        "AXTextField", "AXTextArea", "AXComboBox", "AXSlider", "AXLink", "AXMenuItem",
        "AXTab", "AXCell", "AXRow", "AXDisclosureTriangle", "AXIncrementor", "AXSearchField",
    ]

    func element(bundleID: String, reference: String) -> AXUIElement? {
        references[bundleID]?[reference]
    }

    func snapshot(app: NSRunningApplication) throws -> String {
        let root = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(root, 1.0)
        guard let window = Self.attribute(root, kAXFocusedWindowAttribute) as AXUIElement?
            ?? Self.attribute(root, kAXMainWindowAttribute) as AXUIElement?
            ?? (Self.attribute(root, kAXWindowsAttribute) as [AXUIElement]?)?.first
        else {
            throw HelperError(.failed, "\(app.localizedName ?? "The app") has no window to read.")
        }

        var refs: [String: AXUIElement] = [:]
        var lines: [String] = []
        var characters = 0
        var nodes = 0
        var truncated = false

        func visit(_ element: AXUIElement, depth: Int) {
            guard !truncated else { return }
            guard depth <= Self.maximumDepth, nodes < Self.maximumNodes else {
                truncated = true
                return
            }
            nodes += 1
            let role = Self.attribute(element, kAXRoleAttribute) as String? ?? "AXUnknown"
            let subrole = Self.attribute(element, kAXSubroleAttribute) as String?
            let label = [
                Self.attribute(element, kAXTitleAttribute) as String?,
                Self.attribute(element, kAXDescriptionAttribute) as String?,
            ].compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }.first { !$0.isEmpty }
            let isSecure = subrole == "AXSecureTextField"
            let value: String? = isSecure ? "<masked>" : Self.stringValue(element)

            var printed = 0
            if label != nil || value != nil || Self.actionableRoles.contains(role) {
                let ref = "e\(refs.count + 1)"
                refs[ref] = element
                var line = String(repeating: "  ", count: min(depth, 12)) + "[\(ref)] \(role)"
                if let label { line += " \"\(Self.clip(label, 120))\"" }
                if let value { line += " value=\"\(Self.clip(value, 200))\"" }
                if Self.attribute(element, kAXEnabledAttribute) as Bool? == false { line += " disabled" }
                if let frame = Self.frame(of: element) {
                    line += " (\(Int(frame.minX)),\(Int(frame.minY)) \(Int(frame.width))x\(Int(frame.height)))"
                }
                characters += line.count + 1
                if characters > Self.maximumCharacters {
                    truncated = true
                    return
                }
                lines.append(line)
                printed = 1
            }
            let children = Self.attribute(element, kAXChildrenAttribute) as [AXUIElement]? ?? []
            for child in children {
                visit(child, depth: depth + printed)
            }
        }

        visit(window, depth: 0)
        references[app.bundleIdentifier ?? ""] = refs
        if truncated { lines.append("(truncated: tree larger than the snapshot limit)") }
        return lines.isEmpty ? "(no readable elements)" : lines.joined(separator: "\n")
    }

    static func frame(of element: AXUIElement) -> CGRect? {
        guard let positionValue = attribute(element, kAXPositionAttribute) as AXValue?,
              let sizeValue = attribute(element, kAXSizeAttribute) as AXValue?
        else { return nil }
        var position = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(positionValue, .cgPoint, &position),
              AXValueGetValue(sizeValue, .cgSize, &size)
        else { return nil }
        return CGRect(origin: position, size: size)
    }

    private static func stringValue(_ element: AXUIElement) -> String? {
        guard let raw: CFTypeRef = attribute(element, kAXValueAttribute) else { return nil }
        if let string = raw as? String {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        if let number = raw as? NSNumber { return number.stringValue }
        return nil
    }

    private static func attribute<T>(_ element: AXUIElement, _ name: String) -> T? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success,
              let value
        else { return nil }
        if T.self == AXUIElement.self || T.self == AXValue.self || T.self == CFTypeRef.self {
            return (value as! T)
        }
        return value as? T
    }

    private static func clip(_ text: String, _ limit: Int) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: "\"", with: "'")
        return flat.count > limit ? String(flat.prefix(limit)) + "..." : flat
    }
}
