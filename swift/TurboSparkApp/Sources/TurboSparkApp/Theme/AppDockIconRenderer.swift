import AppKit

/// Renders dock icons for the application presets.
public enum AppDockIconRenderer {
    /// Renders and updates the macOS application dock icon based on current preset choice.
    @MainActor
    public static func applyDockIcon(_ icon: AppDockIcon) {
        let iconSize = NSSize(width: 128, height: 128)
        let image = NSImage(size: iconSize)
        image.lockFocus()

        let bounds = NSRect(origin: .zero, size: iconSize)
        let bgPath = NSBezierPath(roundedRect: bounds.insetBy(dx: 8, dy: 8), xRadius: 26, yRadius: 26)

        switch icon {
        case .emeraldSpark:
            image.unlockFocus()
            if let url = Bundle.module.url(forResource: "AppIcon", withExtension: "png"),
               let bundled = NSImage(contentsOf: url) {
                NSApplication.shared.applicationIconImage = bundled
                return
            } else if let appIcon = NSImage(named: "AppIcon") {
                NSApplication.shared.applicationIconImage = appIcon
                return
            }
            image.lockFocus()

            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.08, green: 0.16, blue: 0.10, alpha: 1.0),
                ending: NSColor(srgbRed: 0.02, green: 0.06, blue: 0.03, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)

            // Outer subtle border
            NSColor(srgbRed: 0.41, green: 0.73, blue: 0.44, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.5
            bgPath.stroke()

            // Sparkle symbol
            if let spark = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 48, weight: .semibold)
                if let tinted = spark.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 32, y: 32, width: 64, height: 64)
                    NSColor(srgbRed: 0.41, green: 0.88, blue: 0.44, alpha: 1.0).set()
                    tinted.draw(in: rect)
                }
            }

        case .codexDark:
            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.16, green: 0.16, blue: 0.18, alpha: 1.0),
                ending: NSColor(srgbRed: 0.08, green: 0.08, blue: 0.09, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)
            NSColor(white: 0.35, alpha: 0.5).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "chevron.left.forwardslash.chevron.right", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 42, weight: .bold)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 28, y: 34, width: 72, height: 60)
                    NSColor.white.set()
                    tinted.draw(in: rect)
                }
            }

        case .terminalPro:
            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.05, green: 0.07, blue: 0.12, alpha: 1.0),
                ending: NSColor(srgbRed: 0.01, green: 0.02, blue: 0.04, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)
            NSColor(srgbRed: 0.22, green: 0.53, blue: 0.95, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "terminal.fill", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 46, weight: .medium)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 30, y: 30, width: 68, height: 68)
                    NSColor(srgbRed: 0.22, green: 0.65, blue: 1.0, alpha: 1.0).set()
                    tinted.draw(in: rect)
                }
            }

        case .minimalist:
            NSColor(white: 0.1, alpha: 1.0).setFill()
            bgPath.fill()
            NSColor(white: 0.3, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "bolt.fill", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 46, weight: .bold)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 32, y: 32, width: 64, height: 64)
                    NSColor.white.set()
                    tinted.draw(in: rect)
                }
            }
        }

        image.unlockFocus()
        NSApplication.shared.applicationIconImage = image
    }
}
