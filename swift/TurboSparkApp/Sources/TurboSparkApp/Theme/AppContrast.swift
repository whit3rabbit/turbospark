import AppKit
import Foundation
import SwiftUI

/// WCAG 2.x contrast math on `#RRGGBB` strings.
///
/// Pure so the theme can pick a text color against a surface and a test can
/// pin the result without a view or an appearance.
///
/// Surfaces must be read from the drawn color with `hex(of:)`, never rebuilt
/// with `blend`: `NSColor.blended(withFraction:of:)`, which builds them, does
/// not mix in plain sRGB (a 9% lift of `#090D16` toward `#F2F5FB` lands on
/// `#282C33`, not `#1E222B`). `blend` is the straight per-channel mix of alpha
/// compositing, which is right for a translucent wash over a surface.
enum AppContrast {
    /// 0...255 sRGB channels, or nil for anything but `#RRGGBB`.
    static func channels(_ hex: String) -> (r: Double, g: Double, b: Double)? {
        guard hex.hasPrefix("#"), hex.count == 7, let n = UInt32(hex.dropFirst(), radix: 16) else { return nil }
        return (Double((n >> 16) & 255), Double((n >> 8) & 255), Double(n & 255))
    }

    /// WCAG relative luminance. An unparseable string reads as black.
    static func luminance(_ hex: String) -> Double {
        guard let c = channels(hex) else { return 0 }
        let linear = [c.r, c.g, c.b].map { channel -> Double in
            let value = channel / 255
            return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
        }
        return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722
    }

    static func ratio(_ a: String, _ b: String) -> Double {
        let la = luminance(a), lb = luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /// `base` moved `fraction` of the way toward `ink`, rounded to a channel.
    static func blend(_ base: String, toward ink: String, _ fraction: Double) -> String {
        guard let b = channels(base), let i = channels(ink) else { return base }
        func mix(_ x: Double, _ y: Double) -> Int { Int((x + (y - x) * fraction).rounded()) }
        return String(format: "#%02X%02X%02X", mix(b.r, i.r), mix(b.g, i.g), mix(b.b, i.b))
    }

    /// `#RRGGBB` for a SwiftUI color, read in sRGB.
    static func hex(of color: Color) -> String {
        guard let c = NSColor(color).usingColorSpace(.sRGB) else { return "#000000" }
        func byte(_ v: CGFloat) -> Int { Int((min(max(v, 0), 1) * 255).rounded()) }
        return String(format: "#%02X%02X%02X", byte(c.redComponent), byte(c.greenComponent), byte(c.blueComponent))
    }
}
