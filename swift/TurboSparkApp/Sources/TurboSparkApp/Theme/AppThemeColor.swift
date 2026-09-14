import AppKit
import SwiftUI

extension ResolvedAppTheme {
    private func blendedSurface(_ fraction: Double) -> Color {
        let base = NSColor(background).usingColorSpace(.sRGB) ?? .windowBackgroundColor
        let ink = NSColor(foreground).usingColorSpace(.sRGB) ?? .textColor
        return Color(nsColor: base.blended(withFraction: fraction, of: ink) ?? base)
    }
    public var surface: Color { blendedSurface(isDark ? 0.055 : 0.025) }
    public var elevatedSurface: Color { blendedSurface(isDark ? 0.09 : 0.05) }
    public var secondaryText: Color { foreground.opacity(isHighContrast ? 0.96 : (contrast > 70 ? 0.9 : 0.75)) }
    public var border: Color { foreground.opacity(isHighContrast ? 0.75 : (0.12 + contrast / 500)) }
    public var selection: Color { accent.opacity(isHighContrast ? (isDark ? 0.45 : 0.30) : (isDark ? 0.25 : 0.12)) }
}

/// Environment-aware styles work even in leaves without an explicit theme property.
public struct AppThemeColor: ShapeStyle {
    enum Role { case page, surface, elevated, text, secondary, border, accent }
    var role: Role
    var alpha: Double = 1
    public func opacity(_ value: Double) -> AppThemeColor { var copy = self; copy.alpha *= value; return copy }
    public func resolve(in environment: EnvironmentValues) -> Color {
        let theme = environment.appTheme
        let color: Color
        switch role {
        case .page: color = theme.background
        case .surface: color = theme.surface
        case .elevated: color = theme.elevatedSurface
        case .text: color = theme.foreground
        case .secondary: color = theme.secondaryText
        case .border: color = theme.border
        case .accent: color = theme.accent
        }
        return color.opacity(alpha)
    }
}

public extension ShapeStyle where Self == AppThemeColor {
    static var appPage: AppThemeColor { .init(role: .page) }
    static var appSurface: AppThemeColor { .init(role: .surface) }
    static var appElevated: AppThemeColor { .init(role: .elevated) }
    static var appText: AppThemeColor { .init(role: .text) }
    static var appSecondary: AppThemeColor { .init(role: .secondary) }
    static var appBorder: AppThemeColor { .init(role: .border) }
    static var appAccent: AppThemeColor { .init(role: .accent) }
}
