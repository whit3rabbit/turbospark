import SwiftUI

/// Live syntax-highlighted and diff-styled code block preview for Appearance settings.
///
/// Demonstrates the real-time effect of font family, code font size, syntax colors,
/// accent highlights, contrast adjustments, and diff marker preferences (+/- or colored gutters).
public struct ThemeCodePreviewView: View {
    @ObservedObject var manager: AppearanceManager
    let isDark: Bool

    /// Initializes a theme code preview component.
    ///
    /// - Parameters:
    ///   - manager: Shared appearance manager holding theme state.
    ///   - isDark: Whether the preview renders in dark mode styling.
    public init(manager: AppearanceManager, isDark: Bool) {
        self.manager = manager
        self.isDark = isDark
    }

    private var config: ThemeModeConfig {
        manager.activeConfig(isDark: isDark)
    }

    private var codeFont: Font {
        manager.codeFont(size: CGFloat(manager.codeFontSize))
    }

    private var bgFillColor: Color {
        isDark ? Color(red: 0.11, green: 0.11, blue: 0.12) : Color(red: 0.96, green: 0.96, blue: 0.97)
    }

    private var redDiffBg: Color {
        isDark ? Color.red.opacity(0.22) : Color.red.opacity(0.12)
    }

    private var greenDiffBg: Color {
        isDark ? Color.green.opacity(0.22) : Color.green.opacity(0.12)
    }

    private var redLineIndicator: Color {
        Color.red.opacity(0.85)
    }

    private var greenLineIndicator: Color {
        Color.green.opacity(0.85)
    }

    public var body: some View {
        HStack(spacing: 0) {
            // Left block (Original / Before)
            VStack(alignment: .leading, spacing: 3) {
                codeLine(number: 1, text: "const themePreview: ThemeConfig = {", keyword: "const", typeName: "ThemeConfig")
                diffLine(number: 2, text: "  surface: \"sidebar\",", isAddition: false)
                codeLine(number: 3, text: "  accent: \"#2563eb\",", isProperty: true, valueColor: manager.activeAccentColor(isDark: isDark))
                codeLine(number: 4, text: "  contrast: 42,", isProperty: true)
                codeLine(number: 5, text: "};")
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)

            Rectangle()
                .fill(Color(nsColor: .separatorColor).opacity(0.4 * (config.contrast / 50.0)))
                .frame(width: 1)

            // Right block (Modified / After)
            VStack(alignment: .leading, spacing: 3) {
                codeLine(number: 1, text: "const themePreview: ThemeConfig = {", keyword: "const", typeName: "ThemeConfig")
                diffLine(number: 2, text: "  surface: \"sidebar-elevated\",", isAddition: true)
                diffLine(number: 3, text: "  accent: \"\(config.accentHex)\",", isAddition: true, valueColor: manager.activeAccentColor(isDark: isDark))
                diffLine(number: 4, text: "  contrast: \(Int(config.contrast)),", isAddition: true)
                codeLine(number: 5, text: "};")
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(codeFont)
        .background(bgFillColor)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.3 * (config.contrast / 50.0)), lineWidth: 1)
        )
    }

    private func codeLine(
        number: Int,
        text: String,
        keyword: String? = nil,
        typeName: String? = nil,
        isProperty: Bool = false,
        valueColor: Color? = nil
    ) -> some View {
        HStack(spacing: 8) {
            Text("\(number)")
                .frame(width: 14, alignment: .trailing)
                .foregroundStyle(Color.secondary.opacity(0.6))
                .font(codeFont)

            if let keyword = keyword, let typeName = typeName {
                HStack(spacing: 4) {
                    Text(keyword).foregroundStyle(Color.purple)
                    Text("themePreview:").foregroundStyle(isDark ? Color.white : Color.black)
                    Text(typeName).foregroundStyle(Color.blue)
                    Text("= {").foregroundStyle(isDark ? Color.white : Color.black)
                }
            } else if isProperty {
                formattedPropertyLine(text: text, valueColor: valueColor)
            } else {
                Text(text)
                    .foregroundStyle(isDark ? Color.white.opacity(0.9) : Color.black.opacity(0.85))
            }
            Spacer(minLength: 0)
        }
        .frame(height: 18)
    }

    private func diffLine(
        number: Int,
        text: String,
        isAddition: Bool,
        valueColor: Color? = nil
    ) -> some View {
        let isPlusMinus = manager.diffMarkers == .plusMinus
        let bg = isAddition ? greenDiffBg : redDiffBg
        let indicatorColor = isAddition ? greenLineIndicator : redLineIndicator
        let marker = isAddition ? "+" : "-"

        return HStack(spacing: 0) {
            Rectangle()
                .fill(indicatorColor)
                .frame(width: 3)

            HStack(spacing: 6) {
                Text("\(number)")
                    .frame(width: 14, alignment: .trailing)
                    .foregroundStyle(Color.secondary.opacity(0.7))

                if isPlusMinus {
                    Text(marker)
                        .foregroundStyle(indicatorColor)
                        .fontWeight(.bold)
                }

                formattedPropertyLine(text: text, valueColor: valueColor)
                Spacer(minLength: 0)
            }
            .padding(.leading, 5)
        }
        .frame(height: 18)
        .background(manager.diffMarkers == .color || isAddition ? bg : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
    }

    private func formattedPropertyLine(text: String, valueColor: Color? = nil) -> some View {
        let parts = text.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        return HStack(spacing: 2) {
            if parts.count == 2 {
                Text(String(parts[0]) + ":")
                    .foregroundStyle(isDark ? Color(red: 0.95, green: 0.55, blue: 0.55) : Color(red: 0.8, green: 0.2, blue: 0.2))
                Text(String(parts[1]))
                    .foregroundStyle(valueColor ?? (isDark ? Color(red: 0.45, green: 0.85, blue: 0.55) : Color(red: 0.15, green: 0.6, blue: 0.25)))
            } else {
                Text(text)
                    .foregroundStyle(isDark ? Color.white.opacity(0.9) : Color.black.opacity(0.85))
            }
        }
    }
}
