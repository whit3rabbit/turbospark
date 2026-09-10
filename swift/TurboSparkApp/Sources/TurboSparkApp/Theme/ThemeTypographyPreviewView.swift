import SwiftUI

/// Live UI typography preview card for Appearance settings.
///
/// Demonstrates the real-time effect of UI font family, base font size, weight,
/// and accent color on interface text, headings, and interactive controls.
public struct ThemeTypographyPreviewView: View {
    @ObservedObject var manager: AppearanceManager
    let isDark: Bool

    public init(manager: AppearanceManager, isDark: Bool) {
        self.manager = manager
        self.isDark = isDark
    }

    private var config: ThemeModeConfig {
        manager.activeConfig(isDark: isDark)
    }

    private var uiFont: Font {
        manager.uiFont(isDark: isDark)
    }

    private var bgFillColor: Color {
        manager.activeBackgroundColor(isDark: isDark)
    }

    private var cardBgColor: Color {
        manager.activeBackgroundColor(isDark: isDark)
    }

    private var accentColor: Color {
        manager.activeAccentColor(isDark: isDark)
    }

    private var textColor: Color {
        manager.activeForegroundColor(isDark: isDark)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("UI Typography Preview", bundle: .module)
                    .font(uiFont.weight(.semibold))
                    .foregroundStyle(textColor)

                Spacer()

                HStack(spacing: 6) {
                    Text(config.uiFontFamily)
                        .font(uiFont)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(accentColor.opacity(0.12))
                        .foregroundStyle(accentColor)
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))

                    Text("\(Int(manager.uiFontSize))px", bundle: .module)
                        .font(uiFont)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.06))
                        .foregroundStyle(.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))

                    Text(config.uiFontWeight)
                        .font(uiFont)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.06))
                        .foregroundStyle(.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("The quick brown fox jumps over the lazy dog", bundle: .module)
                    .font(uiFont)
                    .foregroundStyle(textColor)

                HStack(spacing: 12) {
                    Text("Sample interface text with active styling, contrast and accent colors.", bundle: .module)
                        .font(manager.uiFont(isDark: isDark, size: manager.uiFontSize * 0.88))
                        .foregroundStyle(.secondary)

                    Spacer()

                    HStack(spacing: 5) {
                        Circle()
                            .fill(accentColor)
                            .frame(width: 8, height: 8)
                        Text("Active Theme", bundle: .module)
                            .font(manager.uiFont(isDark: isDark, size: manager.uiFontSize * 0.82, weight: .medium))
                            .foregroundStyle(accentColor)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(accentColor.opacity(0.12))
                    .clipShape(Capsule())
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(cardBgColor)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.3 * (config.contrast / 50.0)), lineWidth: 0.5)
            )
        }
        .padding(12)
        .background(bgFillColor)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.3 * (config.contrast / 50.0)), lineWidth: 1)
        )
    }
}
