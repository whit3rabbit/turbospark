import AppKit
import SwiftUI

public struct AppearanceSettingsPaneView: View {
    @ObservedObject private var manager = AppearanceManager.shared
    @Environment(\.colorScheme) private var colorScheme

    public init() {}

    private var isCurrentlyDark: Bool {
        switch manager.appearance {
        case .system:
            return colorScheme == .dark
        case .light:
            return false
        case .dark:
            return true
        }
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                themeSelectionSection

                ThemeCodePreviewView(manager: manager, isDark: isCurrentlyDark)

                ThemeConfigCardView(
                    title: "Light theme",
                    isDark: false,
                    config: $manager.lightConfig,
                    manager: manager
                )

                ThemeConfigCardView(
                    title: "Dark theme",
                    isDark: true,
                    config: $manager.darkConfig,
                    manager: manager
                )

                AppearancePreferencesCardView(manager: manager)
            }
            .padding(20)
        }
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.6))
    }

    // MARK: - Theme Mode Selector
    private var themeSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Theme")
                .font(.headline)
                .foregroundStyle(.primary)

            HStack(spacing: 16) {
                themeCard(mode: .system, title: "System") {
                    // Split card preview
                    HStack(spacing: 0) {
                        lightThumbnailMockup
                        darkThumbnailMockup
                    }
                }

                themeCard(mode: .light, title: "Light") {
                    lightThumbnailMockup
                }

                themeCard(mode: .dark, title: "Dark") {
                    darkThumbnailMockup
                }
            }
        }
    }

    private func themeCard<Content: View>(
        mode: AppAppearance,
        title: String,
        @ViewBuilder preview: () -> Content
    ) -> some View {
        let isSelected = manager.appearance == mode
        return Button {
            manager.appearance = mode
        } label: {
            VStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color(nsColor: .controlBackgroundColor))

                    preview()
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(
                            isSelected ? manager.activeAccentColor(isDark: isCurrentlyDark) : Color(nsColor: .separatorColor).opacity(0.3),
                            lineWidth: isSelected ? 2.5 : 1
                        )
                }
                .frame(height: 80)
                .shadow(color: isSelected ? manager.activeAccentColor(isDark: isCurrentlyDark).opacity(0.2) : Color.black.opacity(0.04), radius: 4, y: 2)

                Text(title)
                    .font(.caption.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .primary : .secondary)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
        .appPointerCursor()
    }

    private var lightThumbnailMockup: some View {
        ZStack {
            Color(red: 0.94, green: 0.94, blue: 0.96)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Circle().fill(Color.gray.opacity(0.4)).frame(width: 5, height: 5)
                    RoundedRectangle(cornerRadius: 2).fill(Color.gray.opacity(0.3)).frame(width: 28, height: 4)
                }
                .padding(.top, 6)
                .padding(.leading, 8)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white)
                    .padding(.horizontal, 6)
                    .overlay(
                        VStack(alignment: .leading, spacing: 3) {
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.4)).frame(width: 40, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.2)).frame(width: 55, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.gray.opacity(0.2)).frame(width: 32, height: 3)
                        }
                        .padding(6),
                        alignment: .topLeading
                    )
            }
        }
    }

    private var darkThumbnailMockup: some View {
        ZStack {
            Color(red: 0.12, green: 0.12, blue: 0.14)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Circle().fill(Color.gray.opacity(0.5)).frame(width: 5, height: 5)
                    RoundedRectangle(cornerRadius: 2).fill(Color.gray.opacity(0.4)).frame(width: 28, height: 4)
                }
                .padding(.top, 6)
                .padding(.leading, 8)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(red: 0.18, green: 0.18, blue: 0.20))
                    .padding(.horizontal, 6)
                    .overlay(
                        VStack(alignment: .leading, spacing: 3) {
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.4)).frame(width: 40, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.2)).frame(width: 55, height: 3)
                            RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.2)).frame(width: 32, height: 3)
                        }
                        .padding(6),
                        alignment: .topLeading
                    )
            }
        }
    }
}
