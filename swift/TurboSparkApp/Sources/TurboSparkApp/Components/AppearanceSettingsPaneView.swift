import AppKit
import SwiftUI

public struct AppearanceSettingsPaneView: View {
    @Environment(\.settingsTarget) private var settingsTarget
    @Environment(\.appTheme) private var theme
    @ObservedObject private var manager = AppearanceManager.shared
    @Environment(\.colorScheme) private var colorScheme

    @State private var showsCustomization = false
    @State private var showsResetConfirmation = false

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
                ThemeGalleryView(manager: manager)

                ThemeTypographyPreviewView(manager: manager, isDark: isCurrentlyDark)

                ThemeCodePreviewView(manager: manager, isDark: isCurrentlyDark)

                DisclosureGroup(isExpanded: $showsCustomization) {
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

                } label: { Text("Customize colors", bundle: .module)
                    .settingsControl("Customize colors", pane: .appearance, timing: .immediate) }

                AppearancePreferencesCardView(manager: manager)

                resetSection
            }
            .padding(20)
        }
        .background(.appPage.opacity(0.6))
        .onChange(of: settingsTarget, initial: true) { _, target in
            let colorTargets = ["Customize colors", "Accent", "Background", "Foreground", "Contrast", "Translucent sidebar"]
            if colorTargets.contains(where: { target == "appearance:" + $0 }) { showsCustomization = true }
        }
    }

    // MARK: - Reset to Defaults

    /// The one description of what a reset does, shared by the card's
    /// caption and the confirmation dialog's message so the two cannot
    /// drift apart.
    private var resetDescription: Text {
        Text("Restores theme mode, colors, fonts, text size, dock icon, and every preference on this pane. Saved themes are kept.", bundle: .module)
    }

    private var resetSection: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Reset Appearance", bundle: .module)
                    .settingsControl("Reset Appearance", pane: .appearance, timing: .immediate)
                    .font(theme.ui(.base, weight: .medium))
                    .foregroundStyle(.appText)
                resetDescription
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button("Reset to Defaults...") {
                showsResetConfirmation = true
            }
        }
        .padding(16)
        .background(.appSurface.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
        )
        .confirmationDialog(
            Text("Reset Appearance", bundle: .module),
            isPresented: $showsResetConfirmation,
            titleVisibility: .visible
        ) {
            Button("Reset to Defaults", role: .destructive) {
                manager.resetToDefaults()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            resetDescription
        }
    }

    // MARK: - Theme Mode Selector
    private var themeSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Appearance mode", bundle: .module)
                    .settingsControl("Appearance mode", pane: .appearance, timing: .immediate)
                .font(theme.ui(.large, weight: .semibold))
                .foregroundStyle(.appText)

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
                        .fill(.appSurface)

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
                    .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .primary : .secondary)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
        .appPointerCursor()
    }

    private var lightThumbnailMockup: some View { ThemeModeThumbnail(config: manager.lightConfig) }
    private var darkThumbnailMockup: some View { ThemeModeThumbnail(config: manager.darkConfig) }
}
