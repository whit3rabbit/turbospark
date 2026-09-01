import AppKit
import SwiftUI

/// Preferences card view for dock icon, pointer cursors, font smoothing, and sizing.
public struct AppearancePreferencesCardView: View {
    @ObservedObject public var manager: AppearanceManager

    public init(manager: AppearanceManager) {
        self.manager = manager
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Preferences")
                .font(.headline.weight(.semibold))
                .padding(.horizontal, 16)
                .padding(.vertical, 14)

            Divider()

            // 1. Use pointer cursors
            preferenceRow(
                title: "Use pointer cursors",
                description: "Change the cursor to a pointer when hovering over interactive elements"
            ) {
                Toggle("", isOn: $manager.usePointerCursors)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .appPointerCursor()
            }

            Divider().padding(.leading, 16)

            // 2. Status bar benchmarks view mode
            preferenceRow(
                title: "Status bar benchmarks",
                description: "Display bottom toolbar metrics as numeric text or live sparkline graphs"
            ) {
                Picker("", selection: $manager.statusBarViewMode) {
                    ForEach(StatusBarViewMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 175)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 3. Dock icon
            preferenceRow(
                title: "Dock icon",
                description: "Choose the icon the app will use in the dock"
            ) {
                HStack(spacing: 8) {
                    dockIconOption(.emeraldSpark, systemSymbol: "sparkles", gradientColors: [Color(red: 0.1, green: 0.3, blue: 0.15), Color(red: 0.05, green: 0.15, blue: 0.08)])
                    dockIconOption(.codexDark, systemSymbol: "chevron.left.forwardslash.chevron.right", gradientColors: [Color(white: 0.25), Color(white: 0.12)])
                    dockIconOption(.terminalPro, systemSymbol: "terminal.fill", gradientColors: [Color(red: 0.1, green: 0.25, blue: 0.5), Color(red: 0.05, green: 0.1, blue: 0.25)])
                    dockIconOption(.minimalist, systemSymbol: "bolt.fill", gradientColors: [Color(white: 0.15), Color(white: 0.08)])
                }
            }

            Divider().padding(.leading, 16)

            // 3. Reduce motion
            preferenceRow(
                title: "Reduce motion",
                description: "Reduce animations or match your system"
            ) {
                Picker("", selection: $manager.reduceMotion) {
                    ForEach(ReduceMotionPreference.allCases) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                .labelsHidden()
            }

            Divider().padding(.leading, 16)

            // 4. UI font size
            preferenceRow(
                title: "UI font size",
                description: "Adjust the base size used for the TurboSpark UI"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.uiFontSize, in: 11...20, step: 1)
                        .labelsHidden()

                    Text("\(Int(manager.uiFontSize))")
                        .font(.subheadline.monospacedDigit())
                        .frame(width: 24, alignment: .center)

                    Text("px")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
                )
            }

            Divider().padding(.leading, 16)

            // 5. Code font size
            preferenceRow(
                title: "Code font size",
                description: "Adjust the base size used for code across chats and diffs"
            ) {
                HStack(spacing: 6) {
                    Stepper("", value: $manager.codeFontSize, in: 10...18, step: 1)
                        .labelsHidden()

                    Text("\(Int(manager.codeFontSize))")
                        .font(.subheadline.monospacedDigit())
                        .frame(width: 24, alignment: .center)

                    Text("px")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
                )
            }

            Divider().padding(.leading, 16)

            // 6. Diff markers
            preferenceRow(
                title: "Diff markers",
                description: "Show changes using colors or +/- markers"
            ) {
                Picker("", selection: $manager.diffMarkers) {
                    ForEach(DiffMarkerPreference.allCases) { opt in
                        Text(opt.label).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
                .labelsHidden()
            }

        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
        )
    }

    private func preferenceRow<Accessory: View>(
        title: String,
        description: String,
        @ViewBuilder accessory: () -> Accessory
    ) -> some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)

                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 16)
            accessory()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func dockIconOption(_ icon: AppDockIcon, systemSymbol: String, gradientColors: [Color]) -> some View {
        let isSelected = manager.dockIcon == icon
        return Button {
            manager.dockIcon = icon
        } label: {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(LinearGradient(colors: gradientColors, startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 32, height: 32)

                Image(systemName: systemSymbol)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)

                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(
                        isSelected ? Color.accentColor : Color.clear,
                        lineWidth: isSelected ? 2 : 0
                    )
            }
            .frame(width: 32, height: 32)
        }
        .buttonStyle(.plain)
        .help(icon.label)
        .appPointerCursor()
    }
}
