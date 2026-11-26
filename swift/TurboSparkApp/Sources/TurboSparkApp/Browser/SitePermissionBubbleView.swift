import SwiftUI

/// Localized names for the governed classes. Literals sit at the call site so
/// the localization source scan sees them.
enum SitePermissionText {
    static func name(_ permissionClass: SitePermissionClass) -> Text {
        switch permissionClass {
        case .camera: return Text("Camera", bundle: .module)
        case .microphone: return Text("Microphone", bundle: .module)
        case .geolocation: return Text("Location", bundle: .module)
        }
    }

    static func decision(_ decision: SitePermissionAuditDecision) -> Text {
        switch decision {
        case .allowOnce: return Text("Allowed Once", bundle: .module)
        case .allowSite, .allowRecord: return Text("Allowed for Site", bundle: .module)
        case .denyOnce: return Text("Denied Once", bundle: .module)
        case .denySite, .denyRecord: return Text("Denied for Site", bundle: .module)
        case .autoDeny: return Text("Denied Automatically", bundle: .module)
        case .recordRemoved: return Text("Record Removed", bundle: .module)
        }
    }
}

/// The confirmation for a camera, microphone or location request in a
/// user-owned tab. It is deny-first: the default and Escape both deny once,
/// and the countdown denies when it runs out. The origin lines are rendered
/// here from the canonical origin, never from page text.
@MainActor
struct SitePermissionBubbleView: View {
    let prompt: SitePermissionPrompt
    let answer: @MainActor (SitePermissionAnswer) -> Void
    @Environment(\.appTheme) private var theme

    var body: some View {
        ZStack {
            Color.black.opacity(0.28)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 14) {
                Text("Site permission request", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Requested by \(prompt.requestingOrigin.canonicalString)", bundle: .module)
                        .themedCode(.small, weight: .semibold)
                        .textSelection(.enabled)
                    if prompt.showsPageOrigin, let page = prompt.topLevelOrigin {
                        Text("On page \(page.canonicalString)", bundle: .module)
                            .themedCode(.small)
                            .foregroundStyle(.appSecondary)
                            .textSelection(.enabled)
                    }
                }
                .accessibilityElement(children: .combine)

                VStack(alignment: .leading, spacing: 4) {
                    ForEach(prompt.classes, id: \.self) { permissionClass in
                        SitePermissionText.name(permissionClass)
                    }
                }
                .accessibilityElement(children: .combine)

                countdown

                VStack(spacing: 8) {
                    HStack {
                        Button {
                            answer(.denyOnce)
                        } label: {
                            Text("Deny Once", bundle: .module)
                        }
                        .appAction(.primary)
                        .keyboardShortcut(.defaultAction)

                        Spacer(minLength: 8)

                        Button {
                            answer(.allowOnce)
                        } label: {
                            Text("Allow Once", bundle: .module)
                        }
                        .appAction(.secondary)
                    }
                    HStack {
                        Button {
                            answer(.denySite)
                        } label: {
                            Text("Deny for This Site", bundle: .module)
                        }
                        .appAction(.secondary)

                        Spacer(minLength: 8)

                        Button {
                            answer(.allowSite)
                        } label: {
                            Text("Allow for This Site", bundle: .module)
                        }
                        .appAction(.secondary)
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: 440, alignment: .leading)
            .appGlassSurface(.rect(cornerRadius: AppMetrics.cardRadius))
            .shadow(radius: 24)
            .padding(24)
            .accessibilityAddTraits(.isModal)
            .onExitCommand { answer(.denyOnce) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityIdentifier("browser-site-permission-bubble")
        .onAppear {
            _ = AccessibilityNotification.Announcement.post(
                .init(AppStrings.localized("Site permission request")))
        }
    }

    /// The fill shows time left. It ticks once a second and never animates a
    /// sweep, so Reduce Motion needs no special case.
    private var countdown: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let remaining = max(0, prompt.expiresAt.timeIntervalSince(context.date))
            VStack(alignment: .leading, spacing: 6) {
                AppProgressBar(
                    value: prompt.duration > 0 ? remaining / prompt.duration : 0,
                    label: Text("Denies automatically if there is no answer.", bundle: .module),
                    valueText: Text(verbatim: "\(Int(remaining.rounded(.up)))"),
                    size: .compact,
                    animated: false
                )
                Text("Denies automatically if there is no answer.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        }
    }
}
