import SwiftUI

/// The approval card for a URL tool whose host the network policy asks
/// about (Settings > Network): the site, what wants to reach it, and the
/// decision scoped to that HOST, never to the tool name.
///
/// Deliberately no Return or Esc binding. The card sits in the transcript
/// while the composer can still hold focus, so a default-action Return
/// would grant a site in the middle of the user's next message.
@MainActor
struct NetworkAccessApprovalView: View {
    let host: String
    let toolLabel: String
    let onAllowForChat: () -> Void
    let onAllowOnce: () -> Void
    let onAlwaysAllow: () -> Void
    let onAlwaysBlock: () -> Void
    let onDeny: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "globe")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(verbatim: host)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)
                    .textSelection(.enabled)
            }
            .lineLimit(1)
            .truncationMode(.middle)
            .help(host)

            Text("Allow access to \(host)?", bundle: .module)
                .themedFont(.callout, weight: .semibold)
                .foregroundStyle(.appText)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            Text("The model wants \(toolLabel) to reach this site. You can change site access later in Settings > Network.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            // Wide: remember-forever on the leading edge, the per-chat answer
            // trailing, as in Codex. Narrower: the menu drops below. The last
            // option is what ViewThatFits uses when nothing fits, so it must
            // fit anything: one button per row (German at 360 pt clipped the
            // two-row form's primary label).
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    alwaysAllowMenu
                    Spacer(minLength: 8)
                    decisionButtons
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) { decisionButtons }
                    alwaysAllowMenu
                }
                VStack(alignment: .leading, spacing: 8) {
                    decisionButtons
                    alwaysAllowMenu
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCardSurface(padding: 10)
    }

    private var alwaysAllowMenu: some View {
        Menu {
            Button(action: onAlwaysBlock) { Text("Always Block \(host)", bundle: .module) }
        } label: {
            Text("Always Allow", bundle: .module)
        } primaryAction: {
            onAlwaysAllow()
        }
        .appMenuAction(size: .small)
        .help(Text("Always Allow \(host)", bundle: .module))
        .accessibilityLabel(Text("Always Allow \(host)", bundle: .module))
    }

    @ViewBuilder
    private var decisionButtons: some View {
        Button(role: .cancel, action: onDeny) { Text("Deny", bundle: .module) }
            .appAction(.secondary, size: .small)
            .help(Text("Deny this tool call execution", bundle: .module))

        Button(action: onAllowOnce) { Text("Allow Once", bundle: .module) }
            .appAction(.secondary, size: .small)
            .help(Text("Approve this tool call once", bundle: .module))

        Button(action: onAllowForChat) { Text("Allow for This Chat", bundle: .module) }
            .appAction(.primary, size: .small)
            .help(Text("Allow \(host) in This Chat", bundle: .module))
            .accessibilityLabel(Text("Allow \(host) in This Chat", bundle: .module))
    }
}
