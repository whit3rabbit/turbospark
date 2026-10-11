import AppKit
import SwiftUI

/// The pieces every API mode (Text, Image, Audio, TypeSafe) is drawn from.
///
/// The 2026-10-11 audit found each mode spelling the same controls its own
/// way: Start was a large prominent button in Text and a regular one in
/// TypeSafe, Attach and Detach were unstyled while Load and Unload were bare
/// text buttons, and each test editor drew its own border. One mapping now
/// holds in every mode:
///
/// - Verbs that turn something on (Start, Attach, Load) are `.primary`;
///   their inverses (Stop, Detach, Unload) are `.secondary`.
/// - A verb sits in the same place in every mode: Start/Stop in the status
///   band, Attach in the Serving panel header, Detach on each served-model
///   row, Load/Unload on each TypeSafe model row. Band and panel controls
///   are `.regular`; row controls are `.small`.
enum APIControlSize {
    static let panel: ControlSize = .regular
    static let row: ControlSize = .small
}

/// The band at the top of every mode: is it on, what is it doing, and the
/// one Start/Stop control. TypeSafe draws its own service through the same
/// band so the two servers look alike while staying labeled apart.
struct APIStatusBand<Details: View>: View {
    let color: Color
    let title: Text
    let subtitle: Text
    let isRunning: Bool
    let busy: Bool
    let startLabel: Text
    let stopLabel: Text
    let startDisabled: Bool
    let stopDisabled: Bool
    let start: () -> Void
    let stop: () -> Void
    @ViewBuilder let details: () -> Details

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            AppAdaptiveHeader {
                HStack(spacing: 12) {
                    Circle()
                        .fill(color)
                        .frame(width: 10, height: 10)
                        .overlay(Circle().stroke(color.opacity(0.25), lineWidth: 6))
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        title.themedFont(.callout, weight: .semibold)
                        subtitle
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .accessibilityElement(children: .combine)
            } actions: {
                if busy {
                    ProgressView().controlSize(.small)
                }
                if isRunning {
                    Button(action: stop) { stopLabel }
                        .appAction(.secondary, size: APIControlSize.panel)
                        .disabled(stopDisabled)
                } else {
                    Button(action: start) { startLabel }
                        .appAction(.primary, size: APIControlSize.panel)
                        .disabled(startDisabled)
                }
            }
            details()
        }
        .apiCard()
    }
}

/// A titled card holding one part of a mode (Serving, Test request, Models).
struct APIPanel<Actions: View, Content: View>: View {
    let title: Text
    var subtitle: Text?
    @ViewBuilder let actions: () -> Actions
    @ViewBuilder let content: () -> Content

    init(
        title: Text, subtitle: Text? = nil,
        @ViewBuilder actions: @escaping () -> Actions,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.actions = actions
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader {
                AppSectionHeader(title: title, subtitle: subtitle)
            } actions: {
                actions()
            }
            content()
        }
        .apiCard()
    }
}

extension APIPanel where Actions == EmptyView {
    init(title: Text, subtitle: Text? = nil, @ViewBuilder content: @escaping () -> Content) {
        self.init(title: title, subtitle: subtitle, actions: { EmptyView() }, content: content)
    }
}

extension View {
    /// The card treatment shared by every API panel and band.
    func apiCard() -> some View {
        padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 0.5))
    }
}

/// One "label value" status reading (Attached, Progress, Health). Combined
/// for VoiceOver so the label and the reading are announced together.
struct APIStatusLine: View {
    let label: Text
    let value: Text

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            label.foregroundStyle(.appSecondary)
            value.textSelection(.enabled)
        }
        .themedFont(.small)
        .accessibilityElement(children: .combine)
    }
}

/// Status words the model layer stores as English `String`s (State/ is a
/// separate partition). Known words and the two known shapes resolve through
/// the catalog; anything else (an engine error, a step count beside a model
/// name, a health word from the daemon) renders exactly as given.
enum APIStatusText {
    static let knownWords: Set<String> = [
        "Idle", "Ready", "Cancelled", "Loading speech model", "Synthesizing",
        "Running", "Stopped", "Error",
    ]

    static func text(_ raw: String) -> Text {
        if knownWords.contains(raw) {
            return Text(LocalizedStringKey(raw), bundle: .module)
        }
        if raw.hasPrefix("Failed: ") {
            let reason = String(raw.dropFirst("Failed: ".count))
            return Text("Failed: \(reason)", bundle: .module)
        }
        if raw.hasPrefix("Synthesizing ("), raw.hasSuffix(" tokens)"),
           let count = Int(raw.dropFirst("Synthesizing (".count).dropLast(" tokens)".count)) {
            return Text("Synthesizing (\(String(count)) tokens)", bundle: .module)
        }
        return Text(verbatim: raw)
    }
}

/// The run feedback every test panel shows: a spinner while the request is
/// in flight, then the wall time it took.
struct APIRunIndicator: View {
    let running: Bool
    let elapsed: Double?

    var body: some View {
        if running {
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel(Text("Running", bundle: .module))
        } else if let elapsed {
            Text(verbatim: String(format: "%.1f s", elapsed))
                .themedFont(.small)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
        }
    }
}

/// A read-only request or response body, on the same surface as the test
/// editors so request and response read as one pair.
struct APICodeBlock: View {
    let code: String

    var body: some View {
        Text(verbatim: code)
            .themedCode(.small)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .appEditorSurface()
    }
}

/// The route a mode answers plus an example body, shown once the shared
/// server is up.
struct APIRequestPreview: View {
    let method: String
    let url: String
    let example: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(verbatim: method + " " + url)
                .themedCode(.callout)
                .textSelection(.enabled)
            APICodeBlock(code: example)
        }
    }
}

/// The list frame the served-model rows sit in, in every mode.
struct APIRowList<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(spacing: 0) { content() }
            .background(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .fill(.appPage.opacity(0.6)))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 0.5))
    }
}

/// The Detach control on a served-model row: the same verb, role, size and
/// symbol in Text, Image and Audio.
struct APIDetachButton: View {
    let help: Text
    var disabled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label { Text("Detach", bundle: .module) } icon: { Image(systemName: "eject") }
                .labelStyle(.titleAndIcon)
        }
        .appAction(.secondary, size: APIControlSize.row)
        .disabled(disabled)
        .help(help)
    }
}

/// Copy-to-clipboard with a brief checkmark, named for VoiceOver.
struct APICopyButton: View {
    let label: Text
    let value: () -> String?
    @State private var copied = false

    var body: some View {
        Button {
            guard let text = value(), !text.isEmpty else { return }
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            copied = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
        } label: {
            Label { label } icon: { Image(systemName: copied ? "checkmark" : "doc.on.doc") }
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.borderless)
        .help(label)
        .accessibilityValue(copied ? Text("Copied", bundle: .module) : Text(verbatim: ""))
    }
}

/// One model the server is serving, drawn the same in Text, Image and
/// Audio: a live dot, the id clients send, a detail line, then Detach.
struct APIServedModelRow<Details: View, Trailing: View>: View {
    let name: String
    var badge: Text?
    /// nil means the theme accent, resolved in `body` (main actor).
    var badgeTint: Color?
    let detachHelp: Text
    var detachDisabled = false
    let detach: () -> Void
    @ViewBuilder let details: () -> Details
    @ViewBuilder let trailing: () -> Trailing

    var body: some View {
        HStack(spacing: 12) {
            Circle().fill(Color.green).frame(width: 7, height: 7)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(verbatim: name)
                        .themedCode(.base)
                        .textSelection(.enabled)
                    if let badge {
                        badge
                            .themedFont(.micro, weight: .medium)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background((badgeTint ?? TurboSparkTheme.accentColor).opacity(0.15), in: Capsule())
                    }
                }
                details()
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            trailing()
            APIDetachButton(help: detachHelp, disabled: detachDisabled, action: detach)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }
}

extension APIServedModelRow where Trailing == EmptyView {
    init(
        name: String, badge: Text? = nil, badgeTint: Color? = nil,
        detachHelp: Text, detachDisabled: Bool = false, detach: @escaping () -> Void,
        @ViewBuilder details: @escaping () -> Details
    ) {
        self.init(
            name: name, badge: badge, badgeTint: badgeTint, detachHelp: detachHelp,
            detachDisabled: detachDisabled, detach: detach, details: details, trailing: { EmptyView() })
    }
}

/// The placeholder row of an empty Serving list.
struct APINothingServedRow: View {
    var body: some View {
        Text("Nothing attached yet.", bundle: .module)
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
    }
}
