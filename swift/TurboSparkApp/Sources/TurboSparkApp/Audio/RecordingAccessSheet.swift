import AppKit
import SwiftUI

/// What the sheet's primary action does once every needed grant is in.
enum RecordingAccessPurpose: Identifiable, Hashable {
    /// Opened by Start recording: finishing starts the take.
    case start
    /// Opened by the check button: finishing refreshes the device lists.
    case review

    var id: Self { self }
}

/// A capture waiting on macOS access. `onReady` runs after the sheet closes
/// with the sources the user chose to record without (empty when every
/// needed grant is in).
struct RecordingAccessRequest: Identifiable {
    let id = UUID()
    let purpose: RecordingAccessPurpose
    let needed: [RecordingPermission]
    let onReady: (Set<RecordingPermission>) -> Void
}

extension View {
    /// The one way a control that starts a capture asks for access: set the
    /// binding when `missing` is not empty, start directly when it is. The
    /// capture layer only checks access, so a path that skips this gate gets
    /// an error, never an unexplained macOS prompt. Pass
    /// `RecordingPermissions.shared`; tests pass their own.
    @MainActor
    func recordingAccessGate(_ request: Binding<RecordingAccessRequest?>,
                             permissions: RecordingPermissions) -> some View {
        sheet(item: request) { request in
            RecordingAccessSheet(permissions: permissions, request: request)
        }
        // A grant changed in System Settings shows up when the user switches back.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            permissions.refresh()
        }
    }
}

/// Recording access: one row per macOS grant with its status and next step,
/// beside how TurboSpark uses the audio. Shown before a recording whose
/// sources lack a grant, so a missing permission is fixed up front instead
/// of failing partway through capture setup. Access is optional: the sheet
/// offers to record with the allowed sources, and Not now leaves the rest
/// of the app usable. The columns stack when the sheet is narrow.
@MainActor
struct RecordingAccessSheet: View {
    @ObservedObject var permissions: RecordingPermissions
    let request: RecordingAccessRequest
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme

    /// Column widths and the sheet grow with the text size, so Extra Large
    /// text stacks the columns instead of squeezing both. `HumanizeMetrics`
    /// holds the committed spelling of the shared text scale.
    private var scale: CGFloat { HumanizeMetrics.textScale(theme) }

    private var needed: [RecordingPermission] { request.needed }
    private var purpose: RecordingAccessPurpose { request.purpose }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    // Ideal widths are pinned so the two-column layout is
                    // chosen by sheet width, not by how long a translated
                    // sentence is when laid out on one line.
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 20) {
                            rows.frame(minWidth: 320 * scale, idealWidth: 400 * scale, maxWidth: .infinity)
                            RecordingAudioUseCard()
                                .frame(minWidth: 240 * scale, idealWidth: 280 * scale, maxWidth: 340 * scale)
                        }
                        VStack(alignment: .leading, spacing: 20) {
                            rows
                            RecordingAudioUseCard()
                        }
                    }
                }
                .padding(24)
            }
            footer
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
        }
        .frame(minWidth: 520, idealWidth: 760 * scale, maxWidth: 900 * scale, minHeight: 380, idealHeight: 560 * scale)
        .background(.appPage)
        .themedFont(.base)
        .onAppear { permissions.refresh() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Allow recording access", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Text("macOS asks once for each source you record. Nothing is recorded until you press Start recording.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Access is optional. Without it you can still import, transcribe, and edit audio files.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var rows: some View {
        VStack(spacing: 10) {
            ForEach(RecordingPermission.allCases) { permission in
                RecordingPermissionRow(
                    permission: permission,
                    state: permissions.state(permission),
                    isNeeded: needed.contains(permission),
                    isBusy: permissions.isRequesting
                ) {
                    Task { await permissions.request(permission) }
                }
            }
        }
    }

    private var missing: [RecordingPermission] { permissions.missing(needed) }
    private var blockedByPolicy: Bool { missing.contains { permissions.state($0) == .restricted } }

    private var footer: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: AppMetrics.actionSpacing) {
                footerNote
                Spacer(minLength: 12)
                footerActions
            }
            VStack(alignment: .leading, spacing: 10) {
                footerNote
                HStack(spacing: AppMetrics.actionSpacing) { footerActions }
            }
        }
    }

    /// Says why the primary action is disabled when a grant is out of reach.
    @ViewBuilder private var footerNote: some View {
        if blockedByPolicy {
            Text("A device policy blocks a selected source. Turn that source off to record without it.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var footerActions: some View {
        Button { dismiss() } label: { Text("Not now", bundle: .module) }
            .appAction(.secondary)
            .keyboardShortcut(.cancelAction)
        if purpose == .start, let fallback = permissions.fallback(needed) {
            Button {
                dismiss()
                request.onReady(Set(fallback.declined))
            } label: {
                if fallback.allowed == [.microphone] {
                    Text("Record microphone only", bundle: .module)
                } else {
                    Text("Record app or system audio only", bundle: .module)
                }
            }
            .appAction(.secondary)
        }
        if missing.isEmpty {
            Button {
                dismiss()
                request.onReady([])
            } label: {
                if purpose == .start {
                    Label { Text("Start recording", bundle: .module) } icon: { Image(systemName: "record.circle") }
                } else {
                    Text("Done", bundle: .module)
                }
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
        } else {
            // The primary action is the next step for the first missing
            // grant: a prompt while macOS can still show one, System
            // Settings once it cannot.
            Button {
                Task { await permissions.requestAll(needed) }
            } label: {
                if missing.first.map(permissions.state) == .denied {
                    Text("Open System Settings", bundle: .module)
                } else {
                    Text("Allow access", bundle: .module)
                }
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
            .disabled(permissions.isRequesting || blockedByPolicy)
        }
    }
}

/// One grant: what it is for, its status, and the one action that moves it.
@MainActor
struct RecordingPermissionRow: View {
    let permission: RecordingPermission
    let state: RecordingPermissionState
    let isNeeded: Bool
    let isBusy: Bool
    let action: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            info.accessibilityElement(children: .combine)
            Spacer(minLength: 8)
            rowAction
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        }
    }

    private var info: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: permission.symbol)
                .themedFont(.large)
                .foregroundStyle(.appAccent)
                .frame(minWidth: 24, minHeight: 24)
                .padding(8)
                .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                permission.title.themedFont(.base, weight: .semibold)
                permission.detail
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Label { statusText } icon: { Image(systemName: statusSymbol).accessibilityHidden(true) }
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(state == .granted ? .appAccent : .appSecondary)
                    .padding(.top, 2)
                ForEach(Array(notes.enumerated()), id: \.offset) { _, note in
                    note
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var statusText: Text {
        switch state {
        case .granted: return Text("Allowed", bundle: .module)
        case .notDetermined: return Text("Not allowed yet", bundle: .module)
        case .denied: return Text("Off in System Settings", bundle: .module)
        case .restricted: return Text("Blocked by a device policy", bundle: .module)
        }
    }

    private var statusSymbol: String {
        switch state {
        case .granted: return "checkmark.circle.fill"
        case .notDetermined: return "circle.dashed"
        case .denied: return "exclamationmark.triangle.fill"
        case .restricted: return "xmark.octagon.fill"
        }
    }

    private var notes: [Text] {
        guard state != .granted else { return [] }
        var notes: [Text] = []
        if !isNeeded {
            notes.append(Text("Not needed for the sources you selected.", bundle: .module))
        }
        switch state {
        case .denied:
            notes.append(Text("Turn on TurboSpark in System Settings, then return here.", bundle: .module))
        case .restricted:
            notes.append(Text("An administrator or Screen Time controls this setting.", bundle: .module))
        case .granted, .notDetermined:
            break
        }
        // A new screen recording grant can need a relaunch before
        // ScreenCaptureKit sees it; macOS offers Quit and Reopen itself.
        if permission == .systemAudio, isNeeded {
            notes.append(Text("macOS may ask you to quit and reopen TurboSpark before the change applies.", bundle: .module))
        }
        return notes
    }

    /// The visible label is a single word; VoiceOver hears which grant it
    /// moves, because two rows can show the same action.
    @ViewBuilder private var rowAction: some View {
        switch state {
        case .notDetermined:
            Button(action: action) { Text("Allow", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(isBusy)
                .accessibilityLabel(Text("Allow", bundle: .module) + Text(verbatim: ", ") + permission.title)
        case .denied:
            Button(action: action) { Text("Open System Settings", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(isBusy)
                .help(Text("Opens this pane in macOS System Settings", bundle: .module))
                .accessibilityLabel(Text("Open System Settings", bundle: .module) + Text(verbatim: ", ") + permission.title)
        case .granted, .restricted:
            EmptyView()
        }
    }
}

/// How recorded audio is used. Every line is a claim the app keeps: local
/// transcription, the encrypted profile store, and excluding its own playback.
@MainActor
struct RecordingAudioUseCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            AppSectionHeader(title: Text("How TurboSpark uses audio", bundle: .module))
            point("cpu", Text("Audio is transcribed on this Mac by the model you choose.", bundle: .module))
            point("lock", Text("Audio and edits are saved in your encrypted profile.", bundle: .module))
            point("speaker.slash", Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module))
            point("text.badge.checkmark", Text("Transcripts are generated by a model. Review them for accuracy.", bundle: .module))
            point("person.2", Text("Tell others before you record a meeting. Some places require their consent.", bundle: .module))
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        }
    }

    private func point(_ symbol: String, _ text: Text) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(.appAccent)
                .frame(minWidth: 18)
                .accessibilityHidden(true)
            text.fixedSize(horizontal: false, vertical: true)
        }
        .themedFont(.small)
        .accessibilityElement(children: .combine)
    }
}
