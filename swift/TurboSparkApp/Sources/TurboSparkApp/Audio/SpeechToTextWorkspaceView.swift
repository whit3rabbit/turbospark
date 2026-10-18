import SwiftUI

/// The Speech-to-Text page: a pill per job (Notes, Meeting, Any Audio), each
/// with its own list and live or detail view, and one record bar pinned in
/// the same place in every mode.
@MainActor
struct SpeechToTextWorkspaceView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var showingModels = false

    private var mode: SpeechToTextMode { controller.speechToTextMode }

    var body: some View {
        VStack(spacing: 0) {
            AppAdaptiveHeader {
                modePicker
            } actions: {
                modelButton
            }
            .padding(.horizontal, 24).padding(.vertical, 12)
            Divider()
            pane
                .id(mode)
                .transition(.opacity.combined(with: .offset(y: 8)))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            recordBar
                .padding(.horizontal, 24).padding(.vertical, 12)
        }
        .tsAnimation(TSMotion.pane, value: mode)
        .sheet(item: speakerNaming) { item in
            AudioSpeakerNamingSheet(controller: controller, item: item, purpose: .afterTake)
        }
    }

    /// The Meeting take whose final pass just labeled its speakers, while the
    /// Meeting pane shows and nothing records. Closing the sheet clears it.
    private var speakerNaming: Binding<AudioLibraryItem?> {
        Binding(get: {
            guard mode == .meeting, !controller.hasRecordingActivity, let id = controller.speakerNamingID else { return nil }
            return controller.items.first { $0.id == id }
        }, set: { if $0 == nil { controller.speakerNamingID = nil } })
    }

    private var modePicker: some View {
        AppPillPicker(
            selection: Binding(get: { controller.speechToTextMode }, set: { controller.selectSpeechToTextMode($0) }),
            items: SpeechToTextMode.allCases.map { mode in
                AppPillItem(
                    value: mode,
                    label: Text(LocalizedStringKey(mode.titleKey), bundle: .module),
                    systemImage: mode.symbol,
                    liveLabel: controller.liveSpeechToTextMode == mode ? Text("Recording", bundle: .module) : nil)
            },
            label: Text("Speech-to-Text mode", bundle: .module))
    }

    @ViewBuilder private var pane: some View {
        switch mode {
        case .notes, .meeting: SpeechToTextCapturePane(model: model, controller: controller, mode: mode)
        case .anyAudio: SpeechToTextAnyAudioPane(model: model, controller: controller)
        }
    }

    @ViewBuilder private var recordBar: some View {
        switch mode {
        case .notes, .meeting:
            AudioRecordBar(controller: controller, layout: .capsule) {
                AudioCaptureSourcePickers(controller: controller, style: .menus)
            }
        case .anyAudio:
            AudioRecordBar(controller: controller, layout: .capsule)
        }
    }

    private var modelButton: some View {
        Button { showingModels.toggle() } label: {
            Label {
                if let profile = controller.selectedProfile { Text(verbatim: profile.displayName) }
                else { Text("Choose a model", bundle: .module) }
            } icon: { Image(systemName: "cpu") }
            .lineLimit(1)
        }
        .appAction(.secondary, size: .small)
        .help(Text("Audio model", bundle: .module))
        .popover(isPresented: $showingModels, arrowEdge: .bottom) {
            SpeechToTextModelPanel(model: model, controller: controller, mode: mode)
        }
    }
}

/// A mode's list beside its detail, as an adjustable split when there is
/// room and stacked when there is not. The breakpoint scales with text size.
@MainActor
struct SpeechToTextSplit<Rows: View, Detail: View>: View {
    @ViewBuilder let list: () -> Rows
    @ViewBuilder let detail: () -> Detail
    @Environment(\.appTheme) private var theme

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        GeometryReader { geometry in
            if geometry.size.width >= 760 * scale {
                HSplitView {
                    ScrollView { list().padding(16) }
                        .frame(minWidth: 240 * scale, idealWidth: 300 * scale, maxWidth: 420 * scale)
                    ScrollView {
                        detail().padding(24).frame(maxWidth: 900, alignment: .leading).frame(maxWidth: .infinity)
                    }
                    .frame(minWidth: 360 * scale)
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) { list(); detail() }.padding(16)
                }
            }
        }
    }
}

/// The take a mode is recording: live words while streaming, the latest
/// draft with times otherwise, and the markers dropped so far.
@MainActor
struct SpeechToTextLiveTranscript: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(item.title).themedFont(.title3, weight: .semibold).accessibilityAddTraits(.isHeader)
                if !item.collection.isEmpty {
                    Text(item.collection).themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
            AudioLiveTextView(live: controller.live, maxHeight: 420)
            if !controller.live.isStreaming, let draft = item.preferredTranscript, !draft.segments.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(draft.segments) { segment in
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text(verbatim: audioTime(segment.start)).monospacedDigit()
                                .themedFont(.small).foregroundStyle(.appSecondary)
                            Text(verbatim: segment.text).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
            if !item.markers.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Markers", bundle: .module).themedFont(.small, weight: .semibold).accessibilityAddTraits(.isHeader)
                    ForEach(item.markers) { marker in
                        Label {
                            HStack { Text(verbatim: audioTime(marker.time)).monospacedDigit(); Text(verbatim: marker.title) }
                        } icon: { Image(systemName: "bookmark").accessibilityHidden(true) }
                        .themedFont(.small)
                    }
                }
                .accessibilityRotor(Text("Markers", bundle: .module)) {
                    ForEach(item.markers) { marker in
                        AccessibilityRotorEntry(marker.title + ", " + audioTime(marker.time), id: marker.id)
                    }
                }
            }
            AudioLiveTextModelRow(model: model, controller: controller)
        }
    }
}
