import SwiftUI

/// The composer's Tools control: a chip that reports whether tools are on and
/// opens the popover where they are switched.
///
/// Shown in Chat and in Projects. The dot is `.appSuccess` when a turn would
/// be offered tools and `.appDanger` when it would not (Chat's master switch
/// off, or a profile with nothing enabled). It pairs a filled dot with a
/// slashed ring and a count so the state survives color blindness.
struct ToolsStatusChip: View {
    @ObservedObject var model: AppModel
    @State private var showingPopover = false

    var body: some View {
        let isOn = model.toolsAreOn
        let (count, total) = model.activeToolCounts
        let mcpApplies = !model.toolsContextIsRootless
        let isSubset = count < total || (mcpApplies && !model.activeToolProfile.includesMcpAndCustom)
        AppChip(
            isSubset
                ? Text("Tools \(count)/\(total)", bundle: .module)
                : Text("Tools", bundle: .module),
            systemImage: "wrench.and.screwdriver",
            isOn: isOn,
            indicator: isOn ? .on : .off
        ) {
            showingPopover.toggle()
        }
        .help(isOn
              ? Text("Tools are on. Click to choose which tools the model can use.", bundle: .module)
              : Text("Tools are off. Click to turn them on.", bundle: .module))
        .accessibilityValue(isOn
            ? Text("On, \(count) of \(total) tools", bundle: .module)
            : Text("Off", bundle: .module))
        .accessibilityHint(Text("Opens the tool list", bundle: .module))
        .popover(isPresented: $showingPopover, arrowEdge: .top) {
            ToolsPopoverView(model: model)
        }
    }
}

/// Master switch, profile picker and per-tool toggles.
///
/// Per-tool changes made here are a session override over the selected
/// profile: nothing is saved until "Save as profile", and Default is never
/// edited. Profiles themselves are managed in Settings.
struct ToolsPopoverView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @State private var savingName: String?

    private var masterOn: Bool {
        model.interactionMode != .chat || model.chatToolsEnabled
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if model.interactionMode == .chat {
                Toggle(isOn: Binding(
                    get: { model.chatToolsEnabled },
                    set: { value in
                        model.chatToolsEnabled = value
                        model.persistSettingsDebounced()
                    })) {
                    Text("Tools in Chat", bundle: .module)
                        .font(theme.ui(.callout, weight: .semibold))
                }
                .toggleStyle(.switch)
            }

            profilePicker

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(model.toolGroupsForContext) { group in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(LocalizedStringKey(group.title), bundle: .module)
                                .font(theme.ui(.small, weight: .semibold))
                                .foregroundStyle(.appSecondary)
                            ForEach(group.entries) { entry in
                                toolRow(entry)
                            }
                        }
                    }
                    // MCP servers and custom tools are Projects-only.
                    if !model.toolsContextIsRootless { mcpRow }
                }
            }
            .frame(maxHeight: 280)
            .disabled(!masterOn)
            .opacity(masterOn ? 1 : 0.5)

            Divider()

            Button {
                model.openSettings(tab: .tools)
            } label: {
                Text("Manage profiles...", bundle: .module)
            }
            .buttonStyle(.link)
        }
        .font(theme.ui(.small))
        .frame(width: 340, alignment: .leading)
        .padding(16)
    }

    private var profilePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Profile", bundle: .module)
                    .font(theme.ui(.small, weight: .semibold))
                Spacer()
                Picker(selection: Binding(
                    get: { model.selectedToolProfile.id },
                    set: { model.selectToolProfile($0) })) {
                    ForEach(model.allToolProfiles) { profile in
                        Text(profile.displayName).tag(profile.id)
                    }
                } label: {
                    Text("Profile", bundle: .module)
                }
                .labelsHidden()
                .fixedSize()
            }
            if model.toolProfileOverride != nil {
                modifiedRow
            }
        }
    }

    @ViewBuilder
    private var modifiedRow: some View {
        if let name = savingName {
            HStack(spacing: 6) {
                TextField(text: Binding(get: { name }, set: { savingName = $0 })) {
                    Text("Profile name", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .onSubmit(saveOverride)
                Button(action: saveOverride) { Text("Save", bundle: .module) }
                    .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button { savingName = nil } label: { Text("Cancel", bundle: .module) }
            }
        } else {
            HStack(spacing: 8) {
                Text("Changed for this session", bundle: .module)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Button { model.resetToolOverride() } label: { Text("Reset", bundle: .module) }
                    .buttonStyle(.link)
                Button { savingName = "" } label: { Text("Save as profile...", bundle: .module) }
                    .buttonStyle(.link)
            }
        }
    }

    private func saveOverride() {
        guard let name = savingName else { return }
        if model.saveToolOverride(as: name) != nil { savingName = nil }
    }

    private func toolRow(_ entry: ToolProfileEntry) -> some View {
        Toggle(isOn: Binding(
            get: { model.isToolEnabled(entry.id) },
            set: { model.setToolEnabled(entry.id, $0) })) {
            VStack(alignment: .leading, spacing: 1) {
                Text(entry.name)
                Text(entry.summary)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(2)
            }
        }
        .toggleStyle(.checkbox)
        .accessibilityElement(children: .combine)
    }

    private var mcpRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle(isOn: Binding(
                get: { model.activeToolProfile.includesMcpAndCustom },
                set: { model.setMcpAndCustomEnabled($0) })) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("MCP servers and custom tools", bundle: .module)
                    Text("Tools from connected servers, scripts and your own tool files.", bundle: .module)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                        .lineLimit(2)
                }
            }
            .toggleStyle(.checkbox)
            .accessibilityElement(children: .combine)
        }
    }
}
