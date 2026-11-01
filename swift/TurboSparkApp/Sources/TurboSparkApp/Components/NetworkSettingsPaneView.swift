import SwiftUI

/// Settings > Network: which hosts the model's tool calls may reach.
///
/// A grouped Form, one computed property per `Section` (DESIGN.md, Forms).
/// The coverage section is part of the feature, not decoration: Seatbelt
/// cannot filter shell traffic by domain, so the pane states exactly what
/// each mode enforces rather than letting "Block all" imply more.
@MainActor
struct NetworkSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @State private var allowedDraft = ""
    @State private var blockedDraft = ""
    @State private var invalidEntry = false

    private var policy: AppNetworkPolicy { model.networkPolicy }

    var body: some View {
        Form {
            modeSection
            localSection
            if policy.mode != .blockAll {
                domainsSection
            }
            if policy.mode == .custom {
                customSection
            }
            coverageSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
    }

    // MARK: - Sections

    private var modeSection: some View {
        Section {
            Text("Choose which hosts the model's tool calls can reach.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            Picker(selection: binding(\.mode)) {
                ForEach(AppNetworkMode.allCases) { mode in
                    mode.labelText.tag(mode)
                }
            } label: {
                Text("Network access", bundle: .module)
            }
            .pickerStyle(.radioGroup)
            .settingsControl("Network access", pane: .network, timing: .nextTurn)

            modeDescription
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var modeDescription: Text {
        switch policy.mode {
        case .ask:
            return Text("Allowed domains run. Any other domain asks first, and you can allow it once, for the chat, or always.", bundle: .module)
        case .allowAll:
            return Text("Every public domain runs except the blocked list.", bundle: .module)
        case .blockAll:
            return Text("Tool calls cannot reach the network. Shell commands run in a macOS sandbox with networking turned off.", bundle: .module)
        case .custom:
            return Text("Allowed domains run, blocked domains are refused, and you choose what happens to every other domain.", bundle: .module)
        }
    }

    private var localSection: some View {
        Section(header: Text("Local access", bundle: .module)) {
            Toggle(isOn: binding(\.allowLocalhost)) {
                Text("Allow localhost", bundle: .module)
                Text("Dev servers and APIs on this Mac. Applies in every mode, including Block all.", bundle: .module)
            }
            .settingsControl("Allow localhost", pane: .network, timing: .nextTurn)

            Toggle(isOn: binding(\.allowLocalNetwork)) {
                Text("Allow local network", bundle: .module)
                Text("Routers, NAS drives, and other devices on your network.", bundle: .module)
            }
            .settingsControl("Allow local network", pane: .network, timing: .nextTurn)

            Label {
                Text("Cloud metadata endpoints are always blocked.", bundle: .module)
            } icon: {
                Image(systemName: "lock.shield")
            }
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
        }
    }

    private var domainsSection: some View {
        Section {
            NetworkDomainListView(
                title: Text("Allowed domains", bundle: .module),
                domains: policy.allowedDomains,
                draft: $allowedDraft,
                onAdd: { add($0, allow: true) },
                onRemove: { model.removeNetworkDomain($0, allow: true) }
            )
            .settingsControl("Allowed domains", pane: .network, timing: .nextTurn)

            NetworkDomainListView(
                title: Text("Blocked domains", bundle: .module),
                domains: policy.deniedDomains,
                draft: $blockedDraft,
                onAdd: { add($0, allow: false) },
                onRemove: { model.removeNetworkDomain($0, allow: false) }
            )
            .settingsControl("Blocked domains", pane: .network, timing: .nextTurn)

            if invalidEntry {
                Label {
                    Text("Enter a host name such as example.com or *.example.com, without a scheme, port, or path.", bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .themedFont(.small)
                .foregroundStyle(.appWarning)
            }
        } header: {
            Text("Domains", bundle: .module)
        } footer: {
            Text("A domain also covers its subdomains. Blocked domains win over allowed ones.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    private var customSection: some View {
        Section(header: Text("Custom rules", bundle: .module)) {
            Picker(selection: binding(\.customUnlisted)) {
                Text("Ask Before Execution", bundle: .module).tag(AppToolPermission.ask)
                Text("Deny", bundle: .module).tag(AppToolPermission.deny)
            } label: {
                Text("Other domains", bundle: .module)
            }
            .pickerStyle(.menu)

            Toggle(isOn: binding(\.readOnlyMethods)) {
                Text("Read-only requests only", bundle: .module)
                Text("Refuses HTTP requests other than GET, HEAD, and OPTIONS.", bundle: .module)
            }

            Button {
                for domain in AppNetworkPolicy.commonDeveloperDomains {
                    model.addNetworkDomain(domain, allow: true)
                }
            } label: {
                Label {
                    Text("Add common developer domains", bundle: .module)
                } icon: {
                    Image(systemName: "plus.circle")
                }
            }
            .appAction(.secondary, size: .small)
            .help(Text("Adds GitHub, package registries, Hugging Face, and documentation hosts to the allowed list", bundle: .module))
        }
    }

    private var coverageSection: some View {
        Section(header: Text("What these settings cover", bundle: .module)) {
            Text("Web fetch, HTTP requests, web search, remote MCP servers, and agent browser navigation follow these settings.", bundle: .module)
            Text("Shell commands and custom tools are cut off from the network only under Block all. In the other modes they are not filtered by domain.", bundle: .module)
            Text("Local MCP servers, plugins, and hooks are not covered.", bundle: .module)
        }
        .themedFont(.small)
        .foregroundStyle(.appSecondary)
    }

    // MARK: - Helpers

    private func binding<Value>(_ keyPath: WritableKeyPath<AppNetworkPolicy, Value>) -> Binding<Value> {
        Binding(
            get: { model.networkPolicy[keyPath: keyPath] },
            set: { value in model.updateNetworkPolicy { $0[keyPath: keyPath] = value } }
        )
    }

    private func add(_ entry: String, allow: Bool) -> Bool {
        let added = model.addNetworkDomain(entry, allow: allow)
        invalidEntry = !added
        return added
    }
}

extension AppNetworkMode {
    /// The one label for each mode, shared by the Network pane and the chat
    /// status sheet.
    var labelText: Text {
        switch self {
        case .ask: return Text("Ask for new domains", bundle: .module)
        case .allowAll: return Text("Allow all", bundle: .module)
        case .blockAll: return Text("Block all", bundle: .module)
        case .custom: return Text("Custom", bundle: .module)
        }
    }
}

/// One editable host list: the entries with a remove action each, and a
/// field that adds on Return or the Add button.
@MainActor
struct NetworkDomainListView: View {
    let title: Text
    let domains: [String]
    @Binding var draft: String
    let onAdd: (String) -> Bool
    let onRemove: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            title
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            if domains.isEmpty {
                Text("None", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(domains, id: \.self) { domain in
                HStack(spacing: 6) {
                    Text(verbatim: domain)
                        .themedCode(.small)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(Text(verbatim: domain))
                    Spacer(minLength: 0)
                    AppIconButton("minus.circle", label: Text("Remove \(domain)", bundle: .module)) {
                        onRemove(domain)
                    }
                    .appAction(.quiet, size: .small)
                }
                .accessibilityElement(children: .combine)
            }

            HStack(spacing: 6) {
                TextField(text: $draft, prompt: Text(verbatim: "example.com")) {
                    title
                }
                .textFieldStyle(.roundedBorder)
                .onSubmit(submit)
                Button(action: submit) {
                    Text("Add", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                .help(Text("Type a host name to add", bundle: .module))
            }
        }
    }

    private func submit() {
        let entry = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !entry.isEmpty else { return }
        if onAdd(entry) { draft = "" }
    }
}
