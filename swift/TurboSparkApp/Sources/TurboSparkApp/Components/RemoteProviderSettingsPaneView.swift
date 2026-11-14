import SwiftUI
import Security

/// Settings > Network > Remote providers (ZP-22): named providers, each with
/// an API family, base URL, API key and an editable model list.
///
/// The key rules (swift/AGENTS.md): nothing is stored by default. "Use for
/// this session" holds the key in memory. "Save to Keychain" is the explicit,
/// per-credential opt-in that writes the item; "Forget key" deletes it. The
/// Keychain is never read here except by the Test connection button, an
/// explicit click. The key never goes into settings.json or the provider file.
@MainActor
struct RemoteProviderSettingsSectionView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: RemoteProviderStore
    @State private var newName = ""
    @State private var newFamily = RemoteAPIFamily.openAICompatible
    @State private var newURL = ""
    @State private var addError: String?

    var body: some View {
        Section {
            Text("Remote providers send your prompts and tool results to a third party over the network. Local models stay the default. Nothing is sent until you add a provider and key, then choose its model in a chat and confirm. API keys only, no account sign-in.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            ForEach(store.providers) { provider in
                RemoteProviderRowView(model: model, store: store, providerID: provider.id)
            }

            addForm
        } header: {
            Text("Remote providers", bundle: .module)
        }
    }

    private var addForm: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField(text: $newName) { Text("Name", bundle: .module) }
                .textFieldStyle(.roundedBorder)
            Picker(selection: $newFamily) {
                ForEach(RemoteAPIFamily.allCases, id: \.self) { family in
                    family.labelText.tag(family)
                }
            } label: { Text("API family", bundle: .module) }
            TextField(text: $newURL) { Text("Base URL", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
            Button(action: add) { Text("Add provider", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty
                    || newURL.trimmingCharacters(in: .whitespaces).isEmpty)
                .settingsControl("Add provider", pane: .network, timing: .immediate)
            if let addError {
                Label { Text(verbatim: addError) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small)
                    .foregroundStyle(.appWarning)
            }
        }
    }

    private func add() {
        do {
            try store.add(name: newName, family: newFamily, baseURL: newURL)
            newName = ""
            newURL = ""
            addError = nil
        } catch let error as RemoteProviderError {
            addError = model.remoteErrorText(error)
        } catch {
            addError = error.localizedDescription
        }
    }
}

extension RemoteAPIFamily {
    var labelText: Text {
        switch self {
        case .anthropic: return Text("Anthropic", bundle: .module)
        case .openAICompatible: return Text("OpenAI-compatible", bundle: .module)
        }
    }
}

/// One provider: header, base URL, key controls, models, test, remove.
@MainActor
struct RemoteProviderRowView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: RemoteProviderStore
    let providerID: UUID
    @State private var keyDraft = ""
    @State private var keyFailure: String?
    @State private var urlDraft: String?
    @State private var urlError: String?
    @State private var testState: TestState = .idle
    @State private var newModelID = ""

    private enum TestState: Equatable { case idle, running, ok, failed(String) }

    var body: some View {
        if let provider = store.provider(id: providerID) {
            VStack(alignment: .leading, spacing: 8) {
                header(provider)
                urlField(provider)
                keyControls(provider)
                modelsEditor(provider)
                footer(provider)
            }
            .padding(.vertical, 6)
            Divider()
        }
    }

    private func header(_ provider: RemoteProvider) -> some View {
        HStack {
            Text(verbatim: provider.name).themedFont(.base, weight: .semibold)
            provider.family.labelText
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Spacer()
            Toggle(isOn: Binding(
                get: { provider.enabled },
                set: { try? store.update(providerID, enabled: $0) })
            ) { Text("Enabled", bundle: .module) }
            .toggleStyle(.switch)
            .labelsHidden()
            .accessibilityLabel(Text("Enabled", bundle: .module))
        }
    }

    private func urlField(_ provider: RemoteProvider) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            TextField(text: Binding(
                get: { urlDraft ?? provider.baseURL },
                set: { urlDraft = $0 })
            ) { Text("Base URL", bundle: .module) }
            .textFieldStyle(.roundedBorder)
            .onSubmit {
                guard let draft = urlDraft else { return }
                do {
                    try store.update(providerID, baseURL: draft)
                    urlDraft = nil
                    urlError = nil
                } catch let error as RemoteProviderError {
                    urlError = model.remoteErrorText(error)
                } catch {}
            }
            if let urlError {
                Text(verbatim: urlError).themedFont(.tiny).foregroundStyle(.appWarning)
            }
        }
    }

    // MARK: Key

    private var keychainEnabled: Bool { store.keys.isKeychainEnabled(for: providerID) }
    private var keyAvailable: Bool { store.keys.isKeyAvailable(for: providerID) }

    private func keyControls(_ provider: RemoteProvider) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                SecureField(text: $keyDraft) { Text("API key", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
                Button(action: useForSession) { Text("Use for this session", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button(action: saveToKeychain) { Text("Save to Keychain", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .settingsControl("Save to Keychain", pane: .network, timing: .action)
                Button(action: forget) { Text("Forget key", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(!keyAvailable)
            }
            .settingsControl("API key", pane: .network, timing: .nextTurn)
            Group {
                if keychainEnabled {
                    Text("Key saved in the Keychain.", bundle: .module)
                } else if keyAvailable {
                    Text("Key held in memory for this session only.", bundle: .module)
                } else {
                    Text("No key set.", bundle: .module)
                }
            }
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            if let keyFailure {
                Label { Text(verbatim: keyFailure) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small)
                    .foregroundStyle(.appWarning)
            }
        }
    }

    private func useForSession() {
        store.keys.setSessionKey(keyDraft, for: providerID)
        keyDraft = ""
        keyFailure = nil
        store.objectWillChange.send()
    }

    /// The explicit per-credential opt-in: only this click writes the item.
    private func saveToKeychain() {
        let status = store.keys.saveToKeychain(keyDraft, for: providerID)
        if status == errSecSuccess {
            keyDraft = ""
            keyFailure = nil
        } else {
            keyFailure = AppStrings.localized("Could not save the key to the Keychain.")
                + " " + TypeSafeKeychain.describe(status: status)
        }
        store.objectWillChange.send()
    }

    private func forget() {
        store.keys.forget(for: providerID)
        keyDraft = ""
        keyFailure = nil
        testState = .idle
        store.objectWillChange.send()
    }

    // MARK: Models

    private func modelsEditor(_ provider: RemoteProvider) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Models", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
            ForEach(provider.models) { entry in
                RemoteModelRowView(
                    entry: entry,
                    onChange: { updated in replace(entry.id, with: updated, in: provider) },
                    onRemove: { remove(entry.id, in: provider) })
            }
            HStack(spacing: 6) {
                TextField(text: $newModelID) { Text("Model ID", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .onSubmit { addModel(to: provider) }
                Button { addModel(to: provider) } label: { Text("Add model", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(newModelID.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
    }

    private func addModel(to provider: RemoteProvider) {
        let id = newModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty, !provider.models.contains(where: { $0.id == id }) else { return }
        var models = provider.models
        models.append(RemoteProviderModel(id: id, displayName: id, contextWindow: 128_000))
        store.setModels(models, for: providerID)
        newModelID = ""
    }

    private func replace(_ id: String, with updated: RemoteProviderModel, in provider: RemoteProvider) {
        var models = provider.models
        guard let i = models.firstIndex(where: { $0.id == id }) else { return }
        models[i] = updated
        store.setModels(models, for: providerID)
    }

    private func remove(_ id: String, in provider: RemoteProvider) {
        store.setModels(provider.models.filter { $0.id != id }, for: providerID)
    }

    // MARK: Test and remove

    private func footer(_ provider: RemoteProvider) -> some View {
        HStack(spacing: 8) {
            Button(action: test) { Text("Test connection", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(testState == .running || !keyAvailable)
                .settingsControl("Test connection", pane: .network, timing: .action)
            switch testState {
            case .idle: EmptyView()
            case .running: Text("Testing...", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
            case .ok:
                Label { Text("Connection works.", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                    .themedFont(.small).foregroundStyle(.appSuccess)
            case .failed(let text):
                Label { Text(verbatim: text) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small).foregroundStyle(.appWarning)
            }
            Spacer()
            Button(role: .destructive) {
                for key in model.remoteSelections.keys where model.remoteSelections[key]?.providerID == providerID {
                    model.useLocalModel(chatID: key)
                }
                store.remove(providerID)
            } label: { Text("Remove provider", bundle: .module) }
            .appAction(.destructive, size: .small)
        }
    }

    /// An explicit click: the one place settings reads the key (and so the
    /// Keychain, for an opted-in credential).
    private func test() {
        testState = .running
        let service = RemoteProviderService(store: store, transport: model.remoteTransport)
        Task {
            do {
                try await service.testConnection(providerID: providerID)
                testState = .ok
            } catch let error as RemoteProviderError {
                testState = .failed(model.remoteErrorText(error))
            } catch {
                testState = .failed(RemoteProviderRedactor.redact(error.localizedDescription, maxCharacters: 200))
            }
        }
    }
}

/// One model entry: id, display name, context window and the flags.
struct RemoteModelRowView: View {
    let entry: RemoteProviderModel
    let onChange: (RemoteProviderModel) -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(verbatim: entry.id).themedFont(.small).lineLimit(1)
                Spacer()
                Button(action: onRemove) { Image(systemName: "minus.circle") }
                    .buttonStyle(.borderless)
                    .help(Text("Remove model", bundle: .module))
                    .accessibilityLabel(Text("Remove model", bundle: .module))
            }
            HStack(spacing: 6) {
                TextField(text: Binding(
                    get: { entry.displayName },
                    set: { var e = entry; e.displayName = $0; onChange(e) })
                ) { Text("Display name", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                TextField(value: Binding(
                    get: { entry.contextWindow },
                    set: { var e = entry; e.contextWindow = max(1_024, $0); onChange(e) }),
                    format: .number
                ) { Text("Context window (tokens)", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 120)
                .help(Text("Context window (tokens)", bundle: .module))
            }
            HStack(spacing: 12) {
                Toggle(isOn: Binding(
                    get: { entry.supportsVision },
                    set: { var e = entry; e.supportsVision = $0; onChange(e) })
                ) { Text("Vision", bundle: .module) }
                Toggle(isOn: Binding(
                    get: { entry.supportsReasoning },
                    set: { var e = entry; e.supportsReasoning = $0; onChange(e) })
                ) { Text("Reasoning", bundle: .module) }
                Toggle(isOn: Binding(
                    get: { entry.enabled },
                    set: { var e = entry; e.enabled = $0; onChange(e) })
                ) { Text("Enabled", bundle: .module) }
            }
            .toggleStyle(.checkbox)
            .themedFont(.small)
        }
    }
}
