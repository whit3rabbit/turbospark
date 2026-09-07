import SwiftUI

/// Configuration sheet for plugin and hook `userConfig` options.
public struct HookOptionsConfigSheet: View {
    public let group: AppHookSourceGroup
    @ObservedObject var hookStore: AppHookStore = .shared

    @State private var values: [String: String] = [:]
    @Environment(\.dismiss) private var dismiss

    public init(group: AppHookSourceGroup) {
        self.group = group
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(group.title) Options", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                    Text("Configure environment variables and options for \(group.title).", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(.title3)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            if group.optionSpecs.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "slider.horizontal.3")
                        .themedFont(points: 32)
                        .foregroundStyle(.secondary)
                    Text("No configurable options declared for this source.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(40)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(group.optionSpecs) { spec in
                            optionRow(spec: spec)
                        }
                    }
                    .padding(20)
                }
            }

            Divider()

            // Footer
            HStack {
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(minWidth: 460, minHeight: 340)
        .onAppear {
            loadValues()
        }
    }

    @ViewBuilder
    private func optionRow(spec: AppHookOptionSpec) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(spec.title)
                    .themedFont(.small, weight: .semibold)
                if spec.isRequired {
                    Text("*", bundle: .module)
                        .foregroundStyle(.red)
                }
                Spacer()
                Text("$\(spec.key.uppercased())", bundle: .module)
                    .themedCode(.tiny)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }

            Text(spec.description)
                .themedFont(.small)
                .foregroundStyle(.secondary)

            switch spec.type {
            case .boolean:
                Toggle(
                    "",
                    isOn: Binding(
                        get: { (values[spec.key] ?? spec.defaultValue ?? "false") == "true" },
                        set: { val in
                            let strVal = val ? "true" : "false"
                            values[spec.key] = strVal
                            hookStore.updateOptionValue(sourceID: group.id, key: spec.key, value: strVal)
                        }
                    )
                )
                .labelsHidden()

            case .number, .string:
                if spec.isSensitive {
                    SecureField(
                        spec.defaultValue ?? "Enter value...",
                        text: Binding(
                            get: { values[spec.key] ?? "" },
                            set: { val in
                                values[spec.key] = val
                                hookStore.updateOptionValue(sourceID: group.id, key: spec.key, value: val)
                            }
                        )
                    )
                    .textFieldStyle(.roundedBorder)
                } else {
                    TextField(
                        spec.defaultValue ?? "Enter value...",
                        text: Binding(
                            get: { values[spec.key] ?? "" },
                            set: { val in
                                values[spec.key] = val
                                hookStore.updateOptionValue(sourceID: group.id, key: spec.key, value: val)
                            }
                        )
                    )
                    .textFieldStyle(.roundedBorder)
                }

            case .file, .directory:
                HStack(spacing: 8) {
                    TextField(
                        "Path...",
                        text: Binding(
                            get: { values[spec.key] ?? "" },
                            set: { val in
                                values[spec.key] = val
                                hookStore.updateOptionValue(sourceID: group.id, key: spec.key, value: val)
                            }
                        )
                    )
                    .textFieldStyle(.roundedBorder)

                    Button("Browse...") {
                        choosePath(isDirectory: spec.type == .directory) { chosen in
                            if let chosen {
                                values[spec.key] = chosen
                                hookStore.updateOptionValue(sourceID: group.id, key: spec.key, value: chosen)
                            }
                        }
                    }
                    .controlSize(.small)
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
        )
    }

    private func loadValues() {
        var map: [String: String] = [:]
        for spec in group.optionSpecs {
            map[spec.key] = hookStore.getOptionValue(sourceID: group.id, key: spec.key, defaultVal: spec.defaultValue)
        }
        self.values = map
    }

    private func choosePath(isDirectory: Bool, completion: @escaping (String?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = !isDirectory
        panel.canChooseDirectories = isDirectory
        panel.allowsMultipleSelection = false
        panel.begin { response in
            if response == .OK, let url = panel.url {
                completion(url.path)
            } else {
                completion(nil)
            }
        }
    }
}
