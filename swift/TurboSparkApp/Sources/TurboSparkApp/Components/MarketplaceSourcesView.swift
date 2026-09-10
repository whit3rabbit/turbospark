import SwiftUI

/// Source subscriptions are independent of the target of an individual install.
struct MarketplaceSourcesView: View {
    @ObservedObject var model: AppModel
    let kind: MarketplaceKind
    @State var projectID: UUID?
    let select: (MarketplaceSource) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var location = ""
    @State private var sourceType = "github"
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Marketplace sources", bundle: .module).themedFont(.title3, weight: .semibold)
                Spacer()
                Button { dismiss() } label: { Text("Done", bundle: .module) }
            }
            ExtensionScopePicker(model: model, projectID: $projectID)
            Text("Project sources include user sources. Removing a source keeps installed items.", bundle: .module).themedFont(.small)
            List {
                let sources = model.marketplaceSources(kind: kind, projectID: projectID)
                ForEach(sources.keys.sorted(), id: \.self) { name in
                    HStack {
                        Button(name) { if let source = sources[name] { select(source); dismiss() } }
                        Spacer()
                        Text(origin(name)).themedFont(.small)
                        Button(role: .destructive) { model.removeMarketplace(name: name, kind: kind, projectID: projectID) }
                            label: { Text("Remove source", bundle: .module) }
                    }
                }
                if let project = model.projects.first(where: { $0.id == projectID }) {
                    ForEach((project.marketplaces.hidden[kind.rawValue] ?? []).sorted(), id: \.self) { name in
                        HStack {
                            Text(name).themedFont(.small)
                            Spacer()
                            Button { model.restoreMarketplace(name: name, kind: kind, projectID: project.id) }
                                label: { Text("Restore inherited source", bundle: .module) }
                        }
                    }
                }
            }
            TextField("Name", text: $name)
            Picker(selection: $sourceType) {
                Text("GitHub", bundle: .module).tag("github")
                Text("Git URL", bundle: .module).tag("git")
                Text("JSON URL", bundle: .module).tag("url")
                Text("Directory", bundle: .module).tag("directory")
            } label: { Text("Source type", bundle: .module) }
            TextField("Location", text: $location)
            Button { addSource() } label: { Text("Add source", bundle: .module) }
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || location.trimmingCharacters(in: .whitespaces).isEmpty)
            if let error { Text(error).foregroundStyle(.red).themedFont(.small) }
        }
        .padding(20)
        .frame(minWidth: 540, minHeight: 480)
        .textFieldStyle(.roundedBorder)
    }

    private func origin(_ name: String) -> String {
        guard let project = model.projects.first(where: { $0.id == projectID }) else { return "User" }
        return project.marketplaces.sources[kind.rawValue]?[name] == nil ? "Inherited from user" : project.name
    }

    private func addSource() {
        let value = location.trimmingCharacters(in: .whitespacesAndNewlines)
        let source: MarketplaceSource
        switch sourceType {
        case "github": source = .github(repo: value, ref: nil, path: nil, sparsePaths: nil)
        case "git": source = .git(url: value, ref: nil, path: nil, sparsePaths: nil)
        case "url": source = .url(url: value, headers: nil)
        default: source = .directory(path: value)
        }
        do {
            try model.saveMarketplace(name: name.trimmingCharacters(in: .whitespaces), source: source, kind: kind, projectID: projectID)
            name = ""; location = ""; error = nil
        } catch { self.error = error.localizedDescription }
    }
}
