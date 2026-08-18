import SwiftUI
import TurboSpark

/// Installing a catalog row, with the one warning that matters.
struct InstallSheet: View {
    let catalog: [CatalogEntry]
    let onFinished: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selected: String?
    @State private var stage = ""
    @State private var done: UInt64 = 0
    @State private var total: UInt64 = 0
    @State private var running = false
    @State private var failure: String?
    @State private var task: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Install a model").font(.headline)

            // THE WARNING IS SHOWN BEFORE THE BUTTON, not after a failure.
            // The walk streams gigabytes without writing the checkpoint to
            // disk whole, and a failure restarts it from the beginning, so a
            // user who does not know that will kill it at 90% and try again.
            Label(
                "Downloads cannot resume. If one fails or is interrupted, it starts over.",
                systemImage: "exclamationmark.triangle"
            )
            .font(.caption)
            .foregroundStyle(.orange)

            List(catalog, selection: $selected) { row in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(row.name)
                        Text("\(row.alias)  ·  \(row.family)  ·  \(row.status)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if row.installed {
                        Text("installed").font(.caption).foregroundStyle(.green)
                    } else {
                        Text(humanBytes(row.downloadBytes)).font(.caption).monospacedDigit()
                    }
                }
                .tag(row.alias)
            }
            .frame(height: 240)
            .disabled(running)

            if running {
                VStack(alignment: .leading, spacing: 6) {
                    if total > 0 {
                        ProgressView(value: Double(done), total: Double(total))
                        Text("\(humanBytes(done)) of \(humanBytes(total))")
                            .font(.caption).monospacedDigit()
                    } else {
                        ProgressView()
                    }
                    Text(stage).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
            }

            if let failure {
                Text(failure).font(.caption).foregroundStyle(.red)
            }

            HStack {
                Spacer()
                Button("Close") {
                    task?.cancel()
                    dismiss()
                }
                Button("Install") { start() }
                    .buttonStyle(.borderedProminent)
                    .disabled(running || selected == nil || isInstalled(selected))
            }
        }
        .padding(16)
        .frame(width: 560)
    }

    private func isInstalled(_ alias: String?) -> Bool {
        guard let alias else { return false }
        return catalog.first { $0.alias == alias }?.installed ?? false
    }

    private func start() {
        guard let alias = selected else { return }
        running = true
        failure = nil
        done = 0
        total = 0

        task = Task {
            do {
                total = try TurboSparkCatalog.cost(of: alias).downloadBytes
                for try await event in TurboSparkCatalog.install(alias) {
                    switch event {
                    case .stage(let line):
                        stage = line
                    case .bytes(let d, let t):
                        // MAX, not last. Byte events arrive from several
                        // download threads at once and are not ordered, so
                        // taking the latest makes the bar jump backwards.
                        done = max(done, d)
                        if t > 0 { total = max(total, t) }
                    case .finished:
                        onFinished()
                        dismiss()
                    }
                }
            } catch {
                failure = "\(error)"
            }
            running = false
        }
    }
}
