import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The MCP server editor's environment block, one row per variable with the
/// value hidden until that row is revealed.
///
/// Edits the sheet's existing `KEY=VALUE` text through `McpEnvRows`, so what
/// the sheet saves is built exactly as before. Display only: nothing here
/// changes where or how the values are stored.
@MainActor
struct McpEnvRowsEditor: View {
    @Binding var text: String

    var body: some View {
        let rows = McpEnvRows.parse(text)
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(rows.indices, id: \.self) { index in
                    McpEnvRowView(row: rowBinding(index), onRemove: { remove(index) })
                }
            }
            // Rows are identified by position, so a removal would hand one
            // row's revealed state to its neighbour. A new identity per count
            // re-hides every row instead, the safe direction.
            .id(rows.count)

            Button {
                text = McpEnvRows.serialize(McpEnvRows.parse(text) + [McpEnvRows.Row(key: "", value: "")])
            } label: {
                Label { Text("Add Variable", bundle: .module) } icon: { Image(systemName: "plus") }
            }
            .appAction(.secondary, size: .small)
        }
    }

    private func rowBinding(_ index: Int) -> Binding<McpEnvRows.Row> {
        Binding(
            get: {
                let rows = McpEnvRows.parse(text)
                return rows.indices.contains(index) ? rows[index] : McpEnvRows.Row(key: "", value: "")
            },
            set: { newRow in
                var rows = McpEnvRows.parse(text)
                guard rows.indices.contains(index) else { return }
                rows[index] = newRow
                text = McpEnvRows.serialize(rows)
            })
    }

    private func remove(_ index: Int) {
        var rows = McpEnvRows.parse(text)
        guard rows.indices.contains(index) else { return }
        rows.remove(at: index)
        text = McpEnvRows.serialize(rows)
    }
}

/// One variable. While hidden, the value field is a `SecureField` over an
/// empty replacement whose placeholder is the fixed-width mask, not over the
/// value itself: a `SecureField` draws one bullet per character and would
/// show the secret's length. Typing replaces the value; revealing edits it.
@MainActor
private struct McpEnvRowView: View {
    @Binding var row: McpEnvRows.Row
    let onRemove: () -> Void

    @State private var isRevealed = false
    @State private var replacement = ""
    /// Set by the first keystroke into the hidden field. From then on the
    /// field is the value, so clearing it clears the value instead of
    /// falling back to the hidden one.
    @State private var isReplacing = false

    var body: some View {
        HStack(spacing: 6) {
            TextField(text: $row.key, prompt: Text(verbatim: "API_KEY")) {
                Text("Variable name", bundle: .module)
            }
            .labelsHidden()
            .textFieldStyle(.roundedBorder)
            .themedCode(.small)
            .frame(maxWidth: 180)

            Text(verbatim: "=")
                .themedCode(.small)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)

            valueField
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .themedCode(.small)

            AppIconButton(
                isRevealed ? "eye.slash" : "eye",
                label: isRevealed ? Text("Hide value", bundle: .module) : Text("Show value", bundle: .module)
            ) {
                isRevealed.toggle()
                replacement = ""
                isReplacing = false
            }
            .appAction(.quiet, size: .small)

            AppIconButton("minus.circle", label: Text("Remove variable", bundle: .module), action: onRemove)
                .appAction(.quiet, size: .small)
        }
    }

    @ViewBuilder
    private var valueField: some View {
        if isRevealed {
            TextField(text: $row.value) {
                Text("Variable value", bundle: .module)
            }
        } else {
            SecureField(text: $replacement, prompt: Text(verbatim: McpEnvMasking.maskedValue(row.value))) {
                Text("Variable value", bundle: .module)
            }
            .accessibilityHint(row.value.isEmpty || isReplacing
                ? Text(verbatim: "")
                : Text("Typing replaces the hidden value.", bundle: .module))
            .onChange(of: replacement) { _, newValue in
                guard isReplacing || !newValue.isEmpty else { return }
                isReplacing = true
                row.value = newValue
            }
        }
    }
}
