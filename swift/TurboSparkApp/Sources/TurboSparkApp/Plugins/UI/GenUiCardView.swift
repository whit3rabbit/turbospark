import SwiftUI

// The native renderer for a validated `GenUiSpec` (ZP-06). Every string the
// model supplied is drawn with `Text(verbatim:)`: no markdown, no links, no
// HTML. Controls are system controls, so keyboard and VoiceOver come free.

@MainActor
struct GenUiCardView: View {
    @Environment(\.appTheme) private var theme
    let spec: GenUiSpec
    /// Receives the labeled draft text. The caller fills the composer; this
    /// view never sends anything.
    let onFollowUp: (String) -> Void

    @State private var values: GenUiValues

    init(spec: GenUiSpec, onFollowUp: @escaping (String) -> Void) {
        self.spec = spec
        self.onFollowUp = onFollowUp
        _values = State(initialValue: GenUiValues(spec))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(verbatim: spec.title)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            ForEach(Array(spec.components.enumerated()), id: \.offset) { _, component in
                row(for: component)
            }
            Text("Interactive card from the assistant. It runs no code and reaches no network.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCardSurface(padding: 12)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private func row(for component: GenUiComponent) -> some View {
        switch component {
        case .text(let text, let style):
            Text(verbatim: text)
                .themedFont(style == .heading ? .base : (style == .caption ? .tiny : .small),
                            weight: style == .heading ? .semibold : .regular)
                .foregroundStyle(style == .caption ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appText))
                .fixedSize(horizontal: false, vertical: true)
        case .slider(let c):
            numberRow(c) {
                Slider(value: numberBinding(c), in: c.min...c.max, step: c.step) {
                    Text(verbatim: c.label)
                }
                .labelsHidden()
                .accessibilityValue(Text(verbatim: valueText(c)))
            }
        case .stepper(let c):
            numberRow(c) {
                Stepper(value: numberBinding(c), in: c.min...c.max, step: c.step) {
                    Text(verbatim: c.label)
                }
                .labelsHidden()
                .accessibilityLabel(Text(verbatim: c.label))
                .accessibilityValue(Text(verbatim: valueText(c)))
            }
        case .toggle(let id, let label, _):
            Toggle(isOn: Binding(get: { values.toggles[id] ?? false }, set: { values.toggles[id] = $0 })) {
                Text(verbatim: label)
            }
        case .picker(let id, let label, let options, _):
            Picker(selection: Binding(get: { values.picks[id] ?? 0 }, set: { values.picks[id] = $0 })) {
                ForEach(Array(options.enumerated()), id: \.offset) { index, option in
                    Text(verbatim: option).tag(index)
                }
            } label: {
                Text(verbatim: label)
            }
            .pickerStyle(.menu)
        case .readout(let label, let bind, let unit):
            HStack {
                Text(verbatim: label).themedFont(.small).foregroundStyle(.appSecondary)
                Spacer()
                Text(verbatim: readoutText(bind: bind, unit: unit)).themedFont(.small, weight: .semibold)
            }
            .accessibilityElement(children: .combine)
        case .table(let header, let rows):
            tableView(header: header, rows: rows)
        case .bars(let title, let items):
            barsView(title: title, items: items)
        case .followUp(let label, let template):
            Button {
                onFollowUp(values.expand(template, in: spec))
            } label: {
                Text(verbatim: label)
            }
            .appAction(.secondary, size: .small)
            .help(Text("Adds this to the message box. Nothing is sent until you send it.", bundle: .module))
        }
    }

    private func numberBinding(_ c: GenUiNumberControl) -> Binding<Double> {
        Binding(get: { values.numbers[c.id] ?? c.value }, set: { values.numbers[c.id] = $0 })
    }

    private func valueText(_ c: GenUiNumberControl) -> String {
        GenUiParser.formatted(values.numbers[c.id] ?? c.value, step: c.step) + (c.unit.isEmpty ? "" : " " + c.unit)
    }

    private func numberRow<Control: View>(_ c: GenUiNumberControl, @ViewBuilder control: () -> Control) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(verbatim: c.label).themedFont(.small)
                Spacer()
                Text(verbatim: valueText(c)).themedFont(.small, weight: .semibold).foregroundStyle(.appSecondary)
            }
            control()
        }
    }

    private func readoutText(bind: String, unit: String) -> String {
        for component in spec.components {
            switch component {
            case .slider(let c), .stepper(let c):
                if c.id == bind { return valueText(c) }
            case .toggle(let id, _, _):
                if id == bind { return (values.toggles[id] ?? false) ? "on" : "off" }
            case .picker(let id, _, let options, _):
                if id == bind {
                    let index = values.picks[id] ?? 0
                    return options.indices.contains(index) ? options[index] : ""
                }
            default: break
            }
        }
        return ""
    }

    private func tableView(header: [String], rows: [[String]]) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
            GridRow {
                ForEach(Array(header.enumerated()), id: \.offset) { _, title in
                    Text(verbatim: title).themedFont(.tiny, weight: .semibold).foregroundStyle(.appSecondary)
                }
            }
            Divider()
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(verbatim: cell).themedFont(.small)
                    }
                }
            }
        }
    }

    private func barsView(title: String, items: [GenUiBar]) -> some View {
        let peak = max(items.map(\.value).max() ?? 1, 0.000_001)
        return VStack(alignment: .leading, spacing: 4) {
            if !title.isEmpty {
                Text(verbatim: title).themedFont(.small, weight: .semibold)
            }
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                HStack(spacing: 8) {
                    Text(verbatim: item.label).themedFont(.tiny)
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120, alignment: .leading)
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(.appSurface)
                            Capsule().fill(.appAccent)
                                .frame(width: max(2, proxy.size.width * item.value / peak))
                        }
                    }
                    .frame(height: 8)
                    Text(verbatim: GenUiParser.formatted(item.value, step: 0.01)).themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}
