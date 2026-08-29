import SwiftUI

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var shouldAnimate: Bool {
        animated && !reduceMotion
    }

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .contentTransition(shouldAnimate ? .numericText() : .identity)
                .animation(shouldAnimate ? .snappy(duration: 0.25) : nil, value: value)
            Text(label)
                .font(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
        // The big number and the small caption are read together as one
        // announcement ("23.4 tok/s") rather than as two unrelated strings.
        .accessibilityElement(children: .combine)
        .accessibilityLabel(label)
        .accessibilityValue(value)
    }
}
