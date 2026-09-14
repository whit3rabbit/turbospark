import AppKit
import Charts
import SwiftUI

struct ServerLiveChartsView: View {
    @ObservedObject var model: AppModel
    var compact = false
    private var points: [ServerLivePoint] { model.serverLive.points }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) { bandwidth; memory }.frame(minWidth: compact ? 320 : 600)
            VStack(spacing: 14) { bandwidth; memory }
        }
    }

    private var bandwidth: some View {
        VStack(alignment: .leading, spacing: 8) {
            Group {
                if compact { Text("Traffic", bundle: .module) }
                else { Text("HTTP body bandwidth", bundle: .module) }
            }.themedFont(.small, weight: .semibold)
            HStack {
                Text(verbatim: "IN \(rate(points.last?.receivedPerSecond))")
                Spacer()
                Text(verbatim: "OUT \(rate(points.last?.sentPerSecond))")
            }.themedCode(.small).lineLimit(1).minimumScaleFactor(0.8)
            Chart(points) { point in
                if let value = point.receivedPerSecond {
                    LineMark(x: .value("Time", point.date), y: .value("B/s", value))
                        .foregroundStyle(by: .value("Direction", "IN"))
                }
                if let value = point.sentPerSecond {
                    LineMark(x: .value("Time", point.date), y: .value("B/s", value))
                        .foregroundStyle(by: .value("Direction", "OUT"))
                }
            }
            .chartXAxis(.hidden)
            .chartLegend(.hidden)
            .frame(height: compact ? 50 : 130)
        }
        .padding(14).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    private var memory: some View {
        VStack(alignment: .leading, spacing: 8) {
            Group {
                if compact { Text("Memory", bundle: .module) }
                else { Text("App + server memory", bundle: .module) }
            }.themedFont(.small, weight: .semibold)
            Text(verbatim: "\(bytes(points.last?.memoryBytes)) / \(bytes(model.telemetry?.physicalMemoryBytes))")
                .themedCode(.small)
            Chart(points) { point in
                if let value = point.memoryBytes {
                    AreaMark(x: .value("Time", point.date), y: .value("GiB", Double(value) / 1_073_741_824))
                        .foregroundStyle(.appAccent.opacity(0.15))
                    LineMark(x: .value("Time", point.date), y: .value("GiB", Double(value) / 1_073_741_824))
                        .foregroundStyle(.appAccent)
                }
            }
            .chartXAxis(.hidden)
            .chartYScale(domain: 0...max(1, Double(model.telemetry?.physicalMemoryBytes ?? 1_073_741_824) / 1_073_741_824))
            .frame(height: compact ? 50 : 130)
        }
        .padding(14).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    private func rate(_ value: Double?) -> String {
        guard let value else { return "--" }
        if value < 1 { return "0 B/s" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .decimal) + "/s"
    }
    private func bytes(_ value: UInt64?) -> String {
        value.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory) } ?? "--"
    }
}

struct ServerTextPreviewView: View {
    @ObservedObject var model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Optional raw HTTP text. Kept in memory, truncated, cleared on restart.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString((model.serverInfo?.traffic?.previews ?? []).joined(separator: "\n"), forType: .string)
                } label: { Text("Copy", bundle: .module) }
            }
            ScrollView([.horizontal, .vertical]) {
                Text((model.serverInfo?.traffic?.previews ?? []).joined(separator: "\n"))
                    .themedCode(.small).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }.padding(14)
    }
}
