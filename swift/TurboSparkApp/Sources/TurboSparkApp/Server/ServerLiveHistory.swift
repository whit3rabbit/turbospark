import Darwin
import Foundation
import TurboSpark

public struct ServerLivePoint: Identifiable, Equatable {
    public var id: Date { date }
    public let date: Date
    public let receivedPerSecond: Double?
    public let sentPerSecond: Double?
    public let memoryBytes: UInt64?
}

/// One minute at the existing 2 Hz server poll, with no second timer.
public struct ServerLiveHistory: Equatable {
    public private(set) var points: [ServerLivePoint] = []
    private var previous: (Date, UInt64, UInt64)?

    public static func == (lhs: Self, rhs: Self) -> Bool { lhs.points == rhs.points }

    mutating func sample(date: Date, received: UInt64?, sent: UInt64?, memory: UInt64?) {
        var incoming: Double?
        var outgoing: Double?
        if let received, let sent {
            if let (last, oldIn, oldOut) = previous, date > last, received >= oldIn, sent >= oldOut {
                let seconds = date.timeIntervalSince(last)
                incoming = Double(received - oldIn) / seconds
                outgoing = Double(sent - oldOut) / seconds
            }
            previous = (date, received, sent)
        } else { previous = nil }
        points.append(ServerLivePoint(date: date, receivedPerSecond: incoming,
            sentPerSecond: outgoing, memoryBytes: memory))
        if points.count > 120 { points.removeFirst(points.count - 120) }
    }
}

extension AppModel {
    func sampleServerLive() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        serverLive.sample(date: Date(), received: serverInfo?.traffic?.receivedBytes,
            sent: serverInfo?.traffic?.sentBytes,
            memory: result == KERN_SUCCESS ? info.phys_footprint : nil)
    }
}
