import Foundation

/// The worker wire protocol for codemode, mirroring pi's protocol.ts
/// message shapes. Everything crossing the process boundary is newline-
/// delimited JSON, and tool arguments and results cross as JSON STRINGS:
/// neither side ever builds the other side's structured values.
///
/// One process serves exactly one script execution and exits after writing
/// its terminal message, so there is no init/reset lifecycle to manage.
enum CodemodeWire {
    /// Bound on one wire line in either direction. Sized so the largest
    /// nested result (`CodemodeLimits.maximumCallResultBytes`) still fits
    /// after the worst-case double JSON escaping it takes on the way in
    /// (about 7x for control characters), and so a worker that never ends a
    /// line cannot grow the host's buffer without limit.
    static let maximumLineBytes = 16 * 1_024 * 1_024

    /// Host -> worker: run one script. Sent as the first and only request.
    struct RunRequest: Codable {
        var type = "codemode"
        var id: Int
        var code: String
        var tools: [CodemodeToolEntry]
        /// JSON object text of the store snapshot for `load()`.
        var storeSnapshot: String
        /// Runtime bounds the prelude enforces inside the VM.
        var limits: CodemodeLimits
    }

    /// Host -> worker: settle one pending `tools` call.
    struct CallResult: Codable {
        var type = "result"
        var id: Int
        var ok: Bool
        /// JSON result text when `ok`, otherwise the error message.
        var payload: String?
    }

    /// Worker -> host: one nested call. `args` is a JSON string or absent.
    struct Call: Codable {
        var type = "call"
        var id: Int
        var name: String
        var args: String?
    }

    /// Worker -> host: one streamed output item.
    struct Output: Codable {
        var type = "output"
        var kind: String
        var text: String
        /// Console level for `kind == "console"`.
        var level: String?
    }

    /// Worker -> host: terminal message of a completed execution.
    struct Done: Codable {
        var type = "done"
        var ok: Bool
        var value: String?
        /// JSON-encoded `{ name?, message, stack? }` when not ok.
        var error: String?
        /// JSON array of `[key, json]` stores and `[key]` deletes.
        var writes: String?
    }

    /// Worker -> host: the VM or worker failed outside the script's control.
    struct Crash: Codable {
        var type = "crash"
        var message: String
    }

    /// Decodes one worker->host line. Unknown types return nil so a newer
    /// worker talking to an older supervisor degrades to "no message"
    /// instead of a decode abort.
    static func decodeWorkerMessage(_ line: String) -> WorkerMessage? {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = object["type"] as? String
        else { return nil }
        let decoder = JSONDecoder()
        switch type {
        case "call":
            return (try? decoder.decode(Call.self, from: data)).map(WorkerMessage.call)
        case "output":
            return (try? decoder.decode(Output.self, from: data)).map(WorkerMessage.output)
        case "done":
            return (try? decoder.decode(Done.self, from: data)).map(WorkerMessage.done)
        case "crash":
            return (try? decoder.decode(Crash.self, from: data)).map(WorkerMessage.crash)
        default:
            return nil
        }
    }

    enum WorkerMessage {
        case call(Call)
        case output(Output)
        case done(Done)
        case crash(Crash)
    }

    /// Encodes one message to a newline-framed UTF-8 line.
    static func encodeLine<T: Encodable>(_ message: T) -> Data? {
        guard var data = try? JSONEncoder().encode(message) else { return nil }
        data.append(0x0A)
        return data
    }
}
