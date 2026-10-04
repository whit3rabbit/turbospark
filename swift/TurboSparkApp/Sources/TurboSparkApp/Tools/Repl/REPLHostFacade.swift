import JavaScriptCore

typealias REPLConsoleSink = @Sendable (REPLTextOutputEvent) -> Void

/// Installs the worker's host-visible surface onto a fresh context: the
/// console methods and the rooted file facade. Every member is defined
/// non-writable and non-configurable, and both the host objects and their
/// method functions are frozen, so tampering during one call cannot weaken
/// the surface for later calls. The app never receives JavaScriptCore
/// values or objects through this seam.
final class REPLHostFacade: @unchecked Sendable {
    private let output: REPLConsoleSink

    /// File responses are delivered on this serial queue rather than the
    /// evaluation thread. JavaScriptCore serializes context access
    /// internally, and delivering off the evaluation thread is what lets
    /// the worker keep servicing requests while a top-level await is
    /// pending on a file promise.
    private let fileResponseQueue = DispatchQueue(
        label: "com.turbospark.repl.file-response")

    init(output: @escaping REPLConsoleSink) {
        self.output = output
    }

    func installConsole(into context: JSContext) {
        let output = self.output
        let consoleSink: @convention(block) (String, String) -> Void = { level, text in
            guard let level = REPLTextOutputEvent.Level(rawValue: level) else { return }
            output(REPLTextOutputEvent(level: level, text: text))
        }
        context.setObject(consoleSink, forKeyedSubscript: "__turbosparkConsoleSink" as NSString)
        context.evaluateScript(Self.consoleInstallationScript)
    }

    /// Installs the frozen `repl` object with its `fs` capability onto a
    /// fresh worker context. The facade has no filesystem API of its own:
    /// `repl.fs.readFile(path)`, `repl.fs.writeFile(path, contents)`, and
    /// `repl.fs.list(path)` each send one bounded request over the channel
    /// and settle the returned promise with the typed result, so the worker
    /// never learns the granted roots and never touches FileManager. Must
    /// be called on the worker queue, once per fresh context, alongside
    /// `installConsole`.
    func installFileSystem(into context: JSContext, channel: any REPLFileRequestChannel) {
        let responseQueue = fileResponseQueue
        // `contents` arrives as JSValue rather than String? because the
        // automatic block bridging renders a JS null as the four-character
        // string "null"; the explicit isNull/isUndefined check below keeps
        // a payload-less read request payload-less.
        let requestBridge: @convention(block) (String, String, JSValue, JSValue) -> Void =
            { operationName, path, contents, callback in
                guard let operation = REPLFileAccessRequest.Operation.named(operationName) else {
                    responseQueue.async {
                        callback.call(withArguments: [
                            "repl.fs." + operationName + " is not a known file operation.",
                            NSNull()
                        ])
                    }
                    return
                }
                let payloadText: String?
                if contents.isNull || contents.isUndefined {
                    payloadText = nil
                } else {
                    payloadText = contents.toString()
                }
                let request = REPLFileAccessRequest(
                    id: UUID(),
                    operation: operation,
                    path: path,
                    payload: payloadText.map { Data($0.utf8) })
                Task {
                    let result = await channel.send(request)
                    responseQueue.async {
                        Self.deliver(result, to: callback)
                    }
                }
            }
        context.setObject(requestBridge, forKeyedSubscript: "__turbosparkFileRequest" as NSString)
        context.evaluateScript(Self.fileFacadeInstallationScript)
    }

    /// Delivers one broker result to the facade's resolve/reject callback.
    /// The script side sees only text, arrays, null, or an ordinary error;
    /// roots, identifiers, and transport detail never cross.
    private static func deliver(_ result: REPLFileAccessResult, to callback: JSValue) {
        switch result {
        case let .data(data):
            if let text = String(data: data, encoding: .utf8) {
                callback.call(withArguments: [NSNull(), text])
            } else {
                callback.call(withArguments: [
                    "REPL file read failed: the file is not valid UTF-8 text.",
                    NSNull()
                ])
            }
        case let .entries(entries):
            callback.call(withArguments: [NSNull(), entries])
        case .written:
            callback.call(withArguments: [NSNull(), NSNull()])
        case let .denied(message):
            callback.call(withArguments: [message, NSNull()])
        }
    }

    private static let fileFacadeInstallationScript = """
    (() => {
      const request = globalThis.__turbosparkFileRequest;
      const fs = {};
      for (const name of ["readFile", "writeFile", "list"]) {
        const method = (path, contents) => new Promise((resolve, reject) => {
          if (typeof path !== "string") {
            reject(new Error("repl.fs." + name + " requires a string path"));
            return;
          }
          request(name, path, contents, (error, value) => {
            if (error !== null) reject(new Error(error));
            else resolve(value);
          });
        });
        Object.freeze(method);
        Object.defineProperty(fs, name, {
          value: method,
          enumerable: true,
          writable: false,
          configurable: false
        });
      }
      Object.freeze(fs);
      const repl = {};
      Object.defineProperty(repl, "fs", {
        value: fs,
        enumerable: true,
        writable: false,
        configurable: false
      });
      Object.freeze(repl);
      Object.defineProperty(globalThis, "repl", {
        value: repl,
        enumerable: true,
        writable: false,
        configurable: false
      });
      delete globalThis.__turbosparkFileRequest;
    })();
    """

    private static let consoleInstallationScript = """
    (() => {
      const sink = globalThis.__turbosparkConsoleSink;
      const stringify = JSON.stringify;
      const stringValue = String;
      const format = values => {
        let formatted = "";
        for (let index = 0; index < values.length; index++) {
          const value = values[index];
          let part;
          if (typeof value === "string") {
            part = value;
          } else {
            try {
              const encoded = stringify(value);
              part = encoded === undefined ? stringValue(value) : encoded;
            } catch (_) {
              part = stringValue(value);
            }
          }
          if (index > 0) formatted += " ";
          formatted += part;
        }
        return formatted;
      };
      const console = {};
      for (const level of ["log", "info", "debug", "warn", "error"]) {
        const method = (...values) => sink(level, format(values));
        Object.freeze(method);
        Object.defineProperty(console, level, {
          value: method,
          enumerable: true,
          writable: false,
          configurable: false
        });
      }
      Object.freeze(console);
      Object.defineProperty(globalThis, "console", {
        value: console,
        enumerable: true,
        writable: false,
        configurable: false
      });
      delete globalThis.__turbosparkConsoleSink;
    })();
    """
}
