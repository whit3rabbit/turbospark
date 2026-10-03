import JavaScriptCore

typealias REPLConsoleSink = @Sendable (REPLTextOutputEvent) -> Void

/// Installs the worker's host-visible console methods onto a fresh context.
/// The app never receives JavaScriptCore values or objects through this seam.
final class REPLHostFacade: @unchecked Sendable {
    private let output: REPLConsoleSink

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
        Object.defineProperty(console, level, {
          value: (...values) => sink(level, format(values)),
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
