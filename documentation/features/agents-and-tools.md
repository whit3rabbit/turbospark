---
description: Local agent tools, permission gates, MCP servers, and tool-call rescue guardrails.
icon: robot
---

# Agent Tools and Guardrails

TurboSpark includes a native agent runtime enabling local models to interact with your Mac, read documents, execute code, and query external data sources safely.

## Built-In Agent Tools

Models running in TurboSpark can invoke several built-in native tools:

| Tool | Purpose | Capability Scope |
| --- | --- | --- |
| **Shell / Bash** | Command Execution | Run build scripts, command-line utilities, and system commands in a controlled shell. |
| **File System** | File Access | Read, write, and search files within specified workspace directories. |
| **Web Fetch** | Documentation & APIs | Perform HTTP GET requests to fetch public API data and developer documentation. |
| **Git** | Version Control | Inspect Git status, diffs, branches, and commit history. |
| **JavaScript REPL** | Quick Math & Logic | Sandboxed JavaScript execution environment for deterministic calculations. |
| **MCP Client** | Extensibility | Connect to local or remote Model Context Protocol (MCP) servers for custom tools. |

---

## The Permission Gate

To guarantee security, **no model can perform an external action without your permission**:

1. **Interactive Modal**: The first time a tool is called, TurboSpark halts execution and displays the **Permission Gate**.
2. **Full Transparency**: The prompt displays:
   * The tool name (e.g. `execute_shell_command`).
   * The exact arguments (e.g. `cargo test --workspace`).
   * The target directory or URL.
3. **Choice of Action**:
   * **Allow Once**: Authorizes this specific invocation only.
   * **Allow for this Session**: Authorizes identical invocations for the duration of the current chat.
   * **Deny**: Rejects the action. The model receives a structured rejection message and must adjust its plan without executing the tool.

---

## Tool-Call Guardrails & The Rescue Tier

Small local models sometimes produce malformed JSON when calling tools (such as unescaped newlines, markdown code fence wraps, or trailing commas).

Rather than failing the request and returning an error, TurboSpark's server and runtime apply an intelligent **Tool-Call Guardrail Pipeline**:

1. **Native Parsing**: Fast-path JSON schema validation against model output.
2. **Markdown Extraction**: If the model wrapped its tool call in ```` ```json ```` fences or conversational preambles, the extractor extracts the pure payload.
3. **JSON Repair Tier**:
   * Repairs unescaped quotes within string literals.
   * Fixes trailing commas in argument arrays and objects.
   * Normalizes numeric types (e.g. string integers coerced to numeric values).
4. **Validation & Retry Budget**:
   * If parameters fail required field checks, the engine sends a targeted schema retry prompt to the model with a bounded retry budget (default: 2 retries).
   * This rescues over 85% of near-miss tool calls without requiring user intervention.
