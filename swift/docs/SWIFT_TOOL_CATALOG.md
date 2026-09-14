# Swift Tool Implementations Catalog

For execution mechanics, sandboxing, containment, hooks, and adding or removing tools, see [SWIFT_TOOLS.md](SWIFT_TOOLS.md).
For native fused validation, durable output observations, and recall, see [SWIFT_AGENT_EFFICIENCY.md](SWIFT_AGENT_EFFICIENCY.md).
For Syntext indexed search integration, see [SYNTEXT.md](SYNTEXT.md).

This document is the complete reference catalog for all native and built-in tools in TurboSparkApp (`swift/TurboSparkApp/`), documenting parameters, execution behaviors, return payloads, security constraints, and usage patterns.

---

## Table of Contents

- [Overview](#overview)
- [File Operations](#file-operations)
  - [read_file - Multi-Mode File Inspection](#read_file---multi-mode-file-inspection)
  - [write_file - Direct File Writing](#write_file---direct-file-writing)
  - [edit_file - Targeted File Editing and Rollback](#edit_file---targeted-file-editing-and-rollback)
  - [apply_patch - Unified Diff Patch Application](#apply_patch---unified-diff-patch-application)
  - [snip - Precision Code Snippet Extraction](#snip---precision-code-snippet-extraction)
  - [notebookedit - Jupyter Notebook Cell Editing](#notebookedit---jupyter-notebook-cell-editing)
  - [senduserfile - User File Presentation](#senduserfile---user-file-presentation)
  - [recall_tool_output - Observation Recall](#recall_tool_output---observation-recall)
- [Search and Codebase Discovery](#search-and-codebase-discovery)
  - [grep_search - Syntext Indexed Code Search](#grep_search---syntext-indexed-code-search)
  - [search_code - Codebase Text and Regex Search](#search_code---codebase-text-and-regex-search)
  - [list_directory - Workspace Tree Navigation](#list_directory---workspace-tree-navigation)
- [Terminal and Execution](#terminal-and-execution)
  - [run_command / Bash - Shell Command Execution](#run_command--bash---shell-command-execution)
  - [bashoutput - Background Shell Output Polling](#bashoutput---background-shell-output-polling)
  - [killshell - Background Process Tree Termination](#killshell---background-process-tree-termination)
- [Web and Network Access](#web-and-network-access)
  - [websearch - Multi-Provider Web Search](#websearch---multi-provider-web-search)
  - [webfetch - Webpage Content Extraction](#webfetch---webpage-content-extraction)
  - [http_request - SSRF-Guarded REST Client](#http_request---ssrf-guarded-rest-client)
- [Interactive Planning and User Questions](#interactive-planning-and-user-questions)
  - [askuserquestion - Structured User Questionnaires](#askuserquestion---structured-user-questionnaires)
  - [enterplanmode / exitplanmode - Agent Mode Switching](#enterplanmode--exitplanmode---agent-mode-switching)
  - [reportfindings - Findings and Artifact Reporting](#reportfindings---findings-and-artifact-reporting)
  - [proposeskills / proposegoal - Skill and Goal Alignment](#proposeskills--proposegoal---skill-and-goal-alignment)
- [Tasks, Checklists, and Subagents](#tasks-checklists-and-subagents)
  - [todowrite - Interactive Checklist Management](#todowrite---interactive-checklist-management)
  - [agent - Subagent Delegation](#agent---subagent-delegation)
  - [taskcreate / tasklist / taskupdate - Task Management](#taskcreate--tasklist--taskupdate---task-management)
- [Automation, Crons, and Environment](#automation-crons-and-environment)
  - [croncreate / crondelete / cronlist - Cron Scheduling](#croncreate--crondelete--cronlist---cron-scheduling)
  - [sleep / delay - Bounded Delays](#sleep--delay---bounded-delays)
  - [pushnotification - User Notification Dispatch](#pushnotification---user-notification-dispatch)
  - [ctxinspect - Context and Telemetry Inspection](#ctxinspect---context-and-telemetry-inspection)
- [Persistent Memory and Worktrees](#persistent-memory-and-worktrees)
  - [memory - Auto-Memory Index and Storage](#memory---auto-memory-index-and-storage)
  - [enterworktree / exitworktree - Git Worktree Isolation](#enterworktree--exitworktree---git-worktree-isolation)
- [Extensibility: MCP and Custom Tools](#extensibility-mcp-and-custom-tools)
- [Tool Permission and Gating Matrix](#tool-permission-and-gating-matrix)
- [Technical Deep Dives](#technical-deep-dives)
  - [1. Syntext Trigram Index vs Ripgrep](#1-syntext-trigram-index-vs-ripgrep)
  - [2. Multi-Mode File Operations and Snapshot Rollbacks](#2-multi-mode-file-operations-and-snapshot-rollbacks)
  - [3. Background Process Tree Termination (SIGTERM to SIGKILL Sweep)](#3-background-process-tree-termination-sigterm-to-sigkill-sweep)
  - [4. SSRF Defense in Native HTTP Requests](#4-ssrf-defense-in-native-http-requests)
  - [5. Interactive User Question Park-and-Resume Continuations](#5-interactive-user-question-park-and-resume-continuations)
- [Agent Workflow Patterns and Best Practices](#agent-workflow-patterns-and-best-practices)

---

## Overview

TurboSpark provides over 30 native tools covering file manipulation, indexed semantic search, terminal execution, network access, interactive user alignment, task management, automation, and MCP client capabilities.

Unlike single-purpose scripting agents, TurboSpark tools run directly in-process within the Swift desktop application, interfacing with the underlying Rust inference engine and macOS system capabilities. Each tool features strict parameter validation, path containment checks (`resolveSecurePath`), write sandboxing (`AppToolSandbox`), and integration with the multi-tier permission engine (`AppToolPermissionEngine`).

---

## File Operations

### read_file - Multi-Mode File Inspection

- **Aliases**: `view_file`, `cat`, `fileread`, `read`
- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe (Auto-allowed in all modes)
- **Source**: `Tools/File/FileReadWriteTools.swift`, `Tools/Registry/AppToolRegistry+Handlers.swift`

**Purpose**: Read, inspect, search, diff, or audit text and document files with line bounds and metadata modes.

**Parameters**:
```json
{
  "path": "string (relative file path within project root)",
  "file_path": "string (alias for path)",
  "start_line": "integer (1-indexed start line)",
  "offset": "integer (alias for start_line)",
  "end_line": "integer (1-indexed end line)",
  "limit": "integer (maximum number of lines to read)",
  "pages": "string (optional PDF page range, e.g. '1-5')",
  "mode": "string ('lines' | 'stats' | 'preview' | 'search' | 'diff' | 'time_machine')",
  "search_pattern": "string (regex or substring query when mode is 'search')",
  "context_lines": "integer (context lines around matches when mode is 'search', default 2)",
  "comparison_path": "string (comparison file path when mode is 'diff')",
  "num_revisions": "integer (number of git revisions to inspect when mode is 'time_machine')"
}
```

**Modes**:
1. `lines` (Default): Returns 1-based numbered lines between `start_line` and `end_line` (capped at `AppFileReadLimits.maxLinesPerRead`).
2. `stats`: Returns file metadata (byte size, line count, word count, character count, SHA256 checksum, and modification time).
3. `preview`: Head (25 lines) and tail (65 lines) preview with line numbers and count of omitted lines.
4. `search`: In-file regex/substring search returning matching line numbers with surrounding context.
5. `diff`: Unified diff comparison against `comparison_path`.
6. `time_machine`: Git commit log and patch history for the target file.

**Safety Bounds**:
- File size checked before loading (`AppFileReadLimits.maxFileSize` = 16 MiB).
- Long lines bounded to prevent memory blowup.
- Non-UTF8 binary files return clear diagnostic errors rather than corrupt strings.

---

### write_file - Direct File Writing

- **Aliases**: `save_file`, `filewrite`, `write`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying (Prompts in `.ask`, allowed in `.auto` if within workspace)
- **Source**: `Tools/File/FileReadWriteTools.swift`, `Tools/Registry/AppToolRegistry+Handlers.swift`

**Purpose**: Create a new file or completely overwrite an existing file.

**Parameters**:
```json
{
  "path": "string (relative destination path)",
  "file_path": "string (alias for path)",
  "content": "string (file contents to write)",
  "validate_command": "string (optional command to run post-write for verification)",
  "validate_timeout_ms": "integer (optional timeout in ms for validate_command)"
}
```

**Key Features**:
- Automatically creates parent directories if missing.
- Refuses absolute paths or paths resolving outside workspace root (`resolveSecurePath`).
- Enforces write allow/deny policies from `AppToolSandbox`.
- Records file snapshot before write in `FileSnapshotStore` to allow rollback.
- Returns structured diff of additions/deletions.

---

### edit_file - Targeted File Editing and Rollback

- **Aliases**: `fileedit`, `edit`, `editor`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying
- **Source**: `Tools/File/FileReadWriteTools.swift`, `Tools/Registry/AppToolRegistry+Handlers.swift`

**Purpose**: Modify an existing file via targeted replacement, line insertion, regex replacement, or single-step rollback.

**Parameters**:
```json
{
  "path": "string (relative file path)",
  "file_path": "string (alias for path)",
  "command": "string ('str_replace' | 'insert' | 'pattern_replace' | 'undo_edit', default: 'str_replace')",
  "old_string": "string (exact target string to find and replace)",
  "new_string": "string (replacement or insertion string)",
  "replace_all": "boolean (true to replace all occurrences, false for first match)",
  "insert_line": "string (line number or text anchor for insert command)",
  "position": "string ('before' | 'after', default: 'after')",
  "regex_pattern": "string (regular expression pattern for pattern_replace)",
  "validate_command": "string (optional post-edit verification command)",
  "validate_timeout_ms": "integer (optional timeout for verification command)"
}
```

**Commands**:
1. `str_replace`: Exact substring substitution. Fails if `old_string` is not found.
2. `insert`: Injects `new_string` before or after `insert_line`.
3. `pattern_replace`: Regular expression match and replacement with capture group support (`$1`, `$2`).
4. `undo_edit`: Reverts the target file to the exact content state before the previous edit using `FileSnapshotStore.recordBackup`.

**Freshness Invariant**:
- Before applying any destructive edit, `FileSnapshotStore` validates that the file on disk has not changed since the last time the model read it. If the file was modified externally, the edit is aborted to prevent overwriting unseen changes.

---

### apply_patch - Unified Diff Patch Application

- **Aliases**: `applypatch`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying
- **Source**: `Tools/File/ApplyPatchTool.swift`, `Tools/File/ApplyPatchExecutor.swift`

**Purpose**: Apply Git-style unified diff patches across one or more files in the workspace.

**Parameters**:
```json
{
  "patch": "string (unified diff text)",
  "patch_text": "string (alias for patch)",
  "fuzz": "integer (optional context line fuzz tolerance, default: 2)",
  "dry_run": "boolean (optional dry run to validate patch application without disk write)"
}
```

**Features**:
- Supports multi-file patches with `--- a/path` and `+++ b/path` headers.
- Context line matching with configurable fuzz tolerance.
- Dry-run validation mode.
- Automatic creation and deletion of files flagged in diff headers.

---

### multiedit - Atomic Multi-File Editing

- **Aliases**: `multi_edit`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying
- **Source**: `Tools/File/MultiEditToolDefinitions.swift`, `Tools/File/MultiEditExecutor.swift`

**Purpose**: Execute coordinated modifications across multiple files in a single atomic transaction.

**Parameters**:
```json
{
  "edits": [
    {
      "file_path": "string (relative file path within project root)",
      "old_string": "string (exact target string to find and replace)",
      "new_string": "string (replacement string)",
      "replace_all": "boolean (optional, true to replace all occurrences)"
    }
  ]
}
```

**Key Features**:
- **Pre-flight Validation**: Verifies all files exist, paths are contained, snapshots are not stale, and target strings match uniquely before touching disk.
- **Automated Rollback**: If any file fails during modification or writing, all modified files in the batch are immediately rolled back to their pre-transaction snapshots.
- **Multi-File Refactoring**: Ideal for interface migrations, rename refactorings, and coordinated cross-file edits.

---

### snip - Precision Code Snippet Extraction

- **Aliases**: `extract_snippet`
- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/File/SnipExecutor.swift`

**Purpose**: Extract concise, semantically meaningful code blocks (functions, classes, blocks) by line range or symbol without loading entire files into model context.

---

### notebookedit - Jupyter Notebook Cell Editing

- **Aliases**: `notebook_edit`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying
- **Source**: `Tools/File/NotebookEditExecutor.swift`

**Purpose**: Read, modify, insert, or delete cells in `.ipynb` Jupyter Notebook JSON structures while preserving notebook metadata and cell outputs.

---

### senduserfile - User File Presentation

- **Aliases**: `send_user_file`
- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/File/SendUserFileExecutor.swift`

**Purpose**: Present a file directly to the user in the UI with download, preview, or attachment handles.

---

### recall_tool_output - Observation Recall

- **Category**: `.fileRead`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Core/ToolObservationTools.swift`

**Purpose**: Retrieve full untruncated output of a prior tool execution that was compacted or stored in the observation ledger (see [SWIFT_AGENT_EFFICIENCY.md](SWIFT_AGENT_EFFICIENCY.md)).

---

## Search and Codebase Discovery

### grep_search - Syntext Indexed Code Search

- **Aliases**: `Grep`, `search_code`
- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/File/SyntextCodeSearchTool.swift`, `Tools/File/FileSearchTools.swift`

**Purpose**: Sub-millisecond indexed text, symbol, and regex searching across the entire workspace using the embedded Syntext engine (see [SYNTEXT.md](SYNTEXT.md)).

**Parameters**:
```json
{
  "query": "string (search string, symbol name, or regex pattern)",
  "pattern": "string (alias for query)",
  "path": "string (optional subdirectory to scope search)",
  "case_sensitive": "boolean (default false)",
  "max_results": "integer (maximum matches returned, default 40, capped at 100)"
}
```

**Advantage over raw shell grep**:
- Does not spawn shell subprocesses.
- Bounded memory and match counts (prevents terminal token flooding).
- Respects project `.gitignore` and hidden directories automatically.
- Falls back gracefully to `searchCode` file crawler if Syntext indexing is disabled.

---

### search_code - Codebase Text and Regex Search

- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/Registry/AppToolRegistry+Handlers.swift`

**Purpose**: Walk the workspace directory tree and perform line-by-line pattern matching. Used when Syntext index is cold or unavailable.

---

### list_directory - Workspace Tree Navigation

- **Aliases**: `list_dir`, `ls`, `glob`
- **Category**: `.fileRead`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/Registry/AppToolRegistry+Handlers.swift`

**Purpose**: List files and subdirectories with sizes, types, and counts.

**Parameters**:
```json
{
  "path": "string (relative path, defaults to '.')",
  "recursive": "boolean (optional recursive scan)",
  "max_depth": "integer (optional depth limit for recursive scans)"
}
```

---

## Terminal and Execution

### run_command / Bash - Shell Command Execution

- **Aliases**: `Bash`, `shell`, `exec`, `terminal`
- **Category**: `.terminal`
- **Workspace Rooted**: Yes (starts at project root)
- **Permission Tier**: Gated by `AppToolPermissionEngine`, `CommandGate`, and `TerminalCommandClassifier`
- **Source**: `Tools/Terminal/ShellCommandRunner.swift`, `Tools/Terminal/TerminalTools.swift`

**Purpose**: Execute shell commands under `/bin/zsh` within the project root.

**Parameters**:
```json
{
  "command": "string (shell command line)",
  "cmd": "string (alias for command)",
  "timeout": "integer (timeout in milliseconds, default: 120000, max: 600000)",
  "description": "string (optional active-voice explanation of action)",
  "run_in_background": "boolean (if true, spawns background shell and returns task_id immediately)"
}
```

**Key Features**:
- **Merged Stream Order**: Interleaves `stdout` and `stderr` in exact temporal arrival order.
- **Persistent CWD**: `ShellCwdTracker` captures exit `pwd` via appended `pwd -P`. Subsequent calls start in that directory. If a command exits outside workspace, CWD resets to project root with an explanatory note.
- **Hang Prevention Environment**: Child processes inherit `GIT_EDITOR=true`, `GIT_PAGER=cat`, `PAGER=cat`, `TERM=dumb`, and `NO_COLOR=1`.
- **Compacted Output & Spill File**: Output is stripped of ANSI escape sequences and capped at 30,000 characters (20,000 head / 8,000 tail). Oversized outputs are persisted to `/tmp/turbospark-shell-spill-*` with path provided to the model.
- **Benign Exit Codes**: Commands such as `grep` (exit code 1 = no match), `diff` (exit code 1 = differences found), and `test`/`[` (exit code 1 = false) return normally with an informative suffix rather than throwing errors.

---

### bashoutput - Background Shell Output Polling

- **Aliases**: `bash_output`
- **Category**: `.terminal`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/Terminal/BackgroundShellManager.swift`

**Purpose**: Read output from a background shell spawned with `run_in_background: true`.

**Parameters**:
```json
{
  "task_id": "string (background shell id, e.g. 'bg_1')",
  "wait_seconds": "integer (seconds to wait up to 120; 0 for immediate non-blocking poll)"
}
```

---

### killshell - Background Process Tree Termination

- **Aliases**: `kill_shell`
- **Category**: `.terminal`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying / Control
- **Source**: `Tools/Terminal/BackgroundShellManager.swift`, `Tools/Core/ProcessExecutor.swift`

**Purpose**: Terminate a background shell task.

**Features**:
- Snapshots transitive descendant processes from the OS process table before signalling.
- Applies SIGTERM followed by SIGKILL ladder to guarantee process tree cleanup (including double-forked daemons and servers).

---

## Web and Network Access

### websearch - Multi-Provider Web Search

- **Aliases**: `web_search`, `search_web`
- **Category**: `.webSearch`
- **Workspace Rooted**: No
- **Permission Tier**: Safe / Read-only Web
- **Source**: `Tools/Web/WebSearchExecutor.swift`, `Tools/Web/WebTools.swift`

**Purpose**: Query external search engines for documentation, errors, and public information.

**Parameters**:
```json
{
  "query": "string (search query terms)",
  "num_results": "integer (number of results, default: 5)",
  "provider": "string ('tavily' | 'exa' | 'brave' | 'searxng' | 'parallel')"
}
```

**Supported Providers**:
- **Tavily**: Direct AI answer extraction + structured URLs.
- **Exa**: Semantic neural search and code context.
- **Brave / SearXNG / Parallel**: Independent web indexing endpoints.

---

### webfetch - Webpage Content Extraction

- **Aliases**: `web_fetch`, `fetch_url`, `read_url_content`
- **Category**: `.webFetch`
- **Workspace Rooted**: No
- **Permission Tier**: Safe / Read-only Web
- **Source**: `Tools/Web/WebFetchExecutor.swift`

**Purpose**: Fetch HTTP/HTTPS URL content and convert HTML into clean Markdown text with tag stripping and link preservation.

---

### codesearch - Code and API Documentation Search

- **Aliases**: `code_search`
- **Category**: `.webSearch`
- **Workspace Rooted**: No
- **Permission Tier**: Safe / Read-only Web
- **Source**: `Tools/Web/CodeSearchToolDefinitions.swift`, `Tools/Web/CodeSearchExecutor.swift`

**Purpose**: Specialized developer documentation and API reference search returning clean code snippets, type definitions, and library guides.

**Parameters**:
```json
{
  "query": "string (technical documentation search query)",
  "tokens_num": "integer (optional context token budget, default: 5000)",
  "framework": "string (optional framework or language hint, e.g. 'react', 'swift', 'rust')",
  "provider": "string (optional search provider: 'auto', 'tavily', 'exa')"
}
```

**Key Features**:
- Prioritizes authoritative developer documentation domains (docs.rs, developer.apple.com, react.dev, etc.).
- Formats results with clean Markdown code blocks and reference URLs.
- Direct AI answer extraction when supported by provider.

---

### http_request - SSRF-Guarded REST Client

- **Aliases**: `httprequest`
- **Category**: `.webFetch`
- **Workspace Rooted**: No
- **Permission Tier**: Gated / Network
- **Source**: `Tools/Web/HttpRequestTools.swift`, `Tools/Web/HttpRequestExecutor.swift`

**Purpose**: Execute direct HTTP/REST requests (GET, POST, PUT, DELETE, PATCH, HEAD) without spawning shell processes (`curl`).

**Parameters**:
```json
{
  "url": "string (target URL)",
  "method": "string ('GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD', default: 'GET')",
  "headers": "object (key-value dictionary of HTTP headers)",
  "body": "string (request body text or JSON)",
  "auth_type": "string ('bearer' | 'basic' | 'api_key' | 'custom')",
  "auth_token": "string (credentials or token)",
  "format": "string ('json' | 'markdown' | 'raw', default: 'json')",
  "timeout": "integer (timeout in seconds, default: 30, max: 120)"
}
```

**SSRF Protection Invariant**:
- Before executing any request, destination host is resolved against `AppToolSandbox.isPrivateOrMetadataHost(host)`.
- Rejects loopback (`127.0.0.1`, `localhost`, `::1`), RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), and AWS/GCP cloud metadata endpoints.

---

## Interactive Planning and User Questions

### askuserquestion - Structured User Questionnaires

- **Aliases**: `ask_user_question`, `ask_question`, `question`
- **Category**: `.planning`
- **Workspace Rooted**: No
- **Permission Tier**: Safe / Interactive
- **Source**: `Tools/Planning/PlanningInteractiveTools.swift`, `Tools/Planning/PlanningInteractiveExecutors.swift`

**Purpose**: Present interactive multiple-choice or write-in questions to the user in the UI, parking model execution until the user selects or inputs an answer.

**Parameters**:
```json
{
  "questions": [
    {
      "question": "string (full prompt text)",
      "header": "string (short category tag, max 30 chars)",
      "options": [
        {
          "label": "string (option title)",
          "description": "string (explanation of impact)",
          "preview": "string (optional preview snippet)"
        }
      ],
      "multiSelect": "boolean (allow selecting multiple options, default false)"
    }
  ]
}
```

**Shorthand form**:
```json
{
  "question": "Should we proceed with database migration?",
  "header": "Migration",
  "options": ["Proceed", "Abort"]
}
```

**Execution Lifecycle**:
- Parks execution on a Swift async continuation (`answerWaiter`).
- Renders an interactive banner in the UI.
- User dismissal or cancellation resumes the continuation with a default dismissal notice without throwing an unhandled exception.

---

### enterplanmode / exitplanmode - Agent Mode Switching

- **Aliases**: `enter_plan_mode`, `exit_plan_mode`
- **Category**: `.planning`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Planning/PlanningInteractiveTools.swift`

**Purpose**: Transition the session between read-only planning/research mode and active code-generation/build mode.

---

### reportfindings - Findings and Artifact Reporting

- **Aliases**: `report_findings`, `findings`
- **Category**: `.planning`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Planning/PlanningInteractiveTools.swift`

**Purpose**: Produce structured architectural findings, analysis summaries, or verification checklists displayed prominently in the session transcript.

---

### proposeskills / proposegoal - Skill and Goal Alignment

- **Aliases**: `propose_skills`, `propose_goal`
- **Category**: `.planning`
- **Workspace Rooted**: Yes (for skills writing to `.agents/skills`)
- **Permission Tier**: Modifying
- **Source**: `Tools/Planning/PlanningInteractiveTools.swift`

**Purpose**: Suggest new reusable project skills or register autonomous background goals with stopping conditions (see [SWIFT_GOALS.md](SWIFT_GOALS.md)).

---

## Tasks, Checklists, and Subagents

### todowrite - Interactive Checklist Management

- **Aliases**: `todo_write`
- **Category**: `.fileWrite`
- **Workspace Rooted**: No
- **Permission Tier**: Safe / UI Task Sync
- **Source**: `Tools/Tasks/TaskItemTools.swift`, `Tools/Tasks/TodoWriteExecutor.swift`

**Purpose**: Create and update structured task checklists rendered reactively in the transcript panel.

**Parameters**:
```json
{
  "todos": [
    {
      "content": "string (task description)",
      "status": "string ('pending' | 'in_progress' | 'completed' | 'cancelled')",
      "activeForm": "string (optional present-tense description while active)"
    }
  ]
}
```

**Features**:
- Accepts four loose JSON formats emitted by various model families.
- Automatically updates the persistent transcript checklist panel (`TaskChecklistPanelView`).
- Completed items are visually settled while in-progress and pending items remain prominent.

---

### batch - Parallel Tool Execution

- **Category**: `.automation`
- **Workspace Rooted**: No
- **Permission Tier**: Safe meta-tool (Child calls inherit individual permissions)
- **Source**: `Tools/Tasks/BatchToolDefinitions.swift`, `Tools/Tasks/BatchToolExecutor.swift`

**Purpose**: Execute multiple independent tool calls in parallel (up to 25 calls) with non-blocking partial failure handling.

**Parameters**:
```json
{
  "tool_calls": [
    {
      "tool": "string (name of tool to execute)",
      "parameters": {
        "arg1": "val1"
      }
    }
  ]
}
```

**Key Features**:
- **Concurrent Execution**: Dispatches child calls simultaneously using Swift task groups.
- **Partial Failure Resilience**: If one tool call fails, remaining sibling calls continue and complete normally.
- **Anti-Recursion**: Rejects recursive `batch` calls.
- **Structured Output**: Summarizes total, successful, and failed counts along with individual call outputs.

---

### agent - Subagent Delegation

- **Aliases**: `subagent`, `task`
- **Category**: `.task`
- **Workspace Rooted**: No
- **Permission Tier**: Bounded by subagent permission rules
- **Source**: `Tools/Tasks/AgentTools.swift`, `State/SubagentRunner.swift`

**Purpose**: Spawn an autonomous subagent with a dedicated task, bounded recursion depth (`subagentDepth`), and sandboxed tool profile.

**Parameters**:
```json
{
  "task": "string (instruction prompt for the subagent)",
  "agent_type": "string ('coder' | 'researcher' | 'general')",
  "run_in_background": "boolean (run asynchronously in background agents strip)"
}
```

---

### taskcreate / tasklist / taskupdate - Task Management

- **Aliases**: `task_create`, `task_list`, `task_update`, `task_stop`, `task_output`
- **Category**: `.task`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Tasks/TaskItemTools.swift`, `Tools/Tasks/TaskManager.swift`

**Purpose**: Track multi-step background tasks, poll completion status, and retrieve task outputs.

---

## Automation, Crons, and Environment

### croncreate / crondelete / cronlist - Cron Scheduling

- **Category**: `.automation`
- **Workspace Rooted**: No
- **Permission Tier**: Automation gated
- **Source**: `Tools/Automation/WorkflowCronTools.swift`, `Tools/Automation/CronScheduler.swift`

**Purpose**: Schedule recurring background tasks and agent wakeups using 5-field cron syntax.

---

### sleep / delay - Bounded Delays

- **Category**: `.automation`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Automation/AutomationExecutors.swift`

**Purpose**: Suspend execution for a specified duration in seconds (capped at 300 seconds).

---

### pushnotification - User Notification Dispatch

- **Aliases**: `notify`
- **Category**: `.automation`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Automation/MonitoringNotificationTools.swift`

**Purpose**: Send macOS user notifications to alert the user when long-running builds or background subagents finish.

---

### ctxinspect - Context and Telemetry Inspection

- **Category**: `.automation`
- **Workspace Rooted**: No
- **Permission Tier**: Safe
- **Source**: `Tools/Automation/MonitoringNotificationTools.swift`

**Purpose**: Inspect current prompt token usage, context ring breakdown, and active agent limits.

---

## Persistent Memory and Worktrees

### memory - Auto-Memory Index and Storage

- **Aliases**: `remember`
- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Safe
- **Source**: `Tools/Memory/MemoryTool.swift`

**Purpose**: Query and record facts, decisions, and patterns in the project's `.turbospark/memory/` store and `MEMORY.md` index (see [SWIFT_MEMORY.md](SWIFT_MEMORY.md)).

---

### enterworktree / exitworktree - Git Worktree Isolation

- **Category**: `.fileWrite`
- **Workspace Rooted**: Yes
- **Permission Tier**: Modifying
- **Source**: `Tools/Projects/ArtifactWorktreeTools.swift`, `Tools/Projects/WorktreeExecutor.swift`

**Purpose**: Create an isolated git worktree branch to perform experimental code changes without disturbing the main repository branch.

---

## Extensibility: MCP and Custom Tools

1. **Custom JSON Tools**: Located in `.turbospark/tools/*.json` or `~/.turbospark/tools/*.json`. Declares parameter schemas and maps to execution commands with argument injection via `TOOL_ARG_<KEY>` environment variables.
2. **Model Context Protocol (MCP)**: Supports stdio and SSE servers declared in `.mcp.json`. Exposes server tools as `mcp__<server>__<tool>` and resource tools via `call_mcp_tool`, `list_mcp_resources`, and `read_mcp_resource`.

---

## Tool Permission and Gating Matrix

The table below summarizes tool permissions across TurboSpark's four operating modes:
- **Strict Read-Only**: Disables all modifying tools (writes, edits, shell commands, custom tools).
- **Ask**: Prompts the user via an approval sheet for any modifying action.
- **Auto**: Auto-approves safe actions and shell commands verified against the allowlist; prompts for high-risk actions.
- **Permissive**: Grants unrestricted execution to workspace-confined actions.

| Tool Name | Category | Rooted | Safe Arm | Strict Read-Only | Ask Mode | Auto Mode | Permissive Mode |
|---|---|---|---|---|---|---|---|
| `read_file` | `.fileRead` | Yes | Yes | Allow | Allow | Allow | Allow |
| `write_file` | `.fileWrite` | Yes | No | Deny | Ask | Allow (in root) | Allow |
| `edit_file` | `.fileWrite` | Yes | No | Deny | Ask | Allow (in root) | Allow |
| `multiedit` | `.fileWrite` | Yes | No | Deny | Ask | Allow (in root) | Allow |
| `apply_patch` | `.fileWrite` | Yes | No | Deny | Ask | Allow (in root) | Allow |
| `list_directory` | `.fileRead` | Yes | Yes | Allow | Allow | Allow | Allow |
| `grep_search` | `.fileRead` | Yes | Yes | Allow | Allow | Allow | Allow |
| `run_command` | `.terminal` | Yes | Evaluated | Deny | Ask | Allowlist / Gate | Allow |
| `bashoutput` | `.terminal` | Yes | Yes | Allow | Allow | Allow | Allow |
| `killshell` | `.terminal` | Yes | Yes | Allow | Allow | Allow | Allow |
| `websearch` | `.webSearch` | No | Yes | Allow | Allow | Allow | Allow |
| `codesearch` | `.webSearch` | No | Yes | Allow | Allow | Allow | Allow |
| `webfetch` | `.webFetch` | No | Yes | Allow | Allow | Allow | Allow |
| `http_request` | `.webFetch` | No | No | Deny | Ask | Gated (SSRF check) | Allow |
| `askuserquestion` | `.planning` | No | Yes | Allow | Allow | Allow | Allow |
| `enterplanmode` | `.planning` | No | Yes | Allow | Allow | Allow | Allow |
| `exitplanmode` | `.planning` | No | Yes | Allow | Allow | Allow | Allow |
| `todowrite` | `.fileWrite` | No | Yes | Allow | Allow | Allow | Allow |
| `batch` | `.automation` | No | Evaluated | Allow | Allow | Allow | Allow |
| `agent` | `.task` | No | Evaluated | Deny | Ask | Capped Depth | Allow |
| `memory` | `.fileWrite` | Yes | Yes | Allow | Allow | Allow | Allow |
| `croncreate` | `.automation` | No | No | Deny | Ask | Deny | Allow |
| `mcp__*` | `.mcp` | Yes | Evaluated | Deny | Ask | Auto-Approve rule | Allow |

---

## Technical Deep Dives

### 1. Syntext Trigram Index vs Ripgrep

TurboSpark replaces raw shell grepping with the embedded Syntext engine:
- Pre-indexes code tokens, trigrams, and symbols into SQLite and memory buffers.
- Eliminates process creation overhead and prevents the shell from emitting unbound gigabytes of output into model context.
- Maintains line number parity and automatic `.gitignore` exclusion.

### 2. Multi-Mode File Operations and Snapshot Rollbacks

TurboSpark's file operations are protected against drift and stale writes:
- `read_file` supports structured operational modes (`stats`, `preview`, `diff`, `time_machine`, `search`) that allow models to inspect massive repositories without exhausting token context.
- `FileSnapshotStore` tracks SHA256 hashes of all accessed files. A destructive write throws if the file was modified since its last inspection.
- The `undo_edit` command in `edit_file` allows instant single-step rollback of failed code changes.

### 3. Background Process Tree Termination (SIGTERM to SIGKILL Sweep)

Standard shell termination often leaves orphan server processes or daemons running. TurboSpark addresses this by:
- Inspecting the OS process table for all transitive descendant process IDs before signalling the leader.
- Signalling SIGTERM, allowing a grace period, and then sending SIGKILL to all surviving descendants.
- Exposing UI kill surfaces (`BackgroundShellsStripView` and `AppModel.stopAll`) so users can abort runaway scripts instantly.

### 4. SSRF Defense in Native HTTP Requests

To prevent local network scanning or cloud credential exfiltration via `http_request`:
- Destination hosts are resolved to IP addresses before dispatch.
- Loopback (`127.0.0.1`, `localhost`), RFC 1918 private subnets, link-local addresses (`169.254.169.254`), and cloud metadata APIs are unconditionally blocked.

### 5. Interactive User Question Park-and-Resume Continuations

Unlike CLI tools that block stdin or web shells that fail on interactive prompts:
- `AskUserQuestionExecutor` parks the Swift async agent loop using `CheckedContinuation`.
- The UI renders an interactive card with selectable choices and write-in text fields.
- Resuming the continuation returns the user's explicit choices directly into the agent transcript without generating extraneous error rounds.

---

## Agent Workflow Patterns and Best Practices

### Pattern 1: Search and Inspect Before Edit
```
1. grep_search: Locate target functions or symbols across the workspace.
2. read_file (mode: "lines" or "preview"): Inspect surrounding context.
3. edit_file (command: "str_replace"): Perform precise surgical edits.
4. run_command: Run targeted test suite to verify changes.
```

### Pattern 2: Multi-File Atomic Patching
```
1. read_file: Review multiple dependent files.
2. apply_patch (dry_run: true): Test patch applicability against active files.
3. apply_patch: Commit changes simultaneously across files.
```

### Pattern 3: Non-Blocking Background Daemon / Test Monitoring
```
1. run_command (run_in_background: true): Launch long-running compilation or test suite.
2. bashoutput (wait_seconds: 5): Periodically check compilation progress while continuing other inspection tasks.
3. killshell: Gracefully shut down background test server when testing is complete.
```

### Pattern 4: User Alignment for Ambiguous Requirements
```
1. askuserquestion: Clarify user intent between multiple design alternatives.
2. enterplanmode: Draft implementation plan based on user response.
3. todowrite: Initialize progress checklist before executing code modifications.
```
