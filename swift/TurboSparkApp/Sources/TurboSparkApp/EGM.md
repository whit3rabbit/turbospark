# TurboSparkApp (app target)

## Overview
Main executable target of the macOS app. Directories group UI, state, tools, audio, browser, generation, installation, server and design-system code.

## Subdirectories & Boundaries
- App: [./App/EGM.md](./App/EGM.md) - Application shell: the SwiftUI app entry point, root view, window sizing, keyboard shortcut catalog, profile unlock gate and the settings panes and control catalog that configure the app.
- Audio: [./Audio/EGM.md](./Audio/EGM.md) - Audio workspace: capture, recording permissions, live feed, media IO, library and search, custom vocabulary, export, and the SwiftUI views for transcription, speech, music and meeting notes.
- Browser: [./Browser/EGM.md](./Browser/EGM.md) - In-app browser: tab store, pane model and views, per-origin settings and bookmarks, viewport control and dialog approval UI.
- BrowserAutomation: [./BrowserAutomation/EGM.md](./BrowserAutomation/EGM.md) - Agent-driven browser control: automation session and coordinator, control command protocol, DOM snapshots, permission rules and approval, network isolation, WebKit engine and logging.
- Chrome: [./Chrome/EGM.md](./Chrome/EGM.md) - Window chrome views: sidebar, navigation rail, top and status bars, telemetry readouts, model loader control and profile menu.
- Components: [./Components/EGM.md](./Components/EGM.md) - Shared settings panes, sheets and reusable controls (agents, models, MCP, permissions, profiles, downloads, cron jobs, appearance, skills).
- ComputerUse: [./ComputerUse/EGM.md](./ComputerUse/EGM.md) - App-side computer-use support: per-chat session grants, app policy tiers, settings pane and the port that talks to the helper process.
- Diagnostics: [./Diagnostics/EGM.md](./Diagnostics/EGM.md) - Inspector UI: sampling controls, context-ladder selection, essentials section and runner diagnostics.
- Files: [./Files/EGM.md](./Files/EGM.md) - Attachment import, thumbnails, @-mention and context reference parsing, project file index, scan rules and file previews.
- Generation: [./Generation/EGM.md](./Generation/EGM.md) - Chat and generation UI: streaming rows, tool cards, composer, transcript follow, artifact panel, background agent and shell strips, git and worktree sheets.
- Humanize: [./Humanize/EGM.md](./Humanize/EGM.md) - Humanize feature: controller, planner, rules, diff and the engine protocol that rewrites text, plus its views.
- Images: [./Images/EGM.md](./Images/EGM.md) - Image generation workspace: composer, gallery, style catalog and favorites, resolution presets, model controls and the MLX image generation session.
- Import: [./Import/EGM.md](./Import/EGM.md) - Importers for external data: Chrome bookmarks and Markdown frontmatter parsing.
- Installation: [./Installation/EGM.md](./Installation/EGM.md) - Model installation and management UI: catalog and hub install gating, audio model catalogs, download ETA, context ladder views, feature descriptors and steering-vector import.
- Presentation: [./Presentation/EGM.md](./Presentation/EGM.md) - Rendering helpers: Markdown and code highlighting, ANSI colorizing, document text extraction, speech synthesizer, greeting provider and Office archive reading.
- Server: [./Server/EGM.md](./Server/EGM.md) - Local API server panes: Text, Image, Audio and Embeddings API UI, Keychain-backed server keys, advanced settings and agent terminal launch.
- State: [./State/EGM.md](./State/EGM.md) - Application state and persistence layer: AppModel and its extensions, stores, agent management, parsers, stream recovery and model state. Largest directory in the app.
- Theme: [./Theme/EGM.md](./Theme/EGM.md) - Design system implementation: appearance, fonts, language and strings, colors and contrast, controls, chrome layout, attention and motion.
- Tools: [./Tools/EGM.md](./Tools/EGM.md) - Agent tool system of the app: definitions, executors, permission and sandbox gates, registry, MCP client, hooks, shells, web, file and planning tools. Each subdirectory owns one tool family.
- Workflows: [./Workflows/EGM.md](./Workflows/EGM.md) - Workflow scripting: AST, checker and manifest compiler, actor bridge and workflow world.
