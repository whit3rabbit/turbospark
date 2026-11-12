# Tools/Web

## Overview
The native web tools exposed to the model: `WebSearch`, `WebFetch`, `HttpRequest`, and `codesearch`. The directory holds each tool's schema definition, its input and output types, and its executor. All outbound requests share one SSRF defense: a destination validator (network policy, DNS resolution, private and metadata address refusal), per-hop redirect validation, a check of the address the socket actually connected to (DNS rebinding guard), and hard byte caps on response bodies.

## Interface
- Tool schemas [WebTools.swift:WebToolDefinitions] (`webSearch`, `webFetch`, and `all`, which also lists the HttpRequest tool), [HttpRequestTools.swift:HttpRequestToolDefinitions], [CodeSearchToolDefinitions.swift:CodeSearchToolDefinitions] (`codesearch`, `all`).
- Input and output value types [WebTools.swift:WebSearchInput], [WebTools.swift:WebSearchResultItem], [WebTools.swift:WebSearchOutput] (`formatMarkdown`), [WebTools.swift:WebFetchInput], [WebTools.swift:WebFetchOutput], [HttpRequestTools.swift:HttpRequestInput], [HttpRequestTools.swift:HttpRequestOutput] (`formatResponse`).
- Executors:
  - [WebSearchExecutor.swift:WebSearchExecutor] with `search(query:...)`, providers `searchExa`, `searchParallel`, `searchBrave`, `searchSearXNG`, `searchTavily`, `extractTavily`, parsers `parseExaPayload`, `parseExaFormattedText`, `parseParallelPayload`, and `sanitize`.
  - [WebFetchExecutor.swift:WebFetchExecutor] with `fetch(url:...)`, `convert`, `convertHTMLToMarkdown`, `extractTextFromHTML`, `decodeHTMLEntities`, and MIME helpers.
  - [HttpRequestExecutor.swift:HttpRequestExecutor] with `execute(url:...)`, the shared `performRequest`, `boundedData`, and `refuseConnectedPrivateAddress`.
  - [CodeSearchExecutor.swift:CodeSearchExecutor.execute] (wraps `WebSearchExecutor.search`).
- Destination guard [HttpRequestDestinationValidator.swift:HttpRequestDestinationValidator.validate] (internal) and redirect handling [HttpRequestExecutor.swift:HttpRequestRedirectDelegate].
- Subdirectories: none.

## Invariants
- Every outbound hop goes through `HttpRequestDestinationValidator.validate`: network policy decision first, then `getaddrinfo`. A public host must resolve only to public addresses regardless of local-host policy (rebinding guard); a local host the policy allows may resolve only within its own scope via `permitsLocalConnection`. An unresolvable host or one with no usable address is refused. [HttpRequestDestinationValidator.swift:validate]
- The search providers (Exa, Parallel, Brave, SearXNG, Tavily search and extract) send every request through `HttpRequestExecutor.performRequest` via `WebSearchExecutor.fetchLimited`, so they get the same validation, redirect revalidation, connected-address check and an 8 MiB cap. A user-configured SearXNG base URL is therefore subject to the network policy: metadata and `.internal` hosts are always refused, loopback follows "Allow localhost" (on by default), and LAN hosts need "Allow local network". An explicitly configured local SearXNG works only where the policy allows it. An injected test session has its request URL validated before sending but does not get redirect handling. [WebSearchExecutor.swift:fetchLimited]
- Redirects are never auto-followed in production: the delegate captures the redirect, cancels it, and `performRequest` revalidates the new URL and replays it, up to 10 hops. [HttpRequestExecutor.swift:performRequest], [HttpRequestExecutor.swift:HttpRequestRedirectDelegate]
- After each hop the connected remote addresses from task metrics are checked, and a missing address is refused, so a DNS-rebound response is discarded. Known residual risk (stated in code): the check runs after the request was sent, so only the response is withheld. [HttpRequestExecutor.swift:performRequest]
- A test that injects a destination validator into `WebFetchExecutor.fetch` opts out of the connected-address check unless it injects its own; production (no injection) always checks. Injected sessions only validate the final URL. [WebFetchExecutor.swift:fetch], [HttpRequestExecutor.swift:performRequest]
- Body size is capped while streaming, not after buffering: a declared length over the cap is refused before reading and an overrun aborts mid-stream (5 MB for fetch and HttpRequest, 8 MiB for search). Rendered HttpRequest output is capped at 200,000 characters and 150 lines. [HttpRequestExecutor.swift:boundedData], [WebSearchExecutor.swift:fetchLimited], [HttpRequestExecutor.swift:maxOutputCharacters]
- Timeouts are clamped: fetch and HttpRequest default 30 s, max 120 s, minimum 1 s; search 25 s. [WebFetchExecutor.swift], [HttpRequestExecutor.swift], [WebSearchExecutor.swift]
- HttpRequest enforces the network policy's per-method decision (custom mode read-only restriction) in addition to host validation, and accepts only GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS. [HttpRequestExecutor.swift:execute]
- WebFetch accepts only http and https, refuses image and non-textual content types, and returns HTML conversions only for HTML content types. On a 403 it retries once with a fallback user agent through the same validated path. Entity decoding runs `&amp;` last to avoid double-decoding. [WebFetchExecutor.swift:fetch], [WebFetchExecutor.swift:decodeHTMLEntities]
- Search results are sanitized before reaching the model: http or https only, length limits, URLs with embedded credentials dropped, private or metadata hosts dropped, blocked and allowed domain lists matched by exact host or dot-suffix, title and snippet truncated. [WebSearchExecutor.swift:sanitize]
- Exa reports quota and rate-limit failures as HTTP 200 with an MCP error payload; these must throw so they are not read as "no results" and so auto mode still falls back to Parallel. Auto mode tries Brave when a key is given, otherwise Exa then Parallel. [WebSearchExecutor.swift:exaErrorMessage], [WebSearchExecutor.swift:search]
- `codesearch` bounds results to 3 through 10 from `tokens_num` and appends "documentation reference" to the query unless it already contains "doc" or "api". `developerDomains` is declared but not passed as an allow list in `execute`. [CodeSearchExecutor.swift:execute]
- Tool schema names are part of the model contract: `WebSearch`, `WebFetch`, `HttpRequest`, `codesearch`. [WebTools.swift:WebToolDefinitions], [CodeSearchToolDefinitions.swift]

## Side Effects
- Network: HTTPS requests to Exa MCP (`mcp.exa.ai`), Parallel MCP (`search.parallel.ai`), Brave API, a user-configured SearXNG instance, Tavily (`api.tavily.com`), and arbitrary user or model-chosen URLs for fetch and HttpRequest. [WebSearchExecutor.swift], [WebFetchExecutor.swift], [HttpRequestExecutor.swift]
- Credentials leave the process in requests: Exa key as a URL query item, Parallel as a Bearer header, Brave as `X-Subscription-Token`, Tavily in the JSON body; HttpRequest sends caller-supplied Authorization or `X-API-Key` headers. [WebSearchExecutor.swift], [HttpRequestExecutor.swift:execute]
- Reads the `TAVILY_API_KEY` environment variable as a fallback. [WebSearchExecutor.swift:searchTavily]
- DNS resolution through `getaddrinfo` during validation. [HttpRequestDestinationValidator.swift]
- Reads `AppNetworkPolicyStore.current` and `AppToolSandbox` host classification. [HttpRequestDestinationValidator.swift], [HttpRequestExecutor.swift]
- Creates and invalidates an ephemeral `URLSession` per call (unless a test session is injected); the search providers do the same through `fetchLimited` (no more `URLSession.shared`). [HttpRequestExecutor.swift:performRequest], [WebSearchExecutor.swift]
- No disk writes.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
