//! `turbospark-server`: binds the generation router (OpenAI
//! `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`) to
//! loopback, or to this machine's Tailscale IPv4 address. Two modes:
//!
//!   turbospark-server --model <install-dir|alias> [--port N]
//!                   [--max-context N|auto] [--expert-cache-slots auto|N]
//!                   [--bind loopback|tailnet]
//!                   [--power-profile performance|balanced|efficiency]
//!                   [--max-tokens-per-sec R]
//!   turbospark-server <tokenizer-dir> [port]
//!   turbospark-server --help | --version
//!
//! The first serves real generation from a `.gturbo` install through
//! `RealForwardRunner` (macOS only; one runner per process, requests
//! serialized). Its `--model` takes a directory or a `turbospark-model`
//! alias, resolved through the same `catalog::resolve_model_arg` that backs
//! `turbospark-check --model`. The second is the portable scripted mode: it
//! takes only a tokenizer and every response comes from `ScriptedChatModel`,
//! a fixed placeholder sequence, not a forward pass; it always binds
//! loopback.
//!
//! Default port 8080, default bind loopback. `--bind tailnet` requires
//! `--api-key` or `$TURBOSPARK_API_KEY`; it provides no TLS.

mod args;
mod bind;

use std::path::PathBuf;
use std::sync::Arc;

use args::{parse_model_args, short_circuit, BindMode, ModelArgs, USAGE};
use tokenizer::MfTokenizer;

#[cfg(target_os = "macos")]
fn open_real_model(args: &ModelArgs) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    // `--model` takes a path OR a `turbospark-model` alias, resolved the same
    // way `turbospark-check` resolves it, so one install serves both binaries
    // under one name. An existing directory always wins over an alias: a bare
    // name that silently preferred an alias would serve a DIFFERENT model
    // than the one on the command line, and a server does that unattended.
    let dir = catalog::resolve_model_arg(&args.model);
    // A 13 GB install takes a noticeable while to map and compile pipelines
    // for; without this line the startup reads as hung. Print what the
    // argument RESOLVED to when the two differ, since an alias says nothing
    // about which directory is being served.
    if dir.as_os_str() == args.model.as_str() {
        eprintln!("opening {} ...", args.model);
    } else {
        eprintln!("opening {} ({}) ...", args.model, dir.display());
    }
    // Resolved once here, which is also the one place this process asks the
    // OS about Low Power Mode.
    let profile = runtime::resolve_profile(args.power_profile);
    let rate = runtime::rate_control_for(profile, args.max_tokens_per_sec);
    let model = turbospark_server::RealChatModel::open(
        &dir,
        args.max_context,
        args.expert_cache_slots,
        rate,
        args.speculation,
        args.drafter,
        args.guardrails,
        args.steering.clone(),
        args.load_policy,
        args.reasoning,
        args.default_system.clone(),
        args.prefix_reuse,
        args.session_slots,
        args.vision_sidecar.as_deref().map(std::path::Path::new),
        args.kv_bits,
    )?;
    // Both sized figures are the RESOLVED ones, never `args`: under `auto`
    // the request carries no number, and each has to be readable beside any
    // throughput or footprint the operator goes on to measure. The context
    // line additionally names the checkpoint's own trained window, which is
    // what explains an `auto` of 4,096 on a machine with room for more.
    let context = model.context_plan();
    eprintln!(
        "model open (max_context {}{} [{}, {:.0} MiB of KV, suggested {}{}{}], \
         {} expert cache slots{}, {} profile, rate cap {})",
        context.resolved,
        if args.max_context.is_none() {
            " (auto)"
        } else {
            ""
        },
        match context.trained {
            Some(t) => format!("model {t}"),
            None => "model declares none".to_string(),
        },
        context.kv_bytes as f64 / (1024.0 * 1024.0),
        context.suggested,
        // Named only when it is not the default, matching `turbospark-check`'s
        // line. Reported for the reason the slot count beside it is: the tier
        // moves the suggestion, so no window or KV figure from this process is
        // comparable to another without it.
        match args.load_policy.guard {
            runtime::LoadGuard::Relaxed => String::new(),
            other => format!(", guard {}", other.as_str()),
        },
        match args.load_policy.min_auto_context {
            0 => String::new(),
            n => format!(", floor {n}"),
        },
        model.expert_cache_slots(),
        if args.expert_cache_slots.is_none() {
            " (auto)"
        } else {
            ""
        },
        profile.as_str(),
        match rate.max_tokens_per_sec {
            Some(r) => format!("{r} tok/s"),
            None => "none".to_string(),
        }
    );
    // A WARNING AND NOT SILENCE, for the reason `open_session` prints the
    // same line: an install carrying a drafter and decoding one token at a
    // time with nothing said is the failure the feature was built to end. A
    // server says it once, at startup, where an operator sees it.
    eprintln!("{}", model.speculation_line());
    eprintln!(
        "  guardrails: {}",
        if args.guardrails.active() {
            "on (tool calls rescued and argument-checked; a request with tools is buffered, not streamed)"
        } else {
            "off"
        }
    );
    eprintln!(
        "  prefix reuse: {}",
        if args.prefix_reuse {
            "on (a request continues from the previous request's KV where prompts agree; \
             helps only when consecutive requests are the same conversation)"
        } else {
            "off"
        }
    );
    // Reported whenever the resolved pool holds more than the one live
    // session, i.e. whenever `--session-slots` moved it -- matching how the
    // guardrails/prefix-reuse lines above always print (a fixed toggle) but
    // this one, like the reasoning line below, only says something when
    // there is something to say.
    let session_pool_size = model.session_pool_size();
    if session_pool_size > 1 {
        eprintln!(
            "  session slots: {session_pool_size} (this runner may hold reusable KV/recurrent \
             state for {session_pool_size} distinct conversations at once)"
        );
    }
    if args.reasoning != tokenizer::ReasoningEffort::Off {
        eprintln!("  default reasoning: {}", args.reasoning.as_str());
    }
    // Reported only when NOT off, matching the reasoning line above: this
    // flag does not RESOLVE against the machine the way the two sized knobs
    // do, so there is no suggestion to surface on the common (off) path.
    if args.kv_bits.is_on() {
        eprintln!("  kv-bits: {}", args.kv_bits.label());
    }
    // THE LENGTH, NEVER THE TEXT. A deployment prompt is often the operator's
    // policy or persona and a server log is not where it belongs, but a line
    // saying nothing at all leaves "did my --system-file actually load" with
    // no answer short of sending a request.
    if let Some(system) = &args.default_system {
        eprintln!(
            "  default system prompt: {} chars (applied only to requests that send none)",
            system.chars().count()
        );
    }
    // Parsed and stored (`args.rs`'s `ModelArgs`), but reaching NOTHING past
    // here: none of the three is threaded into `RealChatModel::open` above,
    // so printing the value alone would read as this server acting on it.
    // "accepted, not honoured" is what these flags actually get -- omlx
    // compatibility means the command line does not error, not that the
    // feature exists here.
    if let Some(ref ssd) = args.paged_ssd_cache_dir {
        eprintln!(
            "  paged SSD cache dir: {} (accepted, not honoured)",
            ssd.display()
        );
    }
    if let Some(ref hot) = args.hot_cache_max_size {
        eprintln!("  hot cache max size: {hot} (accepted, not honoured)");
    }
    if let Some(ref mcp) = args.mcp_config {
        eprintln!("  mcp config: {} (accepted, not honoured)", mcp.display());
    }
    Ok(Arc::new(model))
}

#[cfg(not(target_os = "macos"))]
fn open_real_model(_args: &ModelArgs) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    Err("--model needs macOS and a Metal device; only the scripted \
         <tokenizer-dir> mode is available on this platform"
        .to_string())
}

#[cfg(target_os = "macos")]
fn open_real_encoder_model(
    model_arg: &str,
) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    let dir = catalog::resolve_model_arg(model_arg);
    if dir.as_os_str() == model_arg {
        eprintln!("opening embedding model {} ...", model_arg);
    } else {
        eprintln!(
            "opening embedding model {} ({}) ...",
            model_arg,
            dir.display()
        );
    }
    let model =
        turbospark_server::RealEncoderModel::open(&dir)?.with_model_id(model_arg.to_string());
    eprintln!("embedding model open ({model_arg})");
    Ok(Arc::new(model))
}

#[cfg(target_os = "macos")]
fn open_models_registry(
    parsed: &ModelArgs,
) -> Result<Arc<dyn turbospark_server::registry::ModelRegistry>, String> {
    let has_model = !parsed.model.is_empty();

    match (has_model, &parsed.embedding_model) {
        (true, Some(emb_arg)) => {
            let chat_model = open_real_model(parsed)?;
            let emb_model = open_real_encoder_model(emb_arg)?;
            let registry =
                turbospark_server::registry::StaticRegistry::new(vec![chat_model, emb_model])?;
            Ok(Arc::new(registry))
        }
        (true, None) => {
            let dir = catalog::resolve_model_arg(&parsed.model);
            if !dir.join("manifest.json").exists() && dir.join("config.json").exists() {
                if parsed.pool_size > 1 {
                    return Err(
                        "--pool-size pools a GENERATION model; the resolved --model is an                          embedding model (config.json without manifest.json)"
                            .to_string(),
                    );
                }
                let emb_model = open_real_encoder_model(&parsed.model)?;
                Ok(Arc::new(turbospark_server::registry::SingleModel::new(
                    emb_model,
                )))
            } else if parsed.pool_size > 1 {
                // N independent opens of ONE install, behind one public id
                // (ROADMAP P3.6). Each open pays the load guard on its own,
                // so a member that does not fit fails HERE with the guard's
                // own subtraction rather than degrading every request later.
                let n = parsed.pool_size;
                eprintln!("opening pool of {n} runners of {} ...", parsed.model);
                let mut members: Vec<Arc<dyn turbospark_server::ChatModel>> =
                    Vec::with_capacity(n as usize);
                for i in 0..n {
                    if i > 0 {
                        eprintln!("opening pool member {}/{n} ...", i + 1);
                    }
                    let model = open_real_model(parsed).map_err(|e| {
                        format!(
                            "--pool-size {n}: member {} of {n} failed to open: {e}",
                            i + 1
                        )
                    })?;
                    members.push(model);
                }
                use turbospark_server::registry::ModelRegistry as _;
                let registry = turbospark_server::registry::PoolRegistry::new(members)?;
                let row = &registry.rows()[0];
                eprintln!(
                    "pool ready: {} runners behind one id {:?} ({} aliases), requests \
                     route to the least-busy member",
                    n,
                    row.id,
                    row.aliases.len()
                );
                Ok(Arc::new(registry))
            } else {
                let chat_model = open_real_model(parsed)?;
                Ok(Arc::new(turbospark_server::registry::SingleModel::new(
                    chat_model,
                )))
            }
        }
        (false, Some(emb_arg)) => {
            let emb_model = open_real_encoder_model(emb_arg)?;
            Ok(Arc::new(turbospark_server::registry::SingleModel::new(
                emb_model,
            )))
        }
        (false, None) => Err("neither --model nor --embedding-model was provided".to_string()),
    }
}

fn open_scripted(tokenizer_dir: &str) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    let tok = MfTokenizer::load_from_dir(&PathBuf::from(tokenizer_dir))
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    Ok(Arc::new(turbospark_server::ScriptedChatModel::new(
        tok,
        4096,
        Vec::new(),
    )))
}

fn resolve_api_key(
    bind: BindMode,
    flag: Option<String>,
    environment: Option<String>,
) -> Result<Option<String>, String> {
    let key = flag.or(environment).filter(|key| !key.is_empty());
    if bind == BindMode::Tailnet && key.is_none() {
        return Err(
            "--bind tailnet requires --api-key KEY or a non-empty TURBOSPARK_API_KEY".to_string(),
        );
    }
    Ok(key)
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return std::process::ExitCode::from(2);
    }
    if let Some(text) = short_circuit(&args) {
        print!("{text}");
        return std::process::ExitCode::SUCCESS;
    }

    let (registry, port, bind, api_key) = match parse_model_args(&args) {
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::from(2);
        }
        Ok(Some(parsed)) => {
            let api_key = match resolve_api_key(
                parsed.bind,
                parsed.api_key.clone(),
                std::env::var("TURBOSPARK_API_KEY").ok(),
            ) {
                Ok(key) => key,
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(2);
                }
            };
            // Resolve the host BEFORE opening the model: a missing Tailscale
            // should not cost a multi-gigabyte map and a pipeline compile
            // first.
            let host = match parsed.bind.host() {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(2);
                }
            };
            // Applied here, once, rather than inside the parser: `args.rs`
            // reads only `args` (matching `power_profile`'s split), and this
            // is the one call site that can actually act on it, ahead of
            // `open_models_registry`'s catalog resolution below.
            if let Some(endpoint) = &parsed.hf_endpoint {
                std::env::set_var("HF_ENDPOINT", endpoint);
            }
            #[cfg(target_os = "macos")]
            match open_models_registry(&parsed) {
                Ok(reg) => (reg, parsed.port, host, api_key),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(2);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("--model is supported on macOS only; use the scripted mode: turbospark-server <tokenizer-dir> [port]");
                return std::process::ExitCode::from(2);
            }
        }
        // `--api-key` is `--model` mode only, matching `--bind`: the
        // portable scripted mode has no `ModelArgs` to carry it and always
        // binds loopback unauthenticated, exactly as it did before this
        // flag existed.
        Ok(None) => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8080);
            match open_scripted(&args[0]) {
                Ok(m) => (
                    Arc::new(turbospark_server::registry::SingleModel::new(m))
                        as Arc<dyn turbospark_server::registry::ModelRegistry>,
                    port,
                    "127.0.0.1".to_string(),
                    None,
                ),
                Err(e) => {
                    eprintln!("{e}");
                    return std::process::ExitCode::from(1);
                }
            }
        }
    };

    let router = turbospark_server::build_router_with_options(
        registry,
        turbospark_server::RouterOptions {
            api_key: api_key.clone(),
            // The standalone binary records nothing. Events exist for a host
            // that EMBEDS this server and has a console to show them in
            // (`crates/ffi`); this process's equivalent is its stdout, and
            // adding a second, structured channel nothing reads would be
            // overhead per request for no reader.
            observer: None,
        },
    );
    let addr = format!("{bind}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {addr}: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    // What was BOUND, never the string handed to `bind`. `--port` takes a bare
    // `u16`, so `--port 0` is accepted, gets an OS-assigned port, and printed
    // the requested `addr` back as `http://127.0.0.1:0` -- an address no client
    // can use, with the usable one sitting unread in `listener`. Falling back
    // to the request when `local_addr` fails keeps a line on stderr in the case
    // where there is nothing better to say.
    let bound = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| addr.clone());
    eprintln!("turbospark-server listening on http://{bound}");
    eprintln!(
        "  auth: {}",
        if api_key.is_some() {
            "on (x-api-key or Authorization: Bearer required, except /health)"
        } else {
            "off"
        }
    );
    eprintln!("  GET  /health");
    eprintln!("  POST /v1/chat/completions   (OpenAI)");
    eprintln!("  POST /v1/completions        (OpenAI legacy)");
    eprintln!("  POST /v1/responses          (OpenAI)");
    eprintln!("  POST /v1/messages           (Anthropic)");
    eprintln!("  POST /v1/messages/count_tokens  (Anthropic)");
    eprintln!("  GET  /v1/models");
    eprintln!("  GET  /v1/models/:model");
    eprintln!("  POST /v1/embeddings         (OpenAI)");
    eprintln!("  GET  /api/tags              (Ollama)");
    eprintln!("  GET  /api/version           (Ollama)");
    eprintln!("  POST /api/show              (Ollama)");
    eprintln!("  POST /api/chat              (Ollama)");
    eprintln!("  POST /api/generate          (Ollama)");
    eprintln!("  POST /api/embeddings        (Ollama)");
    eprintln!("  POST /api/embed             (Ollama)");
    if let Err(e) = axum::serve(listener, router).await {
        eprintln!("server error: {e}");
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
