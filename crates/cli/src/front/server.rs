//! `turbospark status` and `turbospark ps`: the background server, from the
//! daemon's own pid and meta files plus a `/health` probe.
//!
//! Only the managed daemon (`turbospark start`) is visible here. A
//! foreground `turbospark serve` writes no pid file, so it does not show.

use std::process::ExitCode;

use serde_json::json;

use super::{agent, daemon};
use crate::ui::{self, Cell, CellStyle, Paint, Table};

struct Snapshot {
    pid: i32,
    port: u16,
    model: Option<String>,
    healthy: bool,
}

fn snapshot() -> Option<Snapshot> {
    let pid = daemon::get_running_pid()?;
    let server = daemon::running_server()?;
    Some(Snapshot {
        pid,
        port: server.port,
        model: server.model,
        healthy: agent::http_health_ok(server.port),
    })
}

fn endpoints(port: u16) -> [(&'static str, String); 3] {
    let base = format!("http://127.0.0.1:{port}");
    [
        ("OpenAI", format!("{base}/v1")),
        ("Anthropic", base.clone()),
        ("Ollama", format!("{base}/api")),
    ]
}

pub fn status(json: bool) -> ExitCode {
    let snap = snapshot();
    if json {
        let value = match &snap {
            Some(s) => json!({
                "running": true,
                "healthy": s.healthy,
                "pid": s.pid,
                "port": s.port,
                "model": s.model,
                "endpoints": endpoints(s.port)
                    .iter()
                    .map(|(name, url)| json!({ "api": name, "url": url }))
                    .collect::<Vec<_>>(),
                "log": daemon::log_file(),
            }),
            None => json!({ "running": false }),
        };
        println!("{value:#}");
        return ExitCode::SUCCESS;
    }
    let paint = Paint::out();
    println!();
    let Some(s) = snap else {
        println!(
            "  {}  {}",
            paint.bold("turbospark server"),
            paint.dim("stopped")
        );
        println!();
        println!(
            "  {}  {}",
            paint.bold("turbospark start --model NAME"),
            paint.dim("run it in the background")
        );
        println!(
            "  {}  {}",
            paint.bold("turbospark serve --model NAME"),
            paint.dim("run it in this terminal")
        );
        return ExitCode::SUCCESS;
    };
    let state = if s.healthy {
        paint.accent("running")
    } else {
        paint.warn("starting (no answer on /health yet)")
    };
    println!("  {}  {state}", paint.bold("turbospark server"));
    println!();
    let field = |label: &str, value: String| println!("  {:<10} {value}", paint.dim(label));
    field("model", s.model.clone().unwrap_or_else(|| "-".to_string()));
    for (name, url) in endpoints(s.port) {
        field(name, paint.link(&url));
    }
    field("pid", s.pid.to_string());
    field("log", daemon::log_file().display().to_string());
    println!();
    println!(
        "  {}",
        paint.dim("turbospark stop ends it. turbospark launch claude points an agent at it.")
    );
    ExitCode::SUCCESS
}

pub fn ps(json: bool) -> ExitCode {
    let snap = snapshot();
    if json {
        let rows: Vec<_> = snap
            .iter()
            .map(
                |s| json!({ "model": s.model, "pid": s.pid, "port": s.port, "healthy": s.healthy }),
            )
            .collect();
        println!("{:#}", serde_json::Value::Array(rows));
        return ExitCode::SUCCESS;
    }
    let paint = Paint::out();
    let Some(s) = snap else {
        println!(
            "  {}",
            paint.dim("No model is being served. `turbospark start --model NAME` runs one.")
        );
        return ExitCode::SUCCESS;
    };
    let mut table = Table::new(&["MODEL", "PORT", "PID", "STATE"])
        .right(1)
        .right(2);
    table.row(vec![
        Cell::styled(s.model.unwrap_or_else(|| "-".to_string()), CellStyle::Bold),
        Cell::new(s.port.to_string()),
        Cell::new(s.pid.to_string()),
        if s.healthy {
            Cell::styled("running", CellStyle::Accent)
        } else {
            Cell::styled("starting", CellStyle::Warn)
        },
    ]);
    print!("{}", table.render(paint, ui::columns()));
    ExitCode::SUCCESS
}
