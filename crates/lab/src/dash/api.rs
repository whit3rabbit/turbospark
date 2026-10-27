//! The dashboard's JSON API: the same data the pages show, as `tslab/1`
//! envelopes an agent or a dashboard reads. Every route is a GET and
//! filters are query parameters.

use std::collections::HashMap;

use axum::extract::{Path as AxPath, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use super::pages::Params;
use super::Shared;
use crate::schema::SCHEMA;

fn err(code: StatusCode, what: &str) -> (StatusCode, Json<Value>) {
    (
        code,
        Json(json!({"schema": SCHEMA, "ok": false, "error": what})),
    )
}

type Resp = Result<Json<Value>, (StatusCode, Json<Value>)>;

pub async fn experiments(State(app): State<Shared>) -> Resp {
    Ok(Json(app.with_index(|ix| {
        json!({"schema": SCHEMA, "lab": ix.lab.display().to_string(),
               "experiments": ix.summaries().iter().map(|s| serde_json::to_value(s).unwrap_or(Value::Null)).collect::<Vec<_>>(),
               "totals": super::pages::kind_totals(ix)})
    })))
}

pub async fn experiment(State(app): State<Shared>, AxPath(name): AxPath<String>) -> Resp {
    crate::paths::validate_name(&name).map_err(|e| err(StatusCode::BAD_REQUEST, &e))?;
    app.with_index(|ix| {
        let e = ix
            .experiments
            .get(&name)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such experiment"))?;
        Ok(Json(
            json!({"schema": SCHEMA, "experiment": e.summary, "notes": e.notes,
            "verdicts": ix.verdicts.iter().filter(|v| v.experiment == name).collect::<Vec<_>>(),
            "sweeps": ix.sweeps.iter().filter(|v| v.experiment == name).collect::<Vec<_>>(),
            "profiles": ix.profiles.iter().filter(|v| v.experiment == name).collect::<Vec<_>>()}),
        ))
    })
}

pub async fn ledger(State(app): State<Shared>, AxPath(name): AxPath<String>) -> Resp {
    crate::paths::validate_name(&name).map_err(|e| err(StatusCode::BAD_REQUEST, &e))?;
    app.with_index(|ix| {
        let e = ix
            .experiments
            .get(&name)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such experiment"))?;
        Ok(Json(
            json!({"schema": SCHEMA, "experiment": name, "records": e.records}),
        ))
    })
}

pub async fn series(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Resp {
    let p = Params::from_query(&q);
    let names = p.experiments();
    if names.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "exp=a,b names the experiments",
        ));
    }
    Ok(Json(app.with_index(|ix| {
        let mut series = Vec::new();
        for n in &names {
            for s in ix.series_of(n) {
                if p.kind
                    .as_ref()
                    .is_none_or(|k| *k == s.kind || *k == s.source_kind)
                {
                    series.push(s);
                }
            }
        }
        json!({"schema": SCHEMA, "experiments": names, "series": series})
    })))
}

pub async fn verdicts(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Resp {
    let p = Params::from_query(&q);
    Ok(Json(app.with_index(|ix| {
        let rows: Vec<_> = ix
            .verdicts
            .iter()
            .filter(|v| {
                p.family
                    .as_ref()
                    .is_none_or(|f| v.family.as_deref() == Some(f))
                    && p.model
                        .as_ref()
                        .is_none_or(|m| v.model.as_deref() == Some(m))
                    && p.metric.as_ref().is_none_or(|m| v.metric == *m)
                    && p.exp
                        .as_ref()
                        .is_none_or(|e| e.split(',').any(|x| x == v.experiment))
            })
            .collect();
        json!({"schema": SCHEMA, "verdicts": rows})
    })))
}

pub async fn sweeps(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Resp {
    let p = Params::from_query(&q);
    Ok(Json(app.with_index(|ix| {
        let rows: Vec<_> = ix
            .sweeps
            .iter()
            .filter(|s| {
                p.family
                    .as_ref()
                    .is_none_or(|f| s.family.as_deref() == Some(f))
                    && p.model
                        .as_ref()
                        .is_none_or(|m| s.model.as_deref() == Some(m))
            })
            .collect();
        json!({"schema": SCHEMA, "sweeps": rows})
    })))
}

pub async fn family(State(app): State<Shared>, AxPath(family): AxPath<String>) -> Resp {
    let set = crate::family_profile::ProfileSet::embedded()
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e))?;
    let pr = set
        .by_family(&family)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such family profile"))?;
    Ok(Json(app.with_index(|ix| {
        json!({"schema": SCHEMA, "profile": pr,
            "verdicts": ix.verdicts.iter().filter(|v| v.family.as_deref() == Some(family.as_str())).collect::<Vec<_>>(),
            "sweeps": ix.sweeps.iter().filter(|v| v.family.as_deref() == Some(family.as_str())).collect::<Vec<_>>(),
            "profiles": ix.profiles.iter().filter(|v| v.family.as_deref() == Some(family.as_str())).collect::<Vec<_>>()})
    })))
}

pub async fn record(
    State(app): State<Shared>,
    AxPath((name, id)): AxPath<(String, String)>,
) -> Resp {
    crate::paths::validate_name(&name).map_err(|e| err(StatusCode::BAD_REQUEST, &e))?;
    if !crate::family_profile::is_record_id(&id) {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "record ids are <unix_ms>-<pid>-<n>",
        ));
    }
    app.with_index(|ix| {
        let rec = ix
            .record(&name, &id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "no such record"))?;
        Ok(Json(
            json!({"schema": SCHEMA, "experiment": name, "record": rec}),
        ))
    })
}
