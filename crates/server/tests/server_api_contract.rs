//! Contract tests verifying the OpenAPI specification matches server routes.

#[test]
fn turbospark_openapi_lists_all_server_routes() {
    let yaml = include_str!("../../../docs/openapi/turbospark.openapi.yaml");
    let re = regex_lite::Regex::new(r"(?m)^  (/[^:\s]+):\s*$").unwrap();
    let mut documented: Vec<String> = re.captures_iter(yaml).map(|c| c[1].to_string()).collect();
    let mut routed: Vec<String> = turbospark_server::SERVER_ROUTE_PATHS
        .iter()
        .map(|p| p.to_string())
        .collect();
    documented.sort();
    routed.sort();
    assert_eq!(
        documented, routed,
        "OpenAPI spec paths must match SERVER_ROUTE_PATHS"
    );
}

#[test]
fn turbospark_openapi_yaml_alias_matches() {
    let canonical = include_str!("../../../docs/openapi/turbospark.openapi.yaml");
    let alias = include_str!("../../../docs/openapi/openapi.yaml");
    assert_eq!(
        canonical, alias,
        "openapi.yaml must be identical to turbospark.openapi.yaml"
    );
}
