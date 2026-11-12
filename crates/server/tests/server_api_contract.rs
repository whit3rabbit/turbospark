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

#[test]
fn gitbook_openapi_specs_match_docs_openapi_specs() {
    assert_eq!(
        include_str!("../../../docs/openapi/turbospark.openapi.yaml"),
        include_str!("../../../api-reference/openapi/turbospark.openapi.yaml"),
        "gitbook turbospark.openapi.yaml must match docs/"
    );
    assert_eq!(
        include_str!("../../../docs/openapi/openapi.yaml"),
        include_str!("../../../api-reference/openapi/openapi.yaml"),
        "gitbook openapi.yaml must match docs/"
    );
    assert_eq!(
        include_str!("../../../docs/openapi/text.openapi.yaml"),
        include_str!("../../../api-reference/openapi/text.openapi.yaml"),
        "gitbook text.openapi.yaml must match docs/"
    );
    assert_eq!(
        include_str!("../../../docs/openapi/audio.openapi.yaml"),
        include_str!("../../../api-reference/openapi/audio.openapi.yaml"),
        "gitbook audio.openapi.yaml must match docs/"
    );
    assert_eq!(
        include_str!("../../../docs/openapi/images.openapi.yaml"),
        include_str!("../../../api-reference/openapi/images.openapi.yaml"),
        "gitbook images.openapi.yaml must match docs/"
    );
    assert_eq!(
        include_str!("../../../docs/openapi/embeddings.openapi.yaml"),
        include_str!("../../../api-reference/openapi/embeddings.openapi.yaml"),
        "gitbook embeddings.openapi.yaml must match docs/"
    );
}
