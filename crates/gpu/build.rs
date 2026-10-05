//! Compile the same shader compositions the dispatch modules submit at runtime.
//! The byte blobs travel inside the Rust archive, including Swift/FFI packaging.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Expr, ItemConst, ItemMod, ItemStatic, Lit, Token};

const FLAGS: &[&str] = &["-std=metal3.1", "-mmacosx-version-min=14.0", "-ffast-math"];

struct Shader {
    label: String,
    source: String,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn expand(expr: &Expr, parent: &Path) -> Result<String, String> {
    match expr {
        Expr::Lit(value) => match &value.lit {
            Lit::Str(value) => Ok(value.value()),
            _ => Err("shader composition contains a non-string literal".into()),
        },
        Expr::Macro(value) if value.mac.path.is_ident("include_str") => {
            let literal = syn::parse2::<syn::LitStr>(value.mac.tokens.clone())
                .map_err(|err| format!("unsupported include_str: {err}"))?;
            let path = parent.join(literal.value());
            println!("cargo:rerun-if-changed={}", path.display());
            fs::read_to_string(&path).map_err(|err| format!("{}: {err}", path.display()))
        }
        Expr::Macro(value) if value.mac.path.is_ident("concat") => {
            Punctuated::<Expr, Token![,]>::parse_terminated
                .parse2(value.mac.tokens.clone())
                .map_err(|err| format!("unsupported concat: {err}"))?
                .iter()
                .map(|item| expand(item, parent))
                .collect()
        }
        _ => Err("shader composition must use string literals, include_str!, or concat!".into()),
    }
}

struct Inventory<'a> {
    path: &'a Path,
    module_dir: &'a Path,
    shaders: &'a mut Vec<Shader>,
}

fn test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| matches!(meta, syn::Meta::Path(path) if path.is_ident("test")))
    })
}

impl<'ast> Visit<'ast> for Inventory<'_> {
    fn visit_item_static(&mut self, item: &'ast ItemStatic) {
        if test_only(&item.attrs) {
            return;
        }
        let name = item.ident.to_string();
        if name == "SOURCE" || name.ends_with("_SOURCE") {
            let source = expand(&item.expr, self.path.parent().unwrap())
                .unwrap_or_else(|err| panic!("{}::{name}: {err}", self.path.display()));
            self.shaders.push(Shader {
                label: format!("{}::{name}", self.path.display()),
                source,
            });
        }
        syn::visit::visit_item_static(self, item);
    }

    fn visit_item_const(&mut self, item: &'ast ItemConst) {
        if test_only(&item.attrs) {
            return;
        }
        let name = item.ident.to_string();
        if name == "SOURCE" || name.ends_with("_SOURCE") {
            let source = expand(&item.expr, self.path.parent().unwrap())
                .unwrap_or_else(|err| panic!("{}::{name}: {err}", self.path.display()));
            self.shaders.push(Shader {
                label: format!("{}::{name}", self.path.display()),
                source,
            });
        }
        syn::visit::visit_item_const(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        if test_only(&item.attrs) {
            return;
        }
        let module_dir = self.module_dir.join(item.ident.to_string());
        if let Some((_, items)) = &item.content {
            let mut nested = Inventory {
                path: self.path,
                module_dir: &module_dir,
                shaders: self.shaders,
            };
            for child in items {
                nested.visit_item(child);
            }
            return;
        }
        let explicit_path = item.attrs.iter().find_map(|attribute| {
            if !attribute.path().is_ident("path") {
                return None;
            }
            match &attribute.meta {
                syn::Meta::NameValue(value) => match &value.value {
                    Expr::Lit(value) => match &value.lit {
                        Lit::Str(value) => Some(self.path.parent().unwrap().join(value.value())),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
        });
        let file = explicit_path.unwrap_or_else(|| {
            let file = module_dir.with_extension("rs");
            if file.exists() {
                file
            } else {
                module_dir.join("mod.rs")
            }
        });
        collect(&file, &module_dir, self.shaders);
    }
}

fn collect(path: &Path, module_dir: &Path, shaders: &mut Vec<Shader>) {
    println!("cargo:rerun-if-changed={}", path.display());
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let file = syn::parse_file(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    Inventory {
        path,
        module_dir,
        shaders,
    }
    .visit_file(&file);
}

fn tool(args: &[&str]) -> Result<String, String> {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx"])
        .args(args)
        .output()
        .map_err(|err| format!("xcrun: {err}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Ok(format!("{stdout}{stderr}"))
    } else {
        Err(format!("xcrun {}: {stdout}{stderr}", args.join(" ")))
    }
}

fn compile(source: &Path, library: &Path, precise: bool) -> Result<(), String> {
    let air = source.with_extension("air");
    let temporary = library.with_extension("metallib.tmp");
    let mut args = vec!["metal"];
    args.extend(
        FLAGS
            .iter()
            .copied()
            .filter(|flag| !precise || *flag != "-ffast-math"),
    );
    if precise {
        args.push("-fno-fast-math");
    }
    args.extend(["-c", source.to_str().unwrap(), "-o", air.to_str().unwrap()]);
    tool(&args)?;
    let result = tool(&[
        "metallib",
        air.to_str().unwrap(),
        "-o",
        temporary.to_str().unwrap(),
    ])
    .and_then(|_| fs::rename(&temporary, library).map_err(|err| err.to_string()));
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn compiler_metadata() -> Result<String, String> {
    let commands: &[&[&str]] = &[
        &["metal", "--version"],
        &["metallib", "--version"],
        &["--show-sdk-version"],
        &["--show-sdk-build-version"],
    ];
    let metadata: Result<Vec<_>, _> = commands.iter().map(|args| tool(args)).collect();
    metadata.map(|values| format!("{values:?}"))
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    for name in [
        "DEVELOPER_DIR",
        "SDKROOT",
        "TURBOSPARK_METAL_PRECOMPILE_STRICT",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let registry_path = out.join("precompiled_registry.rs");
    // Cross-target builds must not invoke a host Metal compiler or embed its IR.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        fs::write(registry_path, "").unwrap();
        return;
    }
    let strict = env::var("TURBOSPARK_METAL_PRECOMPILE_STRICT").as_deref() == Ok("1");
    let mut shaders = Vec::new();
    collect(Path::new("src/lib.rs"), Path::new("src"), &mut shaders);
    let unique: BTreeMap<_, _> = shaders
        .into_iter()
        .map(|shader| (digest(shader.source.as_bytes()), shader))
        .collect();
    let compiler = compiler_metadata();
    if strict {
        assert!(compiler.is_ok(), "Metal precompile tools: {compiler:?}");
    }
    if let Err(err) = &compiler {
        println!(
            "cargo:warning=Metal precompile tools unavailable, using runtime source fallback: {}",
            err.replace('\n', " ")
        );
    }
    let compiler_key = digest(format!("{compiler:?}{FLAGS:?}").as_bytes());
    let mut registry = String::from("static LIBRARIES: &[BundledLibrary] = &[\n");
    let mut inventory = String::from("#[cfg(test)] static INVENTORY: &[(&str, &str)] = &[\n");
    let mut built = 0;
    let mut total_bytes = 0;
    for (hash, shader) in &unique {
        let stem = format!("{hash}-{compiler_key}");
        let source_path = out.join(format!("{stem}.metal"));
        let library_path = out.join(format!("{stem}.metallib"));
        fs::write(&source_path, &shader.source).unwrap();
        inventory.push_str(&format!(
            "({:?}, include_str!({:?})),\n",
            shader.label,
            source_path.to_str().unwrap()
        ));
        let result = match &compiler {
            Err(err) => Err(err.clone()),
            Ok(_) if library_path.exists() => Ok(()),
            Ok(_) => compile(
                &source_path,
                &library_path,
                shader.source.starts_with("// turbospark: precise-math\n"),
            ),
        };
        match result {
            Ok(()) => {
                let bytes = fs::read(&library_path).unwrap();
                total_bytes += bytes.len();
                built += 1;
                // Archive keys must identify the actual packaged IR, not only
                // the runtime source or a fixed language/options policy label.
                let identity = format!(
                    "packaged-v2|toolchain={compiler_key}|metallib={}",
                    digest(&bytes)
                );
                registry.push_str(&format!(
                    "BundledLibrary {{ hash: {hash:?}, source_len: {}, compiler_identity: {identity:?}, bytes: include_bytes!({:?}) }},\n",
                    shader.source.len(),
                    library_path.to_str().unwrap()
                ));
            }
            Err(err) => {
                // A failed partial output must never be reused on a later build.
                let _ = fs::remove_file(&library_path);
                assert!(!strict, "Metal precompile {} failed: {err}", shader.label);
                if compiler.is_ok() {
                    println!(
                        "cargo:warning=Metal precompile {} unavailable, using runtime source fallback: {}",
                        shader.label,
                        err.replace('\n', " ")
                    );
                }
            }
        }
    }
    registry.push_str("] ;\n");
    inventory.push_str("] ;\n");
    registry.push_str(&inventory);
    registry.push_str(&format!(
        "#[cfg(test)] const BUILD_COMPLETE: bool = {};\n",
        built == unique.len()
    ));
    fs::write(registry_path, registry).unwrap();
    println!(
        "cargo:warning=Bundled Metal libraries: {built}/{} shader compositions, {total_bytes} bytes (Metal 3.1, macOS 14)",
        unique.len()
    );
}
