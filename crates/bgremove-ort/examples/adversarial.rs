//! M3 gate verifier: every manifest contract mutation must fail before a run.
use bgremove_models::{parse_toml, DimensionSpec, TensorElementType};
use bgremove_ort::{RequestedProvider, VerifiedSession};
use std::{fs, path::Path};
fn main() -> anyhow::Result<()> {
    let path = Path::new("models/m3_identity.toml");
    let original = parse_toml(&fs::read_to_string(path)?)?;
    let runtime =
        std::env::var_os("ORT_DYLIB").ok_or_else(|| anyhow::anyhow!("ORT_DYLIB is required"))?;
    type Mutation = Box<dyn Fn(&mut bgremove_models::ModelManifest)>;
    let cases: Vec<(&str, Mutation)> = vec![
        ("input-name", Box::new(|m| m.input_name = "wrong".into())),
        ("output-name", Box::new(|m| m.output_name = "wrong".into())),
        ("output-index", Box::new(|m| m.output_index = Some(1))),
        (
            "input-type",
            Box::new(|m| m.input_type = Some(TensorElementType::I32)),
        ),
        (
            "input-rank",
            Box::new(|m| m.input_shape = vec![DimensionSpec::Dynamic("batch".into())]),
        ),
        (
            "output-rank",
            Box::new(|m| m.output_shape = vec![DimensionSpec::Dynamic("x".into())]),
        ),
        ("opset", Box::new(|m| m.opset = 12)),
    ];
    for (name, mutate) in cases {
        let mut manifest = original.clone();
        mutate(&mut manifest);
        let error = VerifiedSession::open(
            &manifest,
            path,
            Path::new(&runtime),
            RequestedProvider::Cpu,
            false,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("{name} mutation unexpectedly passed"))?;
        println!("{name}: {error:#}");
    }
    let mut manifest = original.clone();
    manifest.sha256 = "0".repeat(64);
    assert!(VerifiedSession::open(
        &manifest,
        path,
        Path::new("/missing"),
        RequestedProvider::Cpu,
        false
    )
    .is_err());
    println!("all adversarial M3 pre-run checks failed closed");
    Ok(())
}
