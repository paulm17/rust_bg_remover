use bgremove_models::parse_toml;
use bgremove_ort::{RequestedProvider, VerifiedSession};
use std::{fs, path::Path};
fn main() -> anyhow::Result<()> {
    let manifest_path = Path::new("models/m3_identity.toml");
    let manifest = parse_toml(&fs::read_to_string(manifest_path)?)?;
    let runtime = std::env::var_os("ORT_DYLIB").expect("ORT_DYLIB");
    let mut session = VerifiedSession::open(
        &manifest,
        manifest_path,
        Path::new(&runtime),
        RequestedProvider::Cpu,
        false,
    )?;
    for shape in [[1, 3, 2, 2], [1, 3, 3, 5]] {
        let n = shape.iter().product::<i64>() as usize;
        let values = (0..n).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let out = session.run(&shape, &values)?;
        assert_eq!(out.shape, shape);
        assert_eq!(out.values, values);
    }
    println!("{}", serde_json::to_string_pretty(&session.inspection)?);
    Ok(())
}
