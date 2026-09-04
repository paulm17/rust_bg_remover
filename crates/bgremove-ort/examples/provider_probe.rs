use bgremove_models::parse_toml;
use bgremove_ort::{RequestedProvider, VerifiedSession};
use std::{fs, path::Path};
fn main() -> anyhow::Result<()> {
    let path = Path::new("models/m3_identity.toml");
    let manifest = parse_toml(&fs::read_to_string(path)?)?;
    let runtime =
        std::env::var_os("ORT_DYLIB").ok_or_else(|| anyhow::anyhow!("ORT_DYLIB is required"))?;
    let strict = std::env::var_os("STRICT_PROVIDER").is_some();
    let session = VerifiedSession::open(
        &manifest,
        path,
        Path::new(&runtime),
        RequestedProvider::Coreml,
        !strict,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&session.inspection.provider)?
    );
    Ok(())
}
