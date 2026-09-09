use bgremove_core::{CanonicalImage, Prompt, PromptPoint};
use bgremove_models::parse_toml;
use bgremove_ort::sam::{
    SamEncoding, SamPromptRequest, SamSegmenter, SamSelectionPolicy, SamVariant,
};
use bgremove_ort::RequestedProvider;
use std::{fs, path::Path};

fn main() -> anyhow::Result<()> {
    let encoder_path = Path::new("models/m14_sam_synthetic_encoder.toml");
    let decoder_path = Path::new("models/m14_sam_synthetic_decoder.toml");
    let encoder = parse_toml(&fs::read_to_string(encoder_path)?)?;
    let decoder = parse_toml(&fs::read_to_string(decoder_path)?)?;
    let runtime =
        std::env::var_os("ORT_DYLIB").ok_or_else(|| anyhow::anyhow!("ORT_DYLIB is required"))?;
    let mut segmenter = SamSegmenter::new(
        &encoder,
        encoder_path,
        &decoder,
        decoder_path,
        Path::new(&runtime),
        SamVariant::VitB,
        SamEncoding::Fp32,
        RequestedProvider::Cpu,
        false,
        SamSelectionPolicy::HighestQuality,
    )?;
    let image = CanonicalImage::new(5, 3, vec![[0.2, 0.4, 0.6]; 15])?;
    let prompt =
        SamPromptRequest::new(Prompt::new(vec![PromptPoint::new(2.0, 1.0, true)?], None)?)?;
    for run in 0..2 {
        let evidence = segmenter.predict_with_evidence(&image, &prompt)?;
        println!(
            "run={run} candidates={} selected={:?}",
            evidence.candidates.len(),
            evidence.selection.selected_index
        );
    }
    Ok(())
}
