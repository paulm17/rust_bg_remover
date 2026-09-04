//! Compatibility re-exports for callers migrating from the M1 CLI loader.
pub use bgremove_core::io::{encode_mask_png, encode_straight_rgba_png, load_canonical};

#[cfg(test)]
mod tests {
    use super::*;
    use bgremove_core::{AlphaMask, Foreground, RgbImageF32};
    use std::path::Path;

    const FIXTURES: &str = "../../tests/fixtures/assets";

    #[test]
    fn production_decoder_accepts_all_m2_codecs_and_orientation() {
        for name in ["codec.png", "codec.jpg", "codec.webp"] {
            let image = load_canonical(Path::new(FIXTURES).join(name).as_path()).unwrap();
            assert_eq!(image.dimensions(), (3, 2), "{name}");
            assert!(image
                .source_alpha()
                .data()
                .iter()
                .all(|a| (*a - 1.0).abs() < f32::EPSILON));
        }
        let oriented =
            load_canonical(Path::new(FIXTURES).join("orientation-6.jpg").as_path()).unwrap();
        assert_eq!(oriented.dimensions(), (2, 3));
        let px = oriented.rgb().data();
        let corners = [px[0], px[1], px[4], px[5]];
        assert!(
            corners[0][1] > corners[0][0] * 0.7 && corners[0][1] > corners[0][2] * 0.7,
            "expected yellow top-left: {:?}",
            corners[0]
        );
        assert!(
            corners[1][0] > corners[1][1] * 2.0 && corners[1][0] > corners[1][2] * 2.0,
            "expected red top-right: {:?}",
            corners[1]
        );
        assert!(
            corners[2][1] > corners[2][0] * 2.0 && corners[2][2] > corners[2][0] * 2.0,
            "expected cyan bottom-left: {:?}",
            corners[2]
        );
        assert!(
            corners[3][2] > corners[3][0] * 2.0 && corners[3][2] > corners[3][1] * 2.0,
            "expected blue bottom-right: {:?}",
            corners[3]
        );
    }

    #[test]
    fn production_decoder_preserves_transparent_source_alpha_and_rejects_gif() {
        let image = load_canonical(Path::new(FIXTURES).join("transparent.png").as_path()).unwrap();
        assert_eq!(image.dimensions(), (2, 1));
        assert_eq!(image.source_alpha().data(), &[0.0, 128.0 / 255.0]);
        let error =
            load_canonical(Path::new(FIXTURES).join("unsupported.gif").as_path()).unwrap_err();
        assert!(error.to_string().contains("decode input image"));
        assert!(format!("{error:#}").contains("not supported"));
    }

    #[test]
    fn straight_alpha_and_mask_png_round_trip_without_premultiplication() {
        let rgb = RgbImageF32::new(2, 1, vec![[1.0, 0.25, 0.0], [0.1, 0.2, 0.3]]).unwrap();
        let alpha = AlphaMask::new(2, 1, vec![0.0, 0.5]).unwrap();
        let foreground = Foreground::new(rgb, alpha.clone()).unwrap();
        let decoded = image::load_from_memory(&encode_straight_rgba_png(&foreground).unwrap())
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 64, 0, 0]);
        assert_eq!(decoded.get_pixel(1, 0).0, [26, 51, 77, 128]);
        let mask = image::load_from_memory(&encode_mask_png(&alpha).unwrap())
            .unwrap()
            .to_luma8();
        assert_eq!(mask.as_raw(), &[0, 128]);
    }
}
