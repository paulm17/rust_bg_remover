"""Generate the pinned development-only rembg/Pillow compatibility fixture."""
from PIL import Image
import json
from pathlib import Path

source_width, source_height = 3, 2
source_rgb = bytes(
    value
    for pixel in range(source_width * source_height)
    for value in (
        (pixel * 71 + 9) % 256,
        (pixel * 43 + 37) % 256,
        (255 - pixel * 29) % 256,
    )
)
resize_width, resize_height = 17, 11
source = Image.frombytes("RGB", (source_width, source_height), source_rgb)
resized = source.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
resized_rgb = resized.tobytes()
global_max = max(resized_rgb)
normalized = [value / max(global_max, 1e-6) - 0.5 for value in resized_rgb]

raw = [((index * 37) % 1000) / 1000.0 for index in range(resize_width * resize_height)]
raw_min, raw_max = min(raw), max(raw)
mask_u8 = [int((value - raw_min) / (raw_max - raw_min) * 255.0) for value in raw]
mask = Image.frombytes("L", (resize_width, resize_height), bytes(mask_u8))
restored = mask.resize((source_width, source_height), Image.Resampling.LANCZOS)

fixture = {
    "schema": "m4.rembg-pillow.v1",
    "pillow_version": __import__("PIL").__version__,
    "source_dimensions": [source_width, source_height],
    "source_rgb": list(source_rgb),
    "resize_dimensions": [resize_width, resize_height],
    "resized_rgb": list(resized_rgb),
    "global_max": global_max,
    "normalized": normalized,
    "raw_mask": raw,
    "raw_min": raw_min,
    "raw_max": raw_max,
    "mask_u8": mask_u8,
    "restored_u8": list(restored.tobytes()),
    # image::Lanczos3 and Pillow LANCZOS are different implementations. These
    # are measured compatibility bounds for this fixed edge/ramp fixture.
    "tolerances": {
        "resize_u8_max_abs": 5,
        "resize_u8_mean_abs": 0.4,
        "normalized_f32_max_abs": 0.019608,
        "normalized_f32_mean_abs": 0.001245,
        "restore_u8_max_abs": 5,
        "restore_u8_mean_abs": 1.0,
    },
}
Path(__file__).with_name("rembg-pillow-fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
