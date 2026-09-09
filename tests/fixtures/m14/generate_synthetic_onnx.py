#!/usr/bin/env python3
"""Build compact deterministic SAM-shaped ONNX graphs without dependencies.

The graphs connect every named input and construct bounded outputs with ONNX
operators; they are runtime wiring/safety fixtures, not real SAM weights.
"""
from __future__ import annotations

import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "models/fixtures"


def varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def field(number: int, wire: int, value: bytes | int) -> bytes:
    if wire == 0:
        return varint((number << 3) | wire) + varint(int(value))
    value = bytes(value)
    return varint((number << 3) | wire) + varint(len(value)) + value


def string(number: int, value: str) -> bytes:
    return field(number, 2, value.encode())


def dim(value: int | str) -> bytes:
    return field(1, 0, value) if isinstance(value, int) else string(2, value)


def value_info(name: str, shape: list[int | str]) -> bytes:
    tensor_shape = b"".join(field(1, 2, dim(item)) for item in shape)
    tensor_type = field(1, 0, 1) + field(2, 2, tensor_shape)
    return string(1, name) + field(2, 2, field(1, 2, tensor_type))


def tensor(name: str, shape: list[int], values: list[float]) -> bytes:
    raw = struct.pack("<" + "f" * len(values), *values)
    body = b"".join(field(1, 0, item) for item in shape)
    body += field(2, 0, 1) + string(8, name) + field(9, 2, raw)
    return body


def int64_tensor(name: str, shape: list[int], values: list[int]) -> bytes:
    body = b"".join(field(1, 0, item) for item in shape)
    body += field(2, 0, 7) + b"".join(field(7, 0, item) for item in values) + string(8, name)
    return body


def constant_node(name: str, output: str, shape: list[int], values: list[float]) -> bytes:
    attribute = string(1, "value") + field(5, 2, tensor(name + "_tensor", shape, values)) + field(20, 0, 4)
    return string(2, output) + string(3, name) + string(4, "Constant") + field(5, 2, attribute)


def int64_constant_node(name: str, output: str, values: list[int]) -> bytes:
    attribute = string(1, "value") + field(5, 2, int64_tensor(name + "_tensor", [len(values)], values)) + field(20, 0, 4)
    return string(2, output) + string(3, name) + string(4, "Constant") + field(5, 2, attribute)


def constant_of_shape_node(name: str, shape_input: str, output: str, value: float = 0.0) -> bytes:
    attribute = string(1, "value") + field(5, 2, tensor(name + "_tensor", [1], [value])) + field(20, 0, 4)
    return string(1, shape_input) + string(2, output) + string(3, name) + string(4, "ConstantOfShape") + field(5, 2, attribute)


def reduce_mean_node(name: str, input_name: str, output: str, axes: list[int]) -> bytes:
    axes_name = name + "_axes"
    # Keepdims=0 gives a true scalar.  Scalar broadcasting is valid for every
    # synthetic output rank and avoids depending on dynamic prompt dimensions
    # during ORT shape inference.
    keepdims = string(1, "keepdims") + field(3, 0, 0) + field(20, 0, 2)
    return string(1, input_name) + string(1, axes_name) + string(2, output) + string(3, name) + string(4, "ReduceMean") + field(5, 2, keepdims)


def add_node(name: str, left: str, right: str, output: str) -> bytes:
    return string(1, left) + string(1, right) + string(2, output) + string(3, name) + string(4, "Add")


def reshape_node(name: str, input_name: str, shape_name: str, output: str) -> bytes:
    return string(1, input_name) + string(1, shape_name) + string(2, output) + string(3, name) + string(4, "Reshape")


def range_node(name: str, start: str, limit: str, delta: str, output: str) -> bytes:
    return string(1, start) + string(1, limit) + string(1, delta) + string(2, output) + string(3, name) + string(4, "Range")


def tile_node(name: str, input_name: str, repeats: str, output: str) -> bytes:
    return string(1, input_name) + string(1, repeats) + string(2, output) + string(3, name) + string(4, "Tile")


def graph(name: str, inputs: list[tuple[str, list[int | str]]], outputs: list[tuple[str, list[int | str]]], nodes: list[bytes]) -> bytes:
    body = b"".join(field(1, 2, node) for node in nodes) + string(2, name)
    body += b"".join(field(11, 2, value_info(input_name, shape)) for input_name, shape in inputs)
    body += b"".join(field(12, 2, value_info(output_name, shape)) for output_name, shape in outputs)
    return body


def model(graph_bytes: bytes) -> bytes:
    # Opset 18 is the first ONNX version where ReduceMean accepts axes as its
    # second tensor input.  Keeping the axes tensors explicit makes every
    # synthetic input graph-valid while exercising the same named-input path
    # as the real decoder.
    opset = string(1, "ai.onnx") + field(2, 0, 18)
    return field(1, 0, 8) + string(2, "bgremove-m14-synthetic") + field(7, 2, graph_bytes) + field(8, 2, opset)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = graph(
        "sam_synthetic_encoder",
        [("input_image", [684, 1024, 3])],
        [("image_embeddings", [1, 256, 64, 64])],
        [int64_constant_node("embedding_shape_constant", "embedding_shape", [1, 256, 64, 64]), constant_of_shape_node("embedding_constant", "embedding_shape", "embedding_base"), int64_constant_node("encoder_reduce_axes_constant", "encoder_reduce_axes", [0, 1, 2]), reduce_mean_node("encoder_reduce", "input_image", "encoder_scalar", [0, 1, 2]), add_node("encoder_add", "embedding_base", "encoder_scalar", "image_embeddings")],
    )
    plane = 684 * 1024
    delta = 2.0 / (plane - 1)
    decoder = graph(
        "sam_synthetic_decoder",
        [("image_embeddings", [1, 256, 64, 64]), ("point_coords", [1, "num_points", 2]), ("point_labels", [1, "num_points"]), ("mask_input", [1, 1, 256, 256]), ("has_mask_input", [1]), ("orig_im_size", [2])],
        [("masks", [1, 3, 684, 1024]), ("iou_predictions", [1, 3]), ("low_res_masks", [1, 3, 256, 256])],
        [int64_constant_node("mask_shape_constant", "mask_shape", [1, 1, 684, 1024]), int64_constant_node("tile_repeats_constant", "tile_repeats", [1, 3, 1, 1]), constant_node("range_start_constant", "range_start", [], [-1.0]), constant_node("range_limit_constant", "range_limit", [], [1.0 + delta * 0.5]), constant_node("range_delta_constant", "range_delta", [], [delta]), range_node("mask_range", "range_start", "range_limit", "range_delta", "mask_range_values"), reshape_node("mask_reshape", "mask_range_values", "mask_shape", "mask_plane"), tile_node("mask_tile", "mask_plane", "tile_repeats", "masks_base"), int64_constant_node("quality_shape_constant", "quality_shape", [1, 3]), constant_node("quality_constant", "quality_base", [1, 3], [0.5, 0.75, 0.75]), int64_constant_node("low_shape_constant", "low_shape", [1, 3, 256, 256]), constant_of_shape_node("low_res_constant", "low_shape", "low_res_base"), int64_constant_node("scalar_shape_constant", "scalar_shape", [1]), int64_constant_node("axes_embedding", "reduce_embedding_axes", [0, 1, 2, 3]), int64_constant_node("axes_points", "reduce_points_axes", [0, 1, 2]), int64_constant_node("axes_labels", "reduce_labels_axes", [0, 1]), int64_constant_node("axes_mask", "reduce_mask_axes", [0, 1, 2, 3]), int64_constant_node("axes_has_mask", "reduce_has_mask_axes", [0]), int64_constant_node("axes_orig", "reduce_orig_size_axes", [0]), reduce_mean_node("reduce_embedding", "image_embeddings", "s0", [0, 1, 2, 3]), reduce_mean_node("reduce_points", "point_coords", "s1", [0, 1, 2]), reduce_mean_node("reduce_labels", "point_labels", "s2", [0, 1]), reduce_mean_node("reduce_mask", "mask_input", "s3", [0, 1, 2, 3]), reduce_mean_node("reduce_has_mask", "has_mask_input", "s4", [0]), reduce_mean_node("reduce_orig_size", "orig_im_size", "s5", [0]), reshape_node("reshape_embedding", "s0", "scalar_shape", "r0"), reshape_node("reshape_points", "s1", "scalar_shape", "r1"), reshape_node("reshape_labels", "s2", "scalar_shape", "r2"), reshape_node("reshape_mask", "s3", "scalar_shape", "r3"), reshape_node("reshape_has_mask", "s4", "scalar_shape", "r4"), reshape_node("reshape_orig_size", "s5", "scalar_shape", "r5"), add_node("sum01", "r0", "r1", "sum01_out"), add_node("sum02", "sum01_out", "r2", "sum02_out"), add_node("sum03", "sum02_out", "r3", "sum03_out"), add_node("sum04", "sum03_out", "r4", "sum04_out"), add_node("sum05", "sum04_out", "r5", "all_inputs_scalar"), add_node("masks_add", "masks_base", "all_inputs_scalar", "masks"), add_node("quality_add", "quality_base", "all_inputs_scalar", "iou_predictions"), add_node("low_add", "low_res_base", "all_inputs_scalar", "low_res_masks")],
    )
    (OUT / "sam_synthetic_encoder.onnx").write_bytes(model(encoder))
    (OUT / "sam_synthetic_decoder.onnx").write_bytes(model(decoder))


if __name__ == "__main__":
    main()
