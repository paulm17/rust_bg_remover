"""Generate the checked-in M3 ONNX fixture without external model packages.

The graph is an opset-13 Relu model with input ``input`` and output ``mask``.
It applies the standard non-negative clamp. Its four dimensions are
[1, 3, height, width], where height and width are symbolic. The protobuf
encoding follows ONNX ModelProto and is kept deliberately small so generation
is reproducible offline.
"""
from pathlib import Path


def varint(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def field(num, wire, value):
    return varint((num << 3) | wire) + value


def string(num, value):
    raw = value.encode()
    return field(num, 2, varint(len(raw)) + raw)


def message(num, value):
    return field(num, 2, varint(len(value)) + value)


def i64(num, value):
    return field(num, 0, varint(value))


def dim(symbol):
    return string(2, symbol)


def value_info(name):
    shape = b"".join(message(1, dim(s)) for s in ["batch", "channel", "height", "width"])
    tensor_shape = message(2, shape)
    tensor_type = i64(1, 1) + tensor_shape  # TensorProto.FLOAT
    type_proto = message(1, tensor_type)
    return string(1, name) + message(2, type_proto)


def main():
    node = string(1, "input") + string(2, "mask") + string(3, "relu") + string(4, "Relu")
    graph = message(1, node) + string(2, "m3_arithmetic")
    graph += message(11, value_info("input")) + message(12, value_info("mask"))
    opset = string(1, "") + i64(2, 13)
    model = i64(1, 8) + string(2, "bgremove-m3-fixture") + message(7, graph) + message(8, opset)
    out = Path("models/fixtures/m3_identity.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(model)


if __name__ == "__main__":
    main()
