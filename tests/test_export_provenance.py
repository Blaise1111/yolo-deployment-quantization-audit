import pytest

from src import export_quantize as export
from src import parity_gate as gate
from src.coco_split import digest, write_json


@pytest.fixture
def cache(tmp_path, monkeypatch):
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"original checkpoint")
    model = tmp_path / "onnx_fp32_640.onnx"
    annotations = tmp_path / "annotations.json"
    annotations.write_bytes(b"annotations")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(export, "WEIGHTS", weights)
    monkeypatch.setattr(export, "IMAGES", tmp_path)
    monkeypatch.setattr(export, "model_path", lambda size, backend: tmp_path / f"{backend}_{size}.onnx")
    monkeypatch.setattr(gate, "WEIGHTS", weights)
    monkeypatch.setattr(gate, "ANNOTATIONS", annotations)
    config = dict(resolutions=[640], cpu_threads=4, weight_sha256=digest(weights))
    partitions = dict(eval_1000=[7], eval_4800=[7, 8], calib_200=[9])
    images = {7: {"file_name": image.name}}
    temporary = tmp_path / "fp32.tmp"
    temporary.write_bytes(b"exported graph")
    export.publish_export(model, temporary, export.fp32_contract(640, 4))
    return weights, model, config, partitions, images


def test_checkpoint_content_change_rejects_export_and_fast_gate(cache, tmp_path):
    weights, model, config, partitions, images = cache
    path = tmp_path / gate.GATE_NAME
    original = gate.parity_contract(config, partitions, images)
    gate.publish_gate(path, original, {"640": 0.0001})

    assert export.export_fp32(640, 4) == model
    gate.validate_gate(gate.read_gate(path), gate.parity_contract(config, partitions, images))

    weights.write_bytes(b"changed checkpoint")
    with pytest.raises(export.ExportCacheError, match="inputs or configuration differ"):
        export.export_fp32(640, 4)

    # A fresh FP32 cache can exist under the same name, even with the same graph bytes.
    # The gate must independently reject the changed checkpoint.
    temporary = tmp_path / "replacement.tmp"
    temporary.write_bytes(model.read_bytes())
    export.publish_export(model, temporary, export.fp32_contract(640, 4))
    assert export.export_fp32(640, 4) == model
    with pytest.raises(gate.ParityGateError, match="checkpoint_sha256"):
        gate.validate_gate(gate.read_gate(path), gate.parity_contract(config, partitions, images))


def test_artifact_and_gate_inputs_must_match(cache, tmp_path):
    _, model, config, partitions, images = cache
    path = tmp_path / gate.GATE_NAME
    with pytest.raises(gate.ParityGateError, match="Missing"):
        gate.read_gate(path)
    write_json(path, {"config": config})
    with pytest.raises(gate.ParityGateError, match="Legacy"):
        gate.read_gate(path)
    contract = gate.parity_contract(config, partitions, images)
    gate.publish_gate(path, contract, {"640": 0.0002})
    (tmp_path / "image.jpg").write_bytes(b"changed image")
    with pytest.raises(gate.ParityGateError, match="fast_images"):
        gate.validate_gate(gate.read_gate(path), gate.parity_contract(config, partitions, images))
    model.write_bytes(b"changed export")
    with pytest.raises(export.ExportCacheError, match="artifact SHA-256"):
        export.export_fp32(640, 4)
    model.with_suffix(".provenance.json").unlink()
    with pytest.raises(export.ExportCacheError, match="Incomplete"):
        export.export_fp32(640, 4)


def test_derived_contract_changes_with_upstream_method_and_calibration_bytes(cache, tmp_path):
    _, base, config, _, _ = cache
    config["calibration_method"] = "MinMax"
    ids = list(range(200))
    calibration = tmp_path / "calibration.jpg"
    calibration.write_bytes(b"original")
    inputs = {i: {"file_name": calibration.name} for i in ids}

    def contract(backend, method="MinMax"):
        return export.variant_contract(base, 640, backend, ids, inputs, {**config, "calibration_method": method})

    fp16 = contract("onnx_fp16")
    static = contract("onnx_int8")
    assert static != contract("onnx_int8", "Entropy")
    calibration.write_bytes(b"changed")
    assert static != contract("onnx_int8")
    assert fp16 == contract("onnx_fp16")
    base.write_bytes(b"other FP32 graph")
    assert fp16 != contract("onnx_fp16")
