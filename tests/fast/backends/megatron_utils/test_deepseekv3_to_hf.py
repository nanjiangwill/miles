import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_converter():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "miles"
        / "backends"
        / "megatron_utils"
        / "megatron_to_hf"
        / "deepseekv3.py"
    )
    spec = importlib.util.spec_from_file_location("test_deepseekv3_to_hf_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.convert_deepseekv3_to_hf


convert_deepseekv3_to_hf = _load_converter()


def _args(**kwargs):
    defaults = {
        "update_weight_transfer_mode": "disk-delta",
        "bf16": True,
        "fp16": False,
        "kv_channels": 128,
        "num_attention_heads": 64,
        "num_query_groups": 1,
        "indexer_rope_interleave": False,
    }
    return SimpleNamespace(**(defaults | kwargs))


@pytest.mark.parametrize(
    ("args", "expected_dtype"),
    [
        (_args(), torch.bfloat16),
        (_args(bf16=False, fp16=True), torch.float16),
        (_args(update_weight_transfer_mode="nccl"), torch.float32),
    ],
)
def test_router_weight_dtype(args, expected_dtype):
    weight = torch.zeros((256, 16), dtype=torch.float32)

    [(name, converted)] = convert_deepseekv3_to_hf(
        args,
        "module.module.decoder.layers.3.mlp.router.weight",
        weight,
    )

    assert name == "model.layers.3.mlp.gate.weight"
    assert converted.dtype == expected_dtype


def test_disk_delta_preserves_router_expert_bias_dtype():
    bias = torch.zeros(256, dtype=torch.float32)

    [(name, converted)] = convert_deepseekv3_to_hf(
        _args(),
        "module.module.decoder.layers.3.mlp.router.expert_bias",
        bias,
    )

    assert name == "model.layers.3.mlp.gate.e_score_correction_bias"
    assert converted.dtype == torch.float32
