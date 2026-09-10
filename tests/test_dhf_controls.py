"""Controls must preserve checkpoint behavior and isolate one mathematical term."""

import copy
import io

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2DHF, HDFMv2


def test_hdfm_control_and_legacy_checkpoint():
    torch.manual_seed(0)
    module = HDFMv2([8, 16, 32], 8).eval()
    features = [torch.randn(2, c, s, s) for c, s in [(8, 16), (16, 8), (32, 4)]]
    aligned = [F.interpolate(a(x), (16, 16), mode="nearest") for a, x in zip(module.align, features)]
    z = torch.cat(aligned, 1)
    weights = (module.spatial_weight(z) + module.scale_weight(z)).softmax(1)
    expected = aligned[0] + module.gamma * module.refine(sum(x * weights[:, i:i+1] for i, x in enumerate(aligned)))
    torch.testing.assert_close(module(features), expected, rtol=0, atol=0)
    del module.global_scale
    torch.testing.assert_close(module(features), expected, rtol=0, atol=0)
    local = copy.deepcopy(module)
    local.global_scale = 0.0
    weights = local.spatial_weight(z).softmax(1)
    expected_local = aligned[0] + local.gamma * local.refine(sum(x * weights[:, i:i+1] for i, x in enumerate(aligned)))
    torch.testing.assert_close(local(features), expected_local, rtol=0, atol=0)
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in module.parameters())
    local(features).square().mean().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in local.scale_weight.parameters())
    stream = io.BytesIO()
    torch.save(local, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert restored.global_scale == 0.0
    torch.testing.assert_close(restored(features), expected_local)


def test_c3k2_contrast_control_and_legacy_checkpoint():
    torch.manual_seed(0)
    module = C3k2DHF(16, 16).eval()
    x = torch.randn(2, 16, 16, 16)
    base = module.main(x)
    y = module.reduce(x)
    gate = module.gate(torch.cat((base.mean(1, keepdim=True), base.amax(1, keepdim=True)), 1))
    edge = module.edge(y - F.avg_pool2d(y, 3, 1, 1))
    expected = base + module.gamma * module.fuse(torch.cat((base, edge * gate, module.context(y) * gate), 1))
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    del module.contrast_scale
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    plain = copy.deepcopy(module)
    plain.contrast_scale = 0.0
    expected_plain = base + plain.gamma * plain.fuse(torch.cat((base, plain.edge(y) * gate, plain.context(y) * gate), 1))
    torch.testing.assert_close(plain(x), expected_plain, rtol=0, atol=0)


@pytest.mark.parametrize("value", [-1, 0.5, 2])
def test_invalid_controls(value):
    with pytest.raises(ValueError):
        HDFMv2([8, 8], 8, global_scale=value)
    with pytest.raises(ValueError):
        C3k2DHF(8, 8, contrast_scale=value)


def test_prepared_queue_configs_keep_nano_scale(tmp_path):
    import yaml
    from tools.submission_revision_queue import prepare_configs
    from ultralytics import YOLO

    prepare_configs(tmp_path, tmp_path / 'dataset')
    for path in tmp_path.glob('yolo26n-*.yaml'):
        cfg = yaml.safe_load(path.read_text())
        assert list(cfg['scales']) == ['n']
        model = YOLO(str(path))
        assert model.model.yaml['scale'] == 'n'
        assert 2_000_000 < sum(p.numel() for p in model.model.parameters()) < 3_000_000
