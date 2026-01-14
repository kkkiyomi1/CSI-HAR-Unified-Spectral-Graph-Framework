import os
import json
import tempfile

import numpy as np
import pytest

import torch

from csihar.core import (
    WidarBVP,
    GenericCSIDataset,
    build_model_from_config,
    evaluate_split,
)

try:
    import scipy.io as sio
except Exception:
    sio = None


@pytest.mark.parametrize("T_new", [16, 32])
def test_generic_csi_dataset_forward(T_new):
    with tempfile.TemporaryDirectory() as td:
        # create a fake complex CSI sample
        T, K = 20, 30
        csi = (np.random.randn(T, K) + 1j * np.random.randn(T, K)).astype(np.complex64)
        p = os.path.join(td, "sample.npz")
        np.savez_compressed(p, csi=csi)

        ds = GenericCSIDataset(
            paths=[p],
            labels=[2],
            file_key="csi",
            T=T_new,
            resample="linear",
            cache_dir=None,
            view={"mode": "amp_phase", "phase_unwrap": True, "phase_sanitize": True},
            normalize={"per_sample": True, "eps": 1e-6},
        )
        x, y = ds[0]
        assert x.ndim == 3  # [C,T,K]
        assert x.shape[1] == T_new
        assert int(y.item()) == 2

        # build a small model and forward
        model_cfg = {"name": "wiprompt_tcn", "params": {"tcn_channels": 32, "tcn_layers": 2, "emb_dim": 64}}
        model = build_model_from_config(model_cfg, n_classes=5)
        h, logits = model(x.unsqueeze(0))
        assert h.shape[-1] == 64
        assert logits.shape[-1] == 5


@pytest.mark.skipif(sio is None, reason="scipy not installed")
def test_widar_bvp_dataset_mat_loading():
    with tempfile.TemporaryDirectory() as td:
        # create a fake Widar-like .mat with velocity_spectrum_ro as [N,N,T]
        N = 4
        T_old = 10
        V = np.abs(np.random.randn(N, N, T_old)).astype(np.float32)

        mat_path = os.path.join(td, "user1-1-1-1-3-1-foo.mat")
        sio.savemat(mat_path, {"velocity_spectrum_ro": V})

        ds = WidarBVP(
            paths=[mat_path],
            labels=[3],
            bvp_T=16,
            temporal_feats=["level", "diff"],
            ma_win=3,
            label_source="manifest",
            cache_dir=None,
        )

        x, y = ds[0]
        assert x.shape[0] == 2            # C=2 for level+diff
        assert x.shape[1] == 16           # resampled T
        assert x.shape[2] == N * N        # flattened K
        assert int(y.item()) == 3


def test_evaluate_split_smoke():
    # smoke test evaluate_split
    model_cfg = {"name": "wiprompt_tcn", "params": {"tcn_channels": 16, "tcn_layers": 1, "emb_dim": 32}}
    model = build_model_from_config(model_cfg, n_classes=3)

    # create a tiny loader
    xs = torch.randn(8, 2, 16, 10)  # [B,C,T,K]
    ys = torch.randint(0, 3, (8,))
    loader = [(xs, ys)]

    res = evaluate_split(model, loader, device=torch.device("cpu"), n_classes=3)
    assert "top1" in res
    assert 0.0 <= float(res["top1"]) <= 1.0
