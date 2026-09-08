"""The engine-free indexer paths: causal scores, training top-k semantics, and the kernel padding
being exact. CPU, torch only (the Hadamard falls back to the torch FWHT explicitly)."""

import torch

from gpt_oss_dsa.config import DSAServingConfig, IndexerRope
from gpt_oss_dsa.indexer import (
    KERNEL_HEAD_DIM,
    GptOssDSAServingIndexer,
    fake_quant_fp8,
    quant_fp8_rows,
)

HIDDEN = 96


def make_indexer(seed: int = 0) -> GptOssDSAServingIndexer:
    torch.manual_seed(seed)
    cfg = DSAServingConfig(
        sparse_layer_ids=(1,),
        n_heads=16,
        head_dim=64,
        index_topk=8,
        fp8=True,
        fp8_ue8m0=True,
        rotate_activation=True,
        scoring_relu=True,
        k_norm_eps=1e-5,
        rope=IndexerRope(type="yarn", dim=64, theta=150000.0, scaling_factor=32.0),
        training_phase=2,
    )
    indexer = GptOssDSAServingIndexer(cfg, HIDDEN, allow_torch_hadamard=True).to(torch.bfloat16)
    with torch.no_grad():
        for p in indexer.parameters():
            p.copy_(torch.randn_like(p, dtype=torch.float32).to(torch.bfloat16) * 0.05)
    return indexer


def test_scores_are_causal_and_topk_marks_masked():
    indexer = make_indexer()
    T = 12
    x = torch.randn(T, HIDDEN).to(torch.bfloat16)
    scores = indexer.torch_scores(x, torch.arange(T))
    assert scores.shape == (T, T)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.isinf(scores[upper]).all() and (scores[upper] < 0).all()
    assert torch.isfinite(scores[~upper]).all()
    topk = indexer.torch_topk(scores, indexer.cfg.index_topk)
    assert topk.shape == (T, 8)
    # Row t has t + 1 legal keys; beyond that the training top-k reports -1.
    for t in range(T):
        assert int((topk[t] >= 0).sum()) == min(t + 1, 8)
        assert set(topk[t][topk[t] >= 0].tolist()) <= set(range(t + 1))


def test_kernel_padding_is_exact():
    """Zero-padding q/k to 128 dims and the heads to 32 changes neither the row scales nor the
    scores, which is what lets the 16x64 indexer run on DeepGEMM's 32x128 kernel."""
    indexer = make_indexer(1)
    T = 10
    x = torch.randn(T, HIDDEN).to(torch.bfloat16)
    positions = torch.arange(T)
    q, k, gates = indexer.project(x, positions)
    reference = indexer.torch_scores(x, positions)

    q_p, k_p, gates_p = indexer._pad_for_kernel(q, k, gates)
    assert q_p.shape == (T, 32, KERNEL_HEAD_DIM) and k_p.shape == (T, KERNEL_HEAD_DIM)
    q_fp8, q_scale = quant_fp8_rows(q_p.reshape(-1, KERNEL_HEAD_DIM))
    k_fp8, k_scale = quant_fp8_rows(k_p)
    q_deq = q_fp8.float().view(T, 32, KERNEL_HEAD_DIM) * q_scale.view(T, 32, 1)
    k_deq = k_fp8.float() * k_scale.view(T, 1)
    dots = torch.relu(torch.einsum("qhd,kd->qhk", q_deq, k_deq))
    padded_scores = torch.einsum("qhk,qh->qk", dots, gates_p.float())
    padded_scores = padded_scores.masked_fill(positions.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))

    finite = torch.isfinite(reference)
    assert torch.allclose(padded_scores[finite], reference[finite], rtol=1e-4, atol=1e-5)
    # The padded rows carried zero scale weight: their gates are zero and their q rows are zero.
    assert torch.equal(gates_p[:, 16:], torch.zeros_like(gates_p[:, 16:]))
    assert torch.equal(q_fp8.view(T, 32, -1)[:, 16:].float(), torch.zeros(T, 16, KERNEL_HEAD_DIM))


def test_fake_quant_is_a_fixed_point_within_e4m3_precision():
    torch.manual_seed(3)
    x = torch.randn(5, 64).to(torch.bfloat16)
    once = fake_quant_fp8(x)
    assert once.dtype == torch.bfloat16
    assert torch.equal(fake_quant_fp8(once), once)
    # E4M3 keeps 3 mantissa bits: normal values round within 2**-4 relative; the UE8M0 scale can
    # sit up to 2x above amax / 448, which bounds the subnormal step.
    scale_bound = 2.0 * torch.pow(2.0, torch.ceil(torch.log2(x.float().abs().amax(dim=-1, keepdim=True) / 448.0)))
    tolerance = x.float().abs() * 2**-4 + scale_bound * 2**-9
    assert torch.all((once.float() - x.float()).abs() <= tolerance)
