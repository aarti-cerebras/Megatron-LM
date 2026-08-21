"""Isolate which TE attention backends accept the gpt-oss attention config.

The 20B run fails with "No dot product attention backend is available for the
provided inputs" when --attention-backend=fused, and silently falls back to
UNFUSED (materializing a seq x seq score matrix -> OOM) when left on auto.

This probe builds TE's DotProductAttention directly with gpt-oss-20b geometry and
tries each variable in isolation, forward-only and forward+backward, so we can see
exactly which feature kills the fused backends. Run with NVTE_DEBUG_LEVEL=2 to get
TE's own per-backend rejection reasons.
"""

import os

import torch
import transformer_engine.pytorch as tep
from transformer_engine.pytorch.attention import DotProductAttention

# gpt-oss-20b attention geometry (config.json)
NH, NKV, HD = 64, 8, 64
SEQ = int(os.environ.get("PROBE_SEQ", "1024"))
B = 1
WINDOW = (127, 0)  # 128-token sliding window
DEV = "cuda"
DT = torch.bfloat16


def try_case(name, *, softmax_type="vanilla", window=None, backward=False, seq=SEQ):
    torch.cuda.empty_cache()
    kwargs = dict(
        num_attention_heads=NH,
        kv_channels=HD,
        num_gqa_groups=NKV,
        attention_dropout=0.0,
        attn_mask_type="causal",
    )
    if softmax_type != "vanilla":
        kwargs["softmax_type"] = softmax_type
    if window is not None:
        kwargs["window_size"] = window

    try:
        dpa = DotProductAttention(**kwargs).to(DEV)
        q = torch.randn(seq, B, NH, HD, device=DEV, dtype=DT, requires_grad=backward)
        k = torch.randn(seq, B, NKV, HD, device=DEV, dtype=DT, requires_grad=backward)
        v = torch.randn(seq, B, NKV, HD, device=DEV, dtype=DT, requires_grad=backward)
        out = dpa(q, k, v)
        if backward:
            out.sum().backward()
        print(f"  PASS   {name}")
        return True
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:150]
        print(f"  FAIL   {name}\n           -> {type(e).__name__}: {msg}")
        return False


if __name__ == "__main__":
    print(
        f"TE {tep.__version__ if hasattr(tep,'__version__') else '?'} | "
        f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}"
    )
    print(
        f"NVTE_FUSED_ATTN={os.environ.get('NVTE_FUSED_ATTN')} "
        f"NVTE_FLASH_ATTN={os.environ.get('NVTE_FLASH_ATTN')} "
        f"NVTE_UNFUSED_ATTN={os.environ.get('NVTE_UNFUSED_ATTN')}"
    )

    print("\n[baseline: no sinks, no window]")
    try_case("vanilla                fwd     ")
    try_case("vanilla                fwd+bwd ", backward=True)

    print("\n[sliding window only]")
    try_case("window=(127,0)         fwd     ", window=WINDOW)
    try_case("window=(127,0)         fwd+bwd ", window=WINDOW, backward=True)

    print("\n[learnable sinks only]  <- gpt-oss")
    try_case("softmax=learnable      fwd     ", softmax_type="learnable")
    try_case("softmax=learnable      fwd+bwd ", softmax_type="learnable", backward=True)

    print("\n[sinks + window]        <- full gpt-oss config")
    try_case("learnable + window     fwd     ", softmax_type="learnable", window=WINDOW)
    try_case(
        "learnable + window     fwd+bwd ", softmax_type="learnable", window=WINDOW, backward=True
    )
