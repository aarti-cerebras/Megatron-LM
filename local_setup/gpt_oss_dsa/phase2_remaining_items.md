# GPT-OSS DSA Phase 2 Remaining Items

**Audit date:** 2026-08-21
**Branch inspected:** `aarti/gpt-oss-dsa`
**HEAD inspected:** `1de90b5a1` (`Add sink-aware GQA sparse attention`)

## Summary

Phase 2 is not yet runnable end to end. Workstream A, sink-aware GQA sparse attention, is committed.
Workstream B is implemented in the working tree and its full focused suite passes under the
repository container on 8 H100s with the installed native GPT-OSS tokenizer. Helper-only reduction
assertions, prompt-gradient coverage, native-tokenizer edge cases, and distributed-optimizer/
Megatron-FSDP coverage still need stronger integration tests. A bias-correct trained Phase 1
checkpoint, the Phase 1-to-Phase 2 transition, Phase 2 diagnostics, launch recipes, functional
coverage, real SFT data, and rollout validation remain open.

The highest-priority issue is model fidelity: the converted GPT-OSS checkpoint contains attention
projection biases, but the current Phase 1 launcher passes `--disable-bias-linear`. A Phase 1
checkpoint produced through that path is not suitable as the starting point for a real Phase 2
fine-tune until the bias contract is resolved and validated.

## Current state

### Completed

- [x] Implement true MQA/GQA/MHA query-head-to-KV-head mapping without expanding K/V by query head.
- [x] Include the learned GPT-OSS sink exactly once in sparse online softmax.
- [x] Cover sparse output and Q/K/V/sink gradients against a direct grouped-query reference.
- [x] Implement real-token provenance in `SFTDataset`.
- [x] Thread `real_token_mask` through TP, PP, CP, packed microbatches, and `PackedSeqParams`.
- [x] Implement a CP-global DSA real-query denominator independent of the LM loss denominator.
- [x] Implement a dedicated `gpt-oss` assistant-target parser using Harmony boundaries.
- [x] Add CPU-level provenance, padding-invariance, reduction, and synthetic Harmony parser tests.

### Completed validation

- [x] Run the full focused Workstream B suite with the repository-standard distributed GPU launcher.
- [x] Validate GPT-OSS target masking with the installed native tokenizer and tokenizer assets.

Validation record (2026-08-21):

- Container verification reported `EDITS LIVE`, 8 H100 80GB GPUs, and the expected MCore checkout.
- `GPT_OSS_TOKENIZER_PATH=/cb/ml-eng/aarti/models/gpt-oss-20b` was supplied to the test run.
- Each rank reported `339 passed, 113 skipped`; the skips require topologies larger than 8 GPUs.
- Combined log:
  `/cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase2_validation_20260821_full_native.log`.

### Completed delivery

- [x] Commit the Workstream B implementation and its previously untracked
  `tests/unit_tests/data/test_gpt_sft_phase2.py` coverage as an atomic change.

## Remaining items

### 1. Resolve GPT-OSS attention-bias fidelity

The README records `attention_bias: true` as an open question. Current artifacts make the mismatch
concrete:

- ModelOpt conversion built bias-enabled attention projections.
- `local_setup/train_gpt_oss_20b_dsa_phase1.sh` passes `--disable-bias-linear`.
- The Phase 1 training log therefore reports `add_bias_linear=False` and `add_qkv_bias=False` when
  loading a checkpoint converted with those values enabled.

Required work:

- [ ] Remove or replace `--disable-bias-linear` with the configuration matching the converted HF
  checkpoint.
- [ ] Verify Q/K/V/output projection biases load and round-trip without missing or unexpected keys.
- [ ] Add a regression test that prevents a converted GPT-OSS checkpoint from silently dropping its
  trained attention biases.
- [ ] Re-run Phase 1 from the bias-correct checkpoint.
- [ ] Save an actual trained Phase 1 checkpoint containing the indexer state.

The artifact store currently contains the converted release checkpoint, but no saved checkpoint
from the successful 100-step Phase 1 run. The latest 4K/10K attempt failed from GPU OOM while the
shared host had insufficient free device memory, so it must be rerun on sufficiently free GPUs.

### 2. Complete Workstream B validation

Required GPU coverage:

- [x] Run the focused SFT provenance and batch-plumbing tests under `torch.distributed.run`.
- [x] Cover PP > 1, CP > 1, and microbatch size > 1 in the same invocation.
- [x] Verify unequal CP real-row reduction against a concatenated reference in a helper test.
- [ ] Exercise genuinely unequal real-row counts across live CP ranks.
- [x] Verify per-microbatch denominator recomputation with different real-row counts in a helper
  test.
- [ ] Exercise different real-row counts through live gradient-accumulation microbatches.
- [x] Confirm in CPU helper coverage that padding changes neither dense indexer KL nor its gradients.
- [ ] Confirm padding invariance through the distributed GPU indexer path.
- [x] Confirm provenance marks prompt rows as real while LM supervision excludes them.
- [ ] Confirm prompt rows produce indexer gradients in a distributed GPU test.
- [x] Confirm in helper coverage that changing only the LM supervision mask does not change DSA
  normalization or gradients.
- [ ] Confirm LM-mask independence through the distributed GPU indexer path.
- [ ] Cover separate base/indexer LR groups under the distributed optimizer and Megatron FSDP.

Required native-tokenizer coverage:

- [x] Render conversations through the installed GPT-OSS Hugging Face chat template.
- [x] Verify developer, user, and tool messages are masked with the native tokenizer.
- [ ] Add explicit native-tokenizer coverage proving system messages are masked.
- [x] Verify assistant analysis, final, and tool-call payloads and terminators are supervised.
- [ ] Verify a real assistant terminator remains supervised when its token ID is also used for
  padding.
- [x] Verify an unterminated Harmony assistant message fails explicitly.
- [ ] Verify other malformed Harmony message streams fail explicitly.

Focused unit tests ran inside the repository container through:

```bash
GPT_OSS_TOKENIZER_PATH=/cb/ml-eng/aarti/models/gpt-oss-20b \
uv run --no-sync python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/data/test_gpt_sft_phase2.py \
  tests/unit_tests/data/test_get_batch.py \
  tests/unit_tests/transformer/experimental_attention_variant/test_attention_variant_dsa.py
```

`--no-sync` is required for this local container launcher because the pre-provisioned `/opt/venv`
is image-owned and the host UID cannot rewrite it.

### 3. Implement the Phase 1-to-Phase 2 transition

Model-state requirements:

- [ ] Build Phase 1 and Phase 2 with identical module trees and state-dict key sets.
- [ ] Strictly load a saved Phase 1 checkpoint into Phase 2 with no missing or unexpected keys.
- [ ] Include indexer state, external input normalization, Q/K/V/output weights and biases, and the
  dense delegate's `softmax_offset`.
- [ ] Verify changing evaluation top-K does not change the state-dict structure.

Optimizer and scheduler requirements:

- [ ] Load Phase 1 as model weights only; do not restore the indexer-only optimizer state.
- [ ] Rebuild the optimizer with both base and indexer parameters trainable.
- [ ] Verify every trainable parameter appears in exactly one optimizer group.
- [ ] Verify indexer groups use the DSA maximum/minimum LR endpoints.
- [ ] Verify base groups use the base maximum/minimum LR endpoints.
- [ ] Start a new Phase 2 warmup and decay schedule.
- [ ] Verify finite, nonzero base and indexer gradients after one SFT backward pass.
- [ ] Verify the expected base/indexer learning rates and gradient norms are logged.

This must be a distributed unit test. The ordinary functional checkpoint-resume test cannot model a
transition in which the phase flags and optimizer population change.

### 4. Add Phase 2 diagnostics

Retain the existing KL, scaled indexer loss, recall, captured-mass, gradient-norm, and learning-rate
metrics. Add:

- [ ] Sparse sink mass per DSA layer.
- [ ] Real DSA query-row count used by each reduction.
- [ ] Supervised SFT token count.
- [ ] Prompt, assistant, and padding token fractions.
- [ ] Configured top-K and mean valid selected-key count.
- [ ] Time spent in index selection, teacher KL, and sparse attention.
- [ ] Peak allocated and reserved GPU memory.

If sparse-support indexer loss is evaluated later, full-teacher quality metrics must still be
computed on diagnostic steps.

### 5. Add reproducible Phase 2 launchers

Create:

- [ ] `local_setup/train_gpt_oss_20b_dsa_phase2_sft.sh`
- [ ] `local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh`
- [ ] `local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh`

Each run directory must record:

- [ ] Resolved launch and training commands.
- [ ] Source branch, commit, git status, and dirty-worktree patch.
- [ ] Phase 1 checkpoint path and selected iteration.
- [ ] Tokenizer and dataset paths.
- [ ] Argument and environment metadata.
- [ ] Per-rank and combined logs.
- [ ] TensorBoard output.
- [ ] Final status and exit code.
- [ ] Peak-memory and timing summaries.

The initial Phase 2 recipe must retain the Phase 1 model/indexer geometry, use sparse attention with
`dsa_kernel_backend=none`, omit `dsa_dense_warmup` and `dsa_freeze_base`, retain full-support indexer
KL, and consume SFT conversation JSONL through `SFTTokenizer` with the validated `gpt-oss` prompt
format.

### 6. Prepare real SFT data

The existing GPT-OSS IFT datasets are stored as Hugging Face Arrow and cannot be supplied directly
to the current SFT `--data-path`. Required work:

- [ ] Select the Phase 2 SFT dataset and document its schema and provenance.
- [ ] Convert or export it to the conversation JSONL schema consumed by `SFTDataset`.
- [ ] Validate representative system, developer, user, assistant, and tool conversations.
- [ ] Record dataset paths and immutable version/manifest information in every run directory.
- [ ] Check packing, truncation, token fractions, and assistant-target counts before a long run.

### 7. Add the functional test and CI recipe

- [ ] Add a small H100 functional case under `tests/functional_tests/test_cases/` using scaled-down
  GPT-OSS geometry and conversation JSONL.
- [ ] Cover GQA, learnable sinks, mixed sliding/standard/DSA layers, sparse top-K attention, SFT and
  DSA losses, diagnostics, and Phase 2 checkpoint save/resume.
- [ ] Exercise TP, PP, and CP where practical.
- [ ] Add an H100 recipe under `tests/test_utils/recipes/h100/` with the required scope,
  environment, platform, repeat count, and time limit.
- [ ] Trigger the functional job with the `Run functional tests` PR label.
- [ ] After a clean run, download and commit golden values with
  `tests/test_utils/python_scripts/download_golden_values.py`.

### 8. Execute the 20B rollout gates

| Gate | Configuration | Required result |
| --- | --- | --- |
| A | Sequence 128; K covers every legal key | Sparse/dense output and gradient parity; strict checkpoint load |
| B | Sequence 128; realistic K; 20 steps | Finite SFT/KL losses; nonzero base/indexer gradients; correct LRs |
| C | Sequence 4K; realistic K; 100 steps | Stable memory; finite/falling loss; no abrupt recall or mass regression |
| D | Sequence 8K; realistic K | Measured throughput and peak memory; stable sparse sink mass |
| E | Longer SFT run | Short- and long-context evaluation within agreed dense-baseline tolerances |
| F | 32K candidate | Run only after quadratic cost is accepted or optimized |

At the Phase 1 exit and throughout Phase 2, evaluate indexer quality at the K intended for Phase 2.
Tune `dsa_indexer_loss_coeff` from the observed base/indexer gradient norms rather than carrying over
a Phase 1 coefficient by assumption.

### 9. Resolve or accept the long-context performance limitation

The PyTorch correctness backend remains quadratic in index scoring, and full-support teacher KL
also scans the query/key space in blocks. FlashMLA is built without H100 kernels, while TileLang on
SM90 remains unverified.

- [ ] Profile index selection, teacher KL, and sparse attention separately at 4K and 8K.
- [ ] Verify peak-memory behavior on an uncontended H100 node.
- [ ] Decide whether correctness-backend performance is acceptable for the intended Phase 2 run.
- [ ] Before 32K, either explicitly accept the measured cost or implement an optimization such as
  scheduled teacher KL, query/layer sampling with an unbiased objective, a fused grouped-query
  teacher-loss backend, or a fused GQA sparse-attention kernel.
- [ ] Preserve a training signal for important omitted keys; do not replace full-support KL with a
  selected-support-only objective solely for speed.

## Documentation corrections

`local_setup/README.md` contains two stale prerequisites:

- [ ] Replace the statement that the GPT-OSS 20B safetensors are missing. The model directory now
  contains the complete top-level safetensor checkpoint and index.
- [ ] Replace or qualify the statement that conversion through Megatron-Bridge is still required.
  A ModelOpt-converted MCore checkpoint exists, although the attention-bias mismatch must be fixed
  before treating it as a fidelity-validated Phase 1 starting point.

The README should state that the remaining real-finetune prerequisites are a bias-correct trained
Phase 1 checkpoint, prepared conversation SFT data, the Phase 2 transition and recipes, and the
validation gates above.

## Separate later milestones

The following remain out of scope for the training-only Phase 2 readiness gate:

- Sparse KV-cache decode.
- Serving and inference support.
- vLLM integration.
- A production fused GQA sparse-attention kernel, except where performance requirements make it
  necessary for the chosen context length.

## Phase 2 readiness condition

Phase 2 is ready for sustained GPT-OSS 20B SFT only when the bias-correct trained Phase 1 checkpoint
loads strictly, base and indexer parameters jointly optimize at their configured learning rates,
assistant-only LM masking and real-query DSA normalization pass distributed/native-tokenizer
validation, required diagnostics and functional coverage are present, the 4K/8K rollout gates fit
the measured memory budget, and the quadratic execution limit is explicitly accepted or resolved
for the target context length.
