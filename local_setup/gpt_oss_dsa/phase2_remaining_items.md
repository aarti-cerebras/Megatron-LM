# GPT-OSS DSA Phase 2 Remaining Items

**Audit date:** 2026-08-21
**Branch inspected:** `aarti/gpt-oss-dsa`
**HEAD inspected:** `7aeefd9eb` (`Add real-token masking for GPT-OSS DSA SFT`)

## Summary

Phase 2 is not yet runnable end to end. Workstream A, sink-aware GQA sparse attention, and
Workstream B, SFT real-token provenance and assistant-only supervision, are committed. Their full
focused suites pass under the repository container on 8 H100s with the installed native GPT-OSS
tokenizer. GPT-OSS attention-bias fidelity is resolved, and a bias-correct 100-step Phase 1
checkpoint containing trained indexer state is available. Helper-only reduction assertions,
prompt-gradient coverage, native-tokenizer edge cases, and distributed-optimizer/Megatron-FSDP
coverage still need stronger integration tests. The Phase 1-to-Phase 2 transition now passes on the
full 20B model: the actual iteration-100 checkpoint loads model-only, a fresh joint optimizer and
schedule run two native-Harmony SFT steps with finite base/indexer gradients, and the resulting
Phase 2 checkpoint saves successfully. Reproducible Phase 2 train/run/validate launchers are now
available. Phase 2 diagnostics, functional coverage, real SFT data, and rollout validation remain
open.

The highest-priority dependency is now preparing and validating the real conversation SFT dataset,
then running Gate B for 20 steps with the new Phase 2 recipe. That run must exercise realistic data
variation, confirm stable base/indexer gradient balance, and provide the remaining diagnostics.

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
- [x] Preserve GPT-OSS Q/K/V/output and expert projection biases in the Phase 1 launcher.
- [x] Save a bias-correct 100-step Phase 1 checkpoint containing trained indexer state.
- [x] Strictly transition that checkpoint into the full 20B Phase 2 SFT model and run joint
  base/indexer optimization.
- [x] Add reproducible Phase 2 SFT train, run, and validation launchers.

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

### 1. GPT-OSS attention-bias fidelity (resolved)

The prior README recorded `attention_bias: true` as an open question. The original artifacts made
the mismatch concrete:

- ModelOpt conversion built bias-enabled attention projections.
- The original `local_setup/train_gpt_oss_20b_dsa_phase1.sh` passed `--disable-bias-linear`.
- The original Phase 1 training log therefore reported `add_bias_linear=False` and
  `add_qkv_bias=False` when loading a checkpoint converted with those values enabled.

Required work:

- [x] Remove or replace `--disable-bias-linear` with the configuration matching the converted HF
  checkpoint.
- [x] Verify Q/K/V/output projection biases load and round-trip without missing or unexpected keys.
- [x] Add a regression test that prevents a converted GPT-OSS checkpoint from silently dropping its
  trained attention biases.
- [x] Re-run Phase 1 from the bias-correct checkpoint.
- [x] Save an actual trained Phase 1 checkpoint containing the indexer state.

Validation record (2026-08-21):

- The launcher now leaves Megatron's `add_bias_linear=True` default enabled; argument validation
  also sets `add_qkv_bias=True`. `--no-bias-dropout-fusion` remains set for grouped-GEMM bias
  compatibility.
- The focused Phase 1 test file passed on every rank under an 8-GPU distributed invocation:
  `42 passed` per rank. It explicitly accepts bias-enabled DSA-GQA and rejects unused
  `linear_qkv.bias` or `linear_proj.bias` checkpoint keys.
- A one-step load/save from the ModelOpt release checkpoint and a second load from that saved
  checkpoint both exited zero with bias enabled. The reload had no incompatible-key report.
- The bias-correct 100-step run completed with zero skipped and zero NaN iterations. Indexer KL
  fell from `4.692320` at step 1 to `0.101674` at step 100; top-K recall reached `0.905476`, and
  attention-score recall reached `0.968656`.
- The trained checkpoint tracker records iteration 100 at:
  `/cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase1_bias_correct_100step_20260821T005100Z/checkpoints`.
  Its distributed metadata contains both `linear_qkv.bias` and `linear_proj.bias` keys as well as
  the trained DSA indexer state.

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

- [x] Build Phase 1 and Phase 2 with identical module trees and state-dict key sets.
- [x] Strictly load a saved Phase 1 checkpoint into Phase 2 with no missing or unexpected keys.
- [x] Include indexer state, external input normalization, Q/K/V/output weights and biases, and the
  dense delegate's `softmax_offset`.
- [x] Verify changing evaluation top-K does not change the state-dict structure.

Optimizer and scheduler requirements:

- [x] Load Phase 1 as model weights only; do not restore the indexer-only optimizer state.
- [x] Rebuild the optimizer with both base and indexer parameters trainable.
- [x] Verify every trainable parameter appears in exactly one optimizer group.
- [x] Verify indexer groups use the DSA maximum/minimum LR endpoints.
- [x] Verify base groups use the base maximum/minimum LR endpoints.
- [x] Start a new Phase 2 warmup and decay schedule.
- [x] Verify finite, nonzero base and indexer gradients after one SFT backward pass.
- [x] Verify the expected base/indexer learning rates and gradient norms are logged.

This must be a distributed unit test. The ordinary functional checkpoint-resume test cannot model a
transition in which the phase flags and optimizer population change.

Distributed contract validation (2026-08-21):

- `test_dsa_phase1_to_phase2_transition_rebuilds_joint_optimizer_and_scheduler` passed on every
  rank under `torch.distributed.run --nproc-per-node 8` as part of the full focused Phase 1 file
  (`42 passed` per rank).
- The test strictly transfers the complete Phase 1 model state into an identical Phase 2 module
  tree with a different evaluation top-K, discards the populated Phase 1 AdamW state, rebuilds
  disjoint base/indexer groups with their independent LR endpoints, starts a fresh warmup, and
  proves finite nonzero gradients on every rank.
- The full 20B gate at
  `/cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase2_sft_smoke_20260821T012127Z` then strictly
  loaded the actual iteration-100 `torch_dist` checkpoint with `--finetune --no-load-optim
  --no-load-rng` and completed two native-Harmony SFT optimizer steps on 8 H100s.
- A follow-up no-save logging gate at
  `/cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase2_sft_smoke_20260821T013826Z` fixed and validated
  indexer-LR reporting under distributed-optimizer sharding. Step 1 logged base/indexer LRs
  `1e-5`/`1e-4`, base/indexer gradient norms `272.402`/`159.479`, LM loss `3.291085`, indexer KL
  `0.909339`, zero skipped iterations, and zero NaN iterations. Step 2 logged the configured
  base/indexer minimum LRs `1e-6`/`1e-5` and remained finite.
- Rank 0 peaked at `59,541.02` MB allocated and `59,636.00` MB reserved. The iteration-2 Phase 2
  checkpoint, including the joint distributed optimizer state, saved successfully. Its eight data
  shards total approximately 293 GB, and the save took about 9 minutes 40 seconds.
- The live run exposed and fixed an integration bug in which MCore-only `real_token_mask_q`
  provenance was forwarded into dense Transformer Engine attention. The focused regression passed
  on all 8 ranks.

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

- [x] `local_setup/train_gpt_oss_20b_dsa_phase2_sft.sh`
- [x] `local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh`
- [x] `local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh`

Each run directory must record:

- [x] Resolved launch and training commands.
- [x] Source branch, commit, git status, and dirty-worktree patch.
- [x] Phase 1 checkpoint path and selected iteration.
- [x] Tokenizer and dataset paths.
- [x] Argument and environment metadata.
- [ ] Per-rank and combined logs.
- [x] TensorBoard output.
- [x] Final status and exit code.
- [x] Peak-memory and timing summaries.

The initial Phase 2 recipe must retain the Phase 1 model/indexer geometry, use sparse attention with
`dsa_kernel_backend=none`, omit `dsa_dense_warmup` and `dsa_freeze_base`, retain full-support indexer
KL, and consume SFT conversation JSONL through `SFTTokenizer` with the validated `gpt-oss` prompt
format.

The two-step smoke gate satisfies this recipe contract. Its bundled JSONL is intentionally synthetic
and validates plumbing only; it does not satisfy the real-data requirements in section 6 or the
20-step Gate B requirement in section 8.

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

- [x] Replace the statement that the GPT-OSS 20B safetensors are missing. The model directory now
  contains the complete top-level safetensor checkpoint and index.
- [x] Replace or qualify the statement that conversion through Megatron-Bridge is still required.
  A ModelOpt-converted MCore checkpoint exists, and its attention-bias fidelity is now validated by
  the bias-correct Phase 1 load/save/reload runs.

The README should state that the remaining real-finetune prerequisites are prepared conversation
SFT data, production diagnostics, a functional test, and the rollout gates above. The bias-correct
Phase 1 checkpoint, live 20B transition, and initial Phase 2 launch recipes are now available.

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
