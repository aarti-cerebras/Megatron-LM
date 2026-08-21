# GPT-OSS DSA Phase 2 Top-K SFT Training Plan

**Status:** Implementation in progress; Workstreams A and B implemented, focused Workstream B GPU
and native-tokenizer validation complete

**Baseline:** `aarti/gpt-oss-dsa` at `26d2e475e`

**Scope:** GPT-OSS 20B, `experimental_attention_variant="dsa_gqa"`, training only

## 1. Objective

Phase 1 trains each DSA indexer against dense GPT-OSS attention while the backbone is frozen and
the language-model forward remains dense. Phase 2 must load that checkpoint, enable indexer-selected
top-K sparse attention, unfreeze the backbone, and jointly fine-tune the backbone and indexer on a
conversation-style supervised fine-tuning (SFT) dataset.

The intended Phase 2 model is:

```text
SFT conversation tokens
  -> standard GPT-OSS Q/K/V projections and RoPE
  -> DSA indexer scores every legal query/key pair
  -> shared token top-K selection
  -> sink-aware grouped-query sparse attention
  -> GPT-OSS output projection and remaining model
  -> assistant-target language-model loss

Dense teacher Q/K (detached)
  -> Design-A indexer KL side loss
```

Serving, sparse KV-cache decode, vLLM integration, and a production fused GQA sparse-attention
kernel remain separate milestones.

## 2. Important distinction: top-K attention versus sparse indexer loss

Two existing controls affect different parts of the computation:

- `dsa_dense_warmup=False` enables the sparse-attention output path. This is the Phase 2 switch.
- `dsa_indexer_use_sparse_loss=True` restricts the auxiliary indexer KL to the indexer's selected
  support. It does not enable sparse attention.

For the first Phase 2 implementation, enable sparse attention but retain the full-support Design-A
KL:

```text
dsa_dense_warmup = False
dsa_freeze_base = False
dsa_indexer_use_sparse_loss = False
dsa_indexer_loss_coeff > 0
```

The sparse-support KL is not the recommended starting objective. Once selection has happened, a KL
computed only over indexer-selected keys cannot directly teach the indexer that an important omitted
key should have been selected. The main SFT loss also cannot provide that signal because integer
top-K indices are non-differentiable. The indexer therefore continues to require an auxiliary
teacher objective in Phase 2.

## 3. Phase 2 loss contract

For DSA layers \(\mathcal{D}\), use:

```text
L_total = L_SFT + sum(layer in D)(lambda_layer * L_indexer_layer)
```

The two losses use deliberately different token populations:

| Loss | Included rows | Excluded rows | Parameters trained |
| --- | --- | --- | --- |
| SFT language-model loss | Assistant target tokens | Prompts and padding | Backbone, including sparse Q/K/V and sink parameters |
| DSA indexer KL | All real query tokens, including prompts | Padding only | Indexer parameters |

Prompt rows must remain in the indexer objective. They are valid attention queries, dominate
long-context prefill, and use sparse attention even when the prompt tokens are not language-model
targets.

The base and indexer learning rates remain independent:

```text
--lr <base maximum LR>
--min-lr <base minimum LR>
--dsa-indexer-lr <indexer maximum LR>
--dsa-indexer-min-lr <indexer minimum LR>
```

Tune `dsa_indexer_loss_coeff` as a per-layer coefficient using the logged indexer/base gradient
norms. Do not infer a coefficient conversion from Phase 1 because Phase 2 changes the trainable
parameter population and primary loss.

## 4. Current implementation status and blockers

Phase 1 already provides:

- the `dsa_gqa` layer pattern and standard GPT-OSS projection path;
- the hidden-state indexer, FP8 serving-compatible fake quantization, and top-K selection;
- grouped-query dense teacher scores for Design-A KL;
- a memory-bounded blockwise teacher/indexer loss;
- dense warmup delegation with a frozen base;
- stock-checkpoint remapping and guarded Phase 1 loading;
- separate base/indexer learning-rate groups and gradient diagnostics;
- per-layer KL, token recall, and decomposed attention-mass quality metrics.

Workstream A resolves the sparse-attention correctness blockers described below. Workstream B now
implements provenance-based SFT masking, CP-global DSA normalization, GPT-OSS assistant targets,
and the required batch plumbing. Its full focused suite passes under the repository container on
8 H100s with the native GPT-OSS tokenizer. Stronger live unequal-row, gradient-accumulation,
prompt-gradient, distributed-optimizer, and Megatron-FSDP integration coverage remains pending.
Checkpoint transition, diagnostics, recipes, and performance gaps still block a full Phase 2
training run.

### 4.1 Sparse attention rejects true GQA (resolved)

`unfused_dsa_fn` now maps each query-head chunk to its contiguous KV groups, covering MQA, GQA, and
MHA without materializing a full query-head-expanded K/V tensor. Numerical and gradient tests cover
the 64-query-head/8-KV-head ordering through a reduced GQA reference.

### 4.2 Sparse attention omits learned sink logits (resolved)

GPT-OSS dense attention uses a learned per-query-head `softmax_offset`. The sparse executor now seeds
its online softmax with that sink exactly once, includes it in the denominator with a zero value
vector, and propagates gradients to the dense delegate's sink parameter.

### 4.3 SFT padding is not excluded from indexer KL (implemented; focused validation complete)

`SFTDataset` now records token provenance independently of token IDs, the batch path threads it
through TP/PP/CP and packed microbatches, and `PackedSeqParams.real_token_mask_q` excludes synthetic
padding rows from DSA KL. CPU-level provenance and padding-invariance coverage passes as part of the
8-GPU distributed suite; an end-to-end GPU indexer padding-invariance assertion remains pending.

### 4.4 SFT and DSA need separate denominators (implemented; focused validation complete)

The DSA loss now uses a CP-global real-query count independently of the SFT language-model count.
Its logging path sums the CP-local normalized numerators rather than averaging local means. Unit
coverage exercises unequal CP row counts and different gradient-accumulation microbatch counts.
The batch plumbing passes with PP/CP and microbatch size greater than one on 8 GPUs; live unequal
row counts across CP ranks and actual gradient-accumulation microbatches remain pending.

### 4.5 GPT-OSS assistant-only masking needs an explicit contract (implemented; native validation complete)

The dedicated `gpt-oss` prompt format uses the native Hugging Face chat template and parses Harmony
message boundaries to supervise assistant analysis/final/tool-call payloads and their terminators
while masking system, developer, user, and tool messages. LM padding validity uses shifted token
provenance, so a real assistant terminator remains supervised even when its ID is also used for
padding. Parser tests use a faithful synthetic Harmony stream, and native validation passes with
the installed GPT-OSS tokenizer for developer/user/tool masking and assistant analysis, final, and
tool-call targets. Explicit native system-message and additional malformed-stream edge cases remain
pending.

### 4.6 The correctness backend is not yet a long-context performance backend

The PyTorch sparse executor is chunked and memory-bounded, but index scoring and full-support teacher
KL still scan the query/key space. This is suitable for correctness and short-to-medium context
training, not yet evidence that a 32K production run will be efficient.

## 5. Workstream A: sink-aware GQA sparse attention

### 5.1 Grouped-query head mapping

For each local query head, map to its local KV group using Megatron's contiguous GQA ordering:

```text
groups_per_kv = num_query_heads / num_kv_heads
kv_head(query_head) = query_head // groups_per_kv
```

For the GPT-OSS global layout:

```text
query heads  0-7  -> KV head 0
query heads  8-15 -> KV head 1
...
query heads 56-63 -> KV head 7
```

Apply this mapping inside the existing query-head chunk loop. Select only the KV heads required by
the current head chunk. Do not materialize expanded `[B, Hq, T, D]` K/V tensors or gathered
`[B, Hq, T, K, D]` tensors.

Use one implementation for MQA, GQA, and MHA:

- MQA: every query head maps to KV head zero;
- GQA: contiguous groups of query heads map to one KV head;
- MHA: each query head maps to the same-numbered KV head.

### 5.2 Learned sink in online softmax

The existing `dense_attention` submodule remains the single owner of `softmax_offset` in both
phases. Pass its local query-head tensor into the sparse executor.

For every query/head row, seed the online-softmax state before processing token chunks:

```text
running_max = sink_logit
denominator = 1        # exp(sink_logit - running_max)
numerator = 0          # sink has no value vector
```

Then apply the ordinary online-softmax update for every selected-key chunk. This produces:

```text
output = sum(exp(token_logit) * value)
         / (exp(sink_logit) + sum(exp(token_logit)))
```

The sink must be inserted exactly once per row, not once per top-K chunk. The operation must remain
differentiable with respect to the sink, query, key, and value tensors.

### 5.3 Sparse-attention constraints for v1

Validate or document the following initial constraints:

- causal self-attention only;
- attention dropout equal to zero unless explicitly implemented in the sparse executor;
- `dsa_kernel_backend="none"` for the correctness version;
- no cross-layer top-K reuse (`dsa_indexer_topk_freq=1`);
- training only; inference and KV-cache decode continue to fail explicitly.

## 6. Workstream B: SFT token provenance and loss normalization

### 6.1 Dataset real-token mask

Construct a Boolean `real_token_mask` in `SFTDataset` from provenance:

- append `True` with every token produced from a conversation;
- append `False` inside every padding operation;
- truncate it alongside tokens and targets;
- slice it as `input_ids` is sliced;
- preserve prompt positions as `True`.

Do not infer validity by comparing token IDs to the pad ID. A tokenizer can reuse an end-of-text ID
as padding, and that ID can appear in real input.

### 6.2 Batch plumbing

Thread `real_token_mask` through every explicit batch schema:

1. `SFTDataset` return dictionary;
2. `pretrain_gpt.py` `BATCH_KEYS` and positional unpacking;
3. tensor-parallel broadcast branches;
4. packed microbatch flattening;
5. per-document and per-sequence CP slicing;
6. intermediate pipeline-stage metadata forwarding;
7. `PackedSeqParams.real_token_mask_q`.

The combined PP > 1, CP > 1, and microbatch-size > 1 case is mandatory coverage because each axis
exercises different hard-coded plumbing.

### 6.3 DSA-specific denominator

For the default non-per-token auxiliary-loss path:

1. compute local indexer KL sum and local real-query count;
2. sum the count across the CP group without autograd;
3. form `local_kl_sum / global_cp_real_query_count`;
4. retain the existing auxiliary-loss scaling across gradient-accumulation microbatches;
5. reduce the logged numerator with SUM over CP;
6. do not use the SFT `loss_mask` count to normalize DSA gradients.

This yields an equal-microbatch average of CP-global real-query means. Exact token weighting across
all microbatches and DP ranks can be a later extension if required.

### 6.4 GPT-OSS assistant targets

Use the GPT-OSS tokenizer's native chat template for rendering. First verify whether the installed
Transformers version and GPT-OSS template provide an assistant-token mask. If they do, use it to
construct targets. If they do not, add a dedicated GPT-OSS prompt format with tested system,
developer, user, tool, assistant, analysis, and final-channel behavior.

The target mask and real-token mask must be tested independently:

```text
prompt token:    LM target = false, DSA real query = true
assistant token: LM target = true,  DSA real query = true
padding token:   LM target = false, DSA real query = false
```

## 7. Workstream C: Phase 1 to Phase 2 transition

### 7.1 Model state

Build Phase 1 and Phase 2 models with the same module tree. The only mode differences should be
configuration and `requires_grad` state. Assert identical checkpoint key sets, including:

- indexer projections and normalization;
- external input normalization;
- Q/K/V/output projection weights and biases that exist in the source model;
- `core_attention.dense_attention.softmax_offset`.

Changing `dsa_indexer_topk` between evaluation runs must not change the state-dict structure.

### 7.2 Optimizer and scheduler

Load the Phase 1 checkpoint as weights for a new fine-tuning run and rebuild optimizer state. Do not
reuse the indexer-only Phase 1 optimizer because the Phase 2 optimizer must include both base and
indexer parameters.

The transition test must assert:

- no missing or unexpected Phase 1/Phase 2 model keys;
- base and indexer parameters both have `requires_grad=True`;
- every trainable parameter appears exactly once in an optimizer group;
- indexer groups use the DSA LR endpoints;
- base groups use the base LR endpoints;
- the Phase 2 scheduler starts its own warmup/decay schedule.

### 7.3 Initial Phase 2 arguments

The Phase 2 recipe should retain the Phase 1 model and indexer geometry while flipping the phase
mode:

```text
--experimental-attention-variant dsa_gqa
--dsa-layer-freq 2
--dsa-indexer-topk <target K>
--dsa-indexer-loss-coeff <lambda>
--dsa-indexer-topk-freq 1
--dsa-kernel-backend none

# Intentionally absent in Phase 2:
# --dsa-dense-warmup
# --dsa-freeze-base

# Initially remain false:
# --dsa-indexer-use-sparse-loss
```

Use SFT data arguments instead of Phase 1 mock data:

```text
--sft
--data-path <conversation JSONL>
--tokenizer-type SFTTokenizer
--tokenizer-model <GPT-OSS tokenizer directory>
--sft-tokenizer-prompt-format <validated GPT-OSS format>
```

## 8. Workstream D: Phase 2 diagnostics

Retain the existing global and per-layer diagnostics:

- indexer KL and scaled indexer loss;
- indexer loss coefficient;
- indexer token top-K recall;
- attention mass captured by the indexer;
- attention mass captured by teacher top-K;
- attention mass captured by the top-K intersection;
- attention score recall;
- base and indexer gradient norms;
- base and indexer learning rates.

Add:

- sparse sink mass per DSA layer;
- number of real DSA query rows used in each reduction;
- number of supervised SFT tokens;
- prompt/assistant/padding token fractions;
- configured top-K and mean valid selected-key count;
- time spent in index selection, teacher KL, and sparse attention;
- allocated and reserved peak memory.

If `dsa_indexer_use_sparse_loss=True` is evaluated later, continue computing quality metrics against
the full dense teacher on diagnostic steps. The current metric implementation intentionally skips
those measurements in sparse-loss mode.

## 9. Validation plan

Follow the repository unit-test launcher described in `skills/mcore-testing/SKILL.md`.

### 9.1 Numerical unit tests

Add tests for:

1. MQA, GQA, and MHA query-head-to-KV-head mapping.
2. Sparse output and Q/K/V gradients versus a direct sink-aware reference.
3. Sink gradient versus the direct reference.
4. Top-K covering every legal key matches dense attention within documented FP32/BF16 tolerance.
5. Exact legal-key set coverage separately from numeric parity.
6. Sink counted once with more than one top-K chunk.
7. Early causal rows with fewer than K legal keys and `-1` padding.
8. Packed THD conversation boundaries and variable sequence lengths.
9. Checkpoint recomputation through the DSA extra-kwargs boundary.

Do not assert identical top-K ordering when ReLU produces tied indexer scores. Compare selected sets
modulo ties and compare the resulting numerical output.

### 9.2 SFT and reduction tests

Add tests for:

1. padding invariance of indexer KL and gradients;
2. prompts included in indexer KL while excluded from LM loss;
3. assistant-only GPT-OSS target masking;
4. changing only the LM supervision mask does not change DSA normalization;
5. unequal CP real-row counts versus a single-rank concatenated reference;
6. different real-row counts across gradient-accumulation microbatches;
7. PP > 1, CP > 1, and microbatch-size > 1 together;
8. separate base/indexer LR groups under distributed optimizer and Megatron FSDP.

### 9.3 Phase transition test

Build and save a Phase 1 model, then build a Phase 2 model with the phase flags flipped. Load the
checkpoint without the Phase 1 optimizer and assert:

- identical model key sets;
- strict load success;
- base and indexer optimizer membership;
- nonzero base and indexer gradients after one SFT backward;
- expected LR endpoints and logged gradient norms.

This should be a distributed unit test. The existing functional checkpoint-resume test type reuses
the same model arguments and cannot represent a phase-flag transition.

### 9.4 Functional test

Add a small H100 functional case using conversation JSONL data and GPT-OSS geometry scaled down to a
testable model. Cover:

- GQA and learnable sinks;
- mixed standard/sliding and DSA layers;
- sparse top-K attention;
- TP, PP, and CP where practical;
- checkpoint save/resume within Phase 2;
- SFT loss and DSA diagnostic logging.

Add the corresponding H100 recipe, trigger the new functional case, and commit golden values only
after a clean run.

## 10. Reproducible 20B rollout

Add scripts analogous to the Phase 1 launchers:

```text
local_setup/train_gpt_oss_20b_dsa_phase2_sft.sh
local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh
local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh
```

Every run directory must record:

- resolved launch and training commands;
- source commit and branch;
- dirty-worktree patch;
- Phase 1 checkpoint path and selected iteration;
- tokenizer and dataset paths;
- argument/environment metadata;
- rank logs and combined log;
- TensorBoard directory;
- final status and exit code;
- peak-memory and timing summaries.

Use the following gates:

| Gate | Configuration | Required result |
| --- | --- | --- |
| A | Sequence 128, K covers every legal key | Sparse/dense output and gradient parity; strict checkpoint load |
| B | Sequence 128, realistic K, 20 steps | Finite SFT/KL losses; nonzero base/indexer grads; correct LRs |
| C | Sequence 4K, realistic K, 100 steps | Stable memory; falling/finite loss; no abrupt recall or mass regression |
| D | Sequence 8K, realistic K | Measured throughput and peak memory; stable sparse sink mass |
| E | Longer SFT run | Short- and long-context eval within agreed dense-baseline tolerances |
| F | 32K candidate | Run only after the quadratic teacher/index-selection cost is acceptable or optimized |

At the Phase 1 exit and throughout Phase 2, evaluate indexer quality at the same K intended for
Phase 2. The Phase 1 KL itself is full-support and does not depend on K, but its exit metrics do.

## 11. Performance follow-up

The correctness implementation still performs quadratic index scoring, and full-support KL
recomputes quadratic teacher scores in blocks. Before treating DSA as a long-context speedup,
profile these components separately.

If teacher KL dominates, consider in order:

1. compute full-support KL only on configured steps while keeping top-K attention every step;
2. sample query rows for teacher KL while retaining prompts and assistant rows proportionally;
3. sample DSA layers for teacher KL with an unbiased schedule;
4. add a fused grouped-query teacher/indexer-loss backend;
5. add a fused GQA sparse-attention kernel.

Any sampling scheme must preserve a training signal for missed keys. Do not replace the teacher with
selected-support-only KL solely for speed without first demonstrating that recall and captured mass
remain stable.

## 12. Atomic delivery sequence

Keep the implementation reviewable as separate commits:

1. **Add sink-aware GQA sparse attention.** Grouped K/V mapping, sink online-softmax state, and
   numerical/gradient tests.
2. **Make packed SFT DSA loss padding-safe.** Real-token provenance, batch plumbing, CP-global DSA
   denominator, GPT-OSS assistant-target masking, and distributed tests.
3. **Add Phase 2 transition and diagnostics.** Strict checkpoint transition, optimizer membership,
   sparse sink/token-count metrics, and LR/gradient assertions.
4. **Add reproducible Phase 2 SFT recipes.** Launch/validation scripts, tiny functional dataset,
   functional recipe, and runbook.
5. **Optimize long-context execution if required.** Teacher-loss scheduling and fused kernels,
   supported by before/after profiles.

## 13. Definition of done

Phase 2 is ready for a sustained GPT-OSS 20B SFT run when:

- true 64-query-head/8-KV-head sparse attention runs without expanding full K/V by query head;
- sparse output and gradients match a sink-aware dense reference when K covers all legal keys;
- the learned sink is included exactly once and receives gradients;
- packed SFT padding contributes to neither indexer KL nor its denominator;
- prompt queries train the indexer while only intended assistant tokens train the LM objective;
- Phase 1 loads strictly into Phase 2 with identical model key sets;
- the rebuilt optimizer contains base and indexer parameters at their configured learning rates;
- base and indexer gradient norms are both finite and nonzero in a Phase 2 step;
- all quality metrics are available globally and per DSA layer;
- unit and functional tests pass using repository-standard launchers;
- the 20B 4K/8K smoke runs are reproducible from logged commands and remain within the measured
  memory budget;
- the remaining quadratic/fused-kernel limitation is explicitly accepted or resolved before 32K.
