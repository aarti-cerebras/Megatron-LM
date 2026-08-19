# SFT Training Codebase Guide

This guide explains the main supervised fine-tuning (SFT) path in Megatron-LM,
with an emphasis on where data is loaded and processed, how dataloaders are
built, how the GPT model is constructed, and how training and optimizer setup
fit together.

The canonical GPT SFT path is a specialization of the normal Megatron training
path:

```text
JSONL conversations
  -> SFTTokenizer
  -> SFTDataset: tokenize, mask prompts, and pack sequences
  -> PyTorch DataLoader plus Megatron distributed sampler
  -> pretrain_gpt.forward_step()
  -> GPTModel
  -> masked next-token loss
  -> pipeline forward/backward schedule
  -> Megatron optimizer and learning-rate scheduler
```

## Recommended reading order

| Order | File | Purpose |
| --- | --- | --- |
| 1 | [`pretrain_gpt.py`](../../pretrain_gpt.py) | Main GPT/SFT entry point; connects datasets, batch handling, the model forward pass, and loss. |
| 2 | [`megatron/training/datasets/sft_dataset.py`](../../megatron/training/datasets/sft_dataset.py) | Loads conversation JSONL and converts each record into a packed training sample. |
| 3 | [`megatron/core/tokenizers/text/libraries/sft_tokenizer.py`](../../megatron/core/tokenizers/text/libraries/sft_tokenizer.py) | Applies chat templates and constructs prompt-masked targets. |
| 4 | [`megatron/training/datasets/data_samplers.py`](../../megatron/training/datasets/data_samplers.py) | Creates distributed samplers and the PyTorch `DataLoader`. |
| 5 | [`megatron/training/training.py`](../../megatron/training/training.py) | Global setup, training loop, pipeline scheduling, checkpointing, and optimizer steps. |
| 6 | [`megatron/training/models/gpt.py`](../../megatron/training/models/gpt.py) | Current GPT configuration and model-builder path. |
| 7 | [`megatron/core/models/gpt/gpt_model.py`](../../megatron/core/models/gpt/gpt_model.py) | Actual GPT model and forward pass. |
| 8 | [`megatron/core/optimizer/__init__.py`](../../megatron/core/optimizer/__init__.py) | Parameter groups and optimizer construction. |
| 9 | [`megatron/core/optimizer/optimizer_config.py`](../../megatron/core/optimizer/optimizer_config.py) | Adam, SGD, Muon, precision, clipping, and distributed-optimizer options. |

A concrete SFT configuration can be found in
[`tests/functional_tests/test_cases/hybrid/hybrid_nemotron_v3_pico_7b_a1b_tp1_ep8_QAD_dgx_h100_1N8G/model_config.yaml`](../../tests/functional_tests/test_cases/hybrid/hybrid_nemotron_v3_pico_7b_a1b_tp1_ep8_QAD_dgx_h100_1N8G/model_config.yaml).

## SFT data format

The canonical SFT loader expects JSONL. Each line must contain a `messages`
field holding a sequence of role/content dictionaries:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is tensor parallelism?"},
  {"role": "assistant", "content": "Tensor parallelism splits..."}
]}
```

`SFTLowLevelDataset` in
[`sft_dataset.py`](../../megatron/training/datasets/sft_dataset.py) loads this
file using Hugging Face Datasets:

```python
load_dataset("json", data_files=dataset_path, split="all")
```

Consequently, this path:

- reads JSONL directly;
- requires the Hugging Face `datasets` package;
- does not require Megatron `.bin` and `.idx` files;
- tokenizes examples online, normally inside dataloader workers.

The general [`tools/preprocess_data.py`](../../tools/preprocess_data.py) script
builds indexed `.bin`/`.idx` datasets for standard GPT pretraining. It is not
used by the canonical `SFTDataset` path.

The repository does not currently provide a general-purpose script that turns
arbitrary raw instruction datasets into this exact conversation JSONL contract.
Dataset-specific preparation therefore needs to produce the `messages` schema
shown above.

## Conversation tokenization and target masking

`SFTTokenizer.tokenize_conversation()` in
[`sft_tokenizer.py`](../../megatron/core/tokenizers/text/libraries/sft_tokenizer.py)
calls the Hugging Face tokenizer's `apply_chat_template()` method.

It produces:

- `tokens`: all tokens in the rendered conversation;
- `targets`: a copy of the token sequence in which non-target positions may be
  replaced with `-100` (`IGNORE_INDEX`).

For prompt formats such as `nemotron-h-aligned`, system, user, and tool tokens
are replaced with `-100`. Assistant answer tokens remain valid targets. The
later loss mask therefore trains only on assistant completions.

There is an important exception: the `default` and `identity` prompt formats do
not mask any prompt tokens. With either format, every token contributes to the
language-model loss. Prompt masking therefore depends on
`--sft-tokenizer-prompt-format`, not only on `--sft`.

The tokenizer is selected with arguments similar to:

```text
--tokenizer-type SFTTokenizer
--tokenizer-model /path/to/tokenizer
--sft-tokenizer-prompt-format nemotron-h-aligned
```

## Packing an SFT sample

`SFTDataset.__getitem__()` performs the following work:

1. Read one JSONL record.
2. Split the record into conversations at each new `system` message.
3. Tokenize every conversation.
4. Append conversations until reaching `seq_length + 1` tokens.
5. Truncate or pad to exactly `seq_length + 1`.
6. Shift the sequence to form next-token inputs and labels.
7. Mask padding and ignored prompt positions from the loss.
8. Return conversation boundaries for packed attention.

The next-token shift is:

```python
input_ids = pack_tokens[:-1]
labels = pack_targets[1:]
```

The loss mask is conceptually:

```python
loss_mask = ones(seq_length)
loss_mask[labels == pad] = 0
loss_mask[labels == IGNORE_INDEX] = 0
```

The dataset also constructs `cu_seqlens`, whose entries mark the boundaries of
individual conversations. Packed/variable-length attention uses these
boundaries to prevent one conversation from attending to another.

Packing happens among conversations already contained in one JSONL record. The
loader does not dynamically combine unrelated JSONL records into a pack. Data
preparation should therefore pre-pack multiple conversations into a record if
that behavior is desired.

For context parallelism, the dataset adds padding so token counts satisfy the
required context-parallel granularity. It finally returns fixed-size tensors:

```text
tokens
labels
loss_mask
position_ids
cu_seqlens
max_seqlen
```

## Building datasets and dataloaders

`train_valid_test_datasets_provider()` in
[`pretrain_gpt.py`](../../pretrain_gpt.py) selects the dataset implementation:

```python
if args.sft:
    dataset_type = SFTDataset
    is_packed_sequence = True
```

It then invokes `BlendedMegatronDatasetBuilder`, located in
[`megatron/core/datasets/blended_megatron_dataset_builder.py`](../../megatron/core/datasets/blended_megatron_dataset_builder.py).
The builder handles:

- train, validation, and test index splitting;
- `--data-path` versus independent `--train-data-path`,
  `--valid-data-path`, and `--test-data-path` inputs;
- weighted blending of multiple datasets;
- constructing the requested number of samples;
- coordinating dataset construction across distributed ranks.

The resulting datasets are passed to `build_pretraining_data_loader()` in
[`data_samplers.py`](../../megatron/training/datasets/data_samplers.py). It
constructs:

- a sequential or cyclic Megatron sampler;
- data-parallel sharding;
- resume offsets derived from `consumed_samples`;
- a standard `torch.utils.data.DataLoader`;
- worker processes, pinned memory, and persistent workers when configured.

Default PyTorch collation can stack SFT examples because `SFTDataset` pads all
returned tensors to fixed shapes.

### Distributed batch handling

`get_batch()` in [`pretrain_gpt.py`](../../pretrain_gpt.py) performs the next
stage of data handling:

1. Tensor-parallel rank zero advances the dataloader.
2. Batch tensors are transferred to CUDA.
3. Required tensors are broadcast to other tensor-parallel ranks.
4. Packed samples are flattened into THD layout.
5. Data is partitioned for context parallelism.
6. Packed-sequence metadata is sent to pipeline stages that require it.

The helpers implementing this logic are in
[`megatron/core/utils.py`](../../megatron/core/utils.py), particularly
`get_batch_on_this_tp_rank()`, `flatten_batch_for_packed_sequences()`, and
`get_batch_on_this_cp_rank()`.

## Program entry and configuration

The executable section of [`pretrain_gpt.py`](../../pretrain_gpt.py) performs
the high-level startup sequence:

1. Parse and validate command-line arguments.
2. Build a `GPTModelConfig` using `gpt_config_from_args()`.
3. Package model, optimizer, scheduler, distributed, checkpoint, and training
   configuration into a `PretrainConfigContainer`.
4. Call `megatron.training.training.pretrain()`.

The relevant configuration conversion functions are in
[`megatron/training/argument_utils.py`](../../megatron/training/argument_utils.py):

- `gpt_config_from_args()` constructs the GPT-specific model configuration;
- `pretrain_cfg_container_from_args()` constructs the full training
  configuration container.

`pretrain()` initializes distributed Megatron state, builds the model and
optimizer, loads a checkpoint if requested, builds the data iterators, and then
enters the training loop.

## GPT model construction

The active model-building path is `GPTModelConfig` plus `GPTModelBuilder` in
[`megatron/training/models/gpt.py`](../../megatron/training/models/gpt.py).

`GPTModelBuilder`:

- selects a Transformer Engine or local layer specification;
- selects dense, MoE, heterogeneous, experimental-attention, or custom
  `--spec` layers;
- pads the vocabulary for tensor parallelism when needed;
- determines which physical and virtual pipeline stages own embeddings and the
  output layer;
- constructs each `GPTModel` stage;
- wraps stages with mixed precision and DDP or FSDP.

The core GPT implementation lives in
[`megatron/core/models/gpt/gpt_model.py`](../../megatron/core/models/gpt/gpt_model.py).
For the decoder internals, continue with:

- [`megatron/core/transformer/transformer_block.py`](../../megatron/core/transformer/transformer_block.py)
- [`megatron/core/transformer/transformer_layer.py`](../../megatron/core/transformer/transformer_layer.py)
- [`megatron/core/transformer/attention.py`](../../megatron/core/transformer/attention.py)
- [`megatron/core/transformer/mlp.py`](../../megatron/core/transformer/mlp.py)

For Hybrid/Mamba models, start with
[`pretrain_hybrid.py`](../../pretrain_hybrid.py) and
[`megatron/training/models/hybrid.py`](../../megatron/training/models/hybrid.py).
Their SFT data and general training plumbing closely mirror the GPT path.

## Forward pass and SFT loss

`forward_step()` in [`pretrain_gpt.py`](../../pretrain_gpt.py) gets a batch,
constructs `PackedSeqParams` from `cu_seqlens`, and invokes the model:

```python
output_tensor = model(
    tokens,
    position_ids,
    attention_mask,
    labels=labels,
    loss_mask=loss_mask,
    packed_seq_params=packed_seq_params,
)
```

The model returns per-token language-model losses. `loss_func()` flattens those
losses, multiplies by `loss_mask`, and reports both:

- the sum of valid token losses;
- the number of valid target tokens.

The training machinery later reduces these values across the applicable
data/context-parallel group and reports total loss divided by the total number
of valid tokens. For a prompt-masking SFT format, valid targets are assistant
completion tokens rather than all tokens in the sequence.

## Main training loop

The important call chain is:

```text
training.pretrain()
  -> training.train()
    -> training.train_step()
      -> selected pipeline forward/backward schedule
        -> pretrain_gpt.forward_step()
          -> GPTModel.forward()
          -> masked language-model loss
      -> optimizer.step()
      -> learning-rate scheduler step
```

Pipeline schedules are selected by
[`megatron/core/pipeline_parallel/schedules.py`](../../megatron/core/pipeline_parallel/schedules.py).
Depending on the parallel configuration, the scheduler runs:

- no pipeline parallelism;
- a non-interleaved pipeline;
- an interleaved pipeline using virtual pipeline stages.

It also handles gradient accumulation over the configured number of
microbatches.

At a high level, each `train_step()` performs:

```text
zero gradient buffers
  -> pipeline forward/backward
  -> distributed gradient synchronization
  -> overflow and gradient-norm checks
  -> gradient clipping
  -> optimizer.step()
  -> advance scheduler after a successful update
```

## Optimizer construction

`setup_model_and_optimizer()` in
[`megatron/training/training.py`](../../megatron/training/training.py) builds the
model first and then constructs the optimizer and learning-rate scheduler:

```python
config, config_overrides = get_megatron_optimizer_config(args)
optimizer = get_megatron_optimizer(
    config,
    model,
    config_overrides=config_overrides,
)
opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
```

`OptimizerConfig` is defined in
[`optimizer_config.py`](../../megatron/core/optimizer/optimizer_config.py).
Common controls include:

```text
--optimizer
--lr
--min-lr
--adam-beta1
--adam-beta2
--adam-eps
--weight-decay
--clip-grad
--lr-decay-style
--lr-warmup-fraction
--use-distributed-optimizer
```

Adam is the default optimizer. With decoupled weight decay enabled, it behaves
as AdamW.

The standard parameter grouping logic in
[`megatron/core/optimizer/__init__.py`](../../megatron/core/optimizer/__init__.py)
does the following:

- excludes frozen parameters;
- gives bias and one-dimensional parameters zero weight decay;
- gives normal matrix weights the configured weight decay;
- optionally gives embedding/output parameters a separate `--decoupled-lr`;
- separates expert-parallel parameters from ordinary dense parameters;
- keeps parameter-group structure consistent across ranks for distributed
  checkpointing.

`--use-distributed-optimizer` shards optimizer state and associated parameter
storage across data-parallel ranks. It changes optimizer memory and
communication behavior, but does not change the SFT objective.

The scheduler can be iteration-based or sample-based. For iteration-based
training, decay and warmup iteration counts are converted to sample counts by
multiplying by the global batch size. After a successful optimizer update,
`train_step()` advances the scheduler by the number of samples consumed across
the data-parallel replicas and microbatches.

## `--sft` versus `--finetune`

These flags control different things and are easy to confuse:

- `--sft` selects `SFTDataset`, conversation tokenization, prompt loss masks,
  and packed attention.
- `--finetune` changes checkpoint-loading semantics. Model weights are loaded,
  but training restarts at iteration zero without restoring the old optimizer,
  learning-rate scheduler, consumed-data position, or RNG state.

A typical full SFT run initialized from pretrained weights therefore uses both:

```text
--sft
--finetune
--load /path/to/pretrained-checkpoint
--tokenizer-type SFTTokenizer
--tokenizer-model /path/to/tokenizer
--data-path /path/to/conversations.jsonl
```

If `--sft` is omitted, the run uses the standard GPT dataset path. If
`--finetune` is omitted while loading a normal training checkpoint, Megatron
attempts to resume its iteration, optimizer, scheduler, RNG, and consumed-data
state.

## Separate ModelOpt example path

[`examples/post_training/modelopt/finetune.py`](../../examples/post_training/modelopt/finetune.py)
contains another class named `SFTDataset`. That example supports direct Hugging
Face dataset loading and contains its own conversation conversion and packing
logic.

It should not be confused with the canonical
`megatron/training/datasets/sft_dataset.py` implementation selected by
`pretrain_gpt.py --sft`. When reading or modifying the code, first determine
which entry point the launch script executes.
