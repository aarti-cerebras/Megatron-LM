# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit
import json
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """Load conversation JSONL or pretokenized Parquet data for SFT.

    Args:
        dataset_path (str): The path to JSONL conversation data or pretokenized Parquet data.
            Parquet files must contain ``input_ids``, ``loss_mask``, and ``length`` columns.
            The stored mask is validated against targets reconstructed by the configured SFT
            tokenizer; token IDs are not decoded and re-tokenized.

            JSONL data must contain a ``messages`` key (List[Dict]), which is a sequence of
            system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("SFTDataset currently requires datasets library to be installed")
        self.is_pretokenized = dataset_path.lower().endswith(".parquet")
        dataset_format = "parquet" if self.is_pretokenized else "json"
        self.dataset = load_dataset(dataset_format, data_files=dataset_path, split="all")
        required_columns = (
            {"input_ids", "loss_mask", "length"} if self.is_pretokenized else {"messages"}
        )
        missing_columns = required_columns - set(self.dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"SFT {dataset_format} data at {dataset_path!r} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list | dict:
        row = self.dataset[idx]
        return row if self.is_pretokenized else row["messages"]


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg)  # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    @staticmethod
    def _prepare_pretokenized_record(record, tokenizer):
        tokens = np.asarray(record["input_ids"], dtype=np.int64)
        provided_loss_mask = np.asarray(record["loss_mask"])
        if tokens.ndim != 1 or provided_loss_mask.ndim != 1:
            raise ValueError("Pretokenized SFT input_ids and loss_mask must be one-dimensional.")
        if len(tokens) < 2:
            raise ValueError("Pretokenized SFT records must contain at least two token IDs.")
        if len(tokens) != len(provided_loss_mask):
            raise ValueError(
                "Pretokenized SFT input_ids and loss_mask lengths differ: "
                f"{len(tokens)} != {len(provided_loss_mask)}."
            )
        if int(record["length"]) != len(tokens):
            raise ValueError(
                "Pretokenized SFT length metadata does not match input_ids: "
                f"{record['length']} != {len(tokens)}."
            )
        if not np.isin(provided_loss_mask, (0, 1)).all():
            raise ValueError("Pretokenized SFT loss_mask must contain only zero and one values.")
        if tokens.min() < 0 or tokens.max() >= tokenizer.vocab_size:
            raise ValueError(
                "Pretokenized SFT token IDs fall outside the configured tokenizer vocabulary: "
                f"range=[{tokens.min()}, {tokens.max()}], vocab_size={tokenizer.vocab_size}."
            )

        targets = tokenizer.build_targets_from_token_ids(tokens)
        canonical_loss_mask = targets != IGNORE_INDEX
        missing_targets = canonical_loss_mask & ~provided_loss_mask.astype(bool)
        if missing_targets.any():
            raise ValueError(
                "Pretokenized SFT loss_mask excludes "
                f"{int(missing_targets.sum())} canonical assistant targets."
            )
        return tokens, targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        record = self.dataset[int(self.indices[idx % len(self.indices)])]
        records = [record] if isinstance(record, dict) else self._split_conversations(record)

        def extend_with_padding(tokens, targets, positions, real_token_mask, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1] + 1, positions[-1] + 1 + pad_len))
            real_token_mask.extend([False] * pad_len)

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        pack_real_token_mask = []
        cu_seqlens = [0]
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for record in records:

            if isinstance(record, dict):
                tokens, targets = self._prepare_pretokenized_record(record, tokenizer)
            else:
                tokens, targets = tokenizer.tokenize_conversation(
                    record, return_target=True, add_generation_prompt=False
                )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)
            pack_real_token_mask.extend([True] * len(tokens_list))

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(
                        pack_tokens, pack_targets, pack_positions, pack_real_token_mask, pad_len
                    )

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                pack_tokens = pack_tokens[: pack_length + 1]
                pack_targets = pack_targets[: pack_length + 1]
                pack_real_token_mask = pack_real_token_mask[: pack_length + 1]
                pack_positions = pack_positions[: pack_length + 1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(
                pack_tokens, pack_targets, pack_positions, pack_real_token_mask, pad_len
            )
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1
        assert len(pack_real_token_mask) == pack_length + 1

        # Align and convert to tensors
        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)
        real_token_mask = torch.tensor(pack_real_token_mask[:-1], dtype=torch.bool)
        target_real_token_mask = torch.tensor(pack_real_token_mask[1:], dtype=torch.bool)

        # Loss mask.
        # Tokenizers may reuse a real end-of-text token as padding, so token IDs cannot
        # distinguish genuine assistant targets from synthetic tail padding. Use the shifted
        # provenance mask for target validity and the tokenizer target mask for supervision.
        loss_mask = (target_real_token_mask & (labels != IGNORE_INDEX)).float()

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        # Pad cu_seqlens to a fixed length so that default_collate can
        # stack samples with different numbers of documents.  Trailing
        # entries are filled with pack_length; the merge helper strips
        # them later.
        padded_cu_seqlens = torch.full((pack_length + 1,), pack_length, dtype=torch.int32)
        padded_cu_seqlens[: cu_seqlens.numel()] = cu_seqlens

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'real_token_mask': real_token_mask,
            'cu_seqlens': padded_cu_seqlens,
            'max_seqlen': max_seqlen,
        }
