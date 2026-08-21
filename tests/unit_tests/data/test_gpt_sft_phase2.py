# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer
from megatron.core.utils import _get_batch_on_this_cp_rank_per_document_balancing
from megatron.training.datasets.sft_dataset import SFTDataset, SFTLowLevelDataset


class _DatasetTokenizer:
    pad = 99
    eod = 98

    @staticmethod
    def tokenize_conversation(conversation, return_target, add_generation_prompt):
        assert conversation[0]["role"] == "system"
        assert return_target is True
        assert add_generation_prompt is False
        # Both the second prompt token and fourth assistant token deliberately equal the pad ID.
        # Provenance, rather than token identity, must decide whether each occurrence is padding.
        tokens = np.array([10, 99, 12, 99, 13], dtype=np.int64)
        targets = np.array([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 99, 13], dtype=np.int64)
        return tokens, targets


class _PretokenizedTokenizer:
    pad = 99
    eod = 98
    vocab_size = 128

    @staticmethod
    def build_targets_from_token_ids(tokens):
        targets = np.full_like(tokens, IGNORE_INDEX)
        targets[3:] = tokens[3:]
        return targets


class _AllTargetTokenizer:
    pad = 99
    eod = 98

    @staticmethod
    def tokenize_conversation(conversation, return_target, add_generation_prompt):
        assert conversation[0]["role"] == "system"
        assert return_target is True
        assert add_generation_prompt is False
        tokens = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
        return tokens, tokens.copy()


class _Rows(list):
    def __init__(self, rows, column_names):
        super().__init__(rows)
        self.column_names = column_names


class _HarmonyTokenizer:
    token_ids = {"<|start|>": 1, "<|message|>": 2, "<|end|>": 3, "<|return|>": 4, "<|call|>": 5}
    assistant_ids = [6]

    def __init__(self, rendered_tokens):
        self.rendered_tokens = np.asarray(rendered_tokens, dtype=np.int64)

    def convert_tokens_to_ids(self, token):
        return self.token_ids.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        assert text == "assistant"
        assert add_special_tokens is False
        return self.assistant_ids

    def apply_chat_template(self, conversation, **kwargs):
        assert conversation
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["return_tensors"] == "np"
        assert kwargs["chat_template"] == "native-harmony-template"
        return self.rendered_tokens[np.newaxis, :]


def test_sft_dataset_tracks_real_tokens_from_provenance():
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ]
    ]
    dataset.indices = np.array([0], dtype=np.int64)
    dataset.config = SimpleNamespace(
        tokenizer=_DatasetTokenizer(),
        sequence_length=8,
        reset_position_ids=False,
        context_parallel_size=1,
        create_attention_mask=False,
        reset_attention_mask=False,
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["real_token_mask"],
        torch.tensor([True, True, True, True, True, False, False, False]),
    )
    assert sample["tokens"][1].item() == _DatasetTokenizer.pad
    assert sample["real_token_mask"][1].item() is True
    # Prompt rows remain real DSA queries even though they are not LM targets.
    assert sample["real_token_mask"][0].item() is True
    assert sample["loss_mask"][0].item() == 0.0
    # A genuine assistant target equal to the pad ID remains supervised. Synthetic padding does
    # not, even though both have the same token ID.
    assert sample["labels"][2].item() == _DatasetTokenizer.pad
    assert sample["loss_mask"][2].item() == 1.0
    assert sample["labels"][4].item() == _DatasetTokenizer.pad
    assert sample["loss_mask"][4].item() == 0.0


def test_sft_low_level_dataset_loads_pretokenized_parquet():
    rows = _Rows(
        [{"input_ids": [10, 11], "loss_mask": [0, 1], "length": 2}],
        ["input_ids", "loss_mask", "length"],
    )
    with patch("datasets.load_dataset", return_value=rows) as load_dataset:
        dataset = SFTLowLevelDataset("train-00000.parquet")

    load_dataset.assert_called_once_with("parquet", data_files="train-00000.parquet", split="all")
    assert dataset.is_pretokenized
    assert dataset[0] == rows[0]


def test_sft_low_level_dataset_rejects_incomplete_parquet_schema():
    rows = _Rows([{"input_ids": [10, 11], "length": 2}], ["input_ids", "length"])
    with (
        patch("datasets.load_dataset", return_value=rows),
        pytest.raises(ValueError, match="loss_mask"),
    ):
        SFTLowLevelDataset("train-00000.parquet")


def test_sft_dataset_consumes_pretokenized_tokens_without_retokenizing():
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [
        {
            "input_ids": [10, 11, 12, 13, 14, 15],
            # The source also supervises one generated assistant-header token. Canonical
            # GPT-OSS reconstruction below intentionally keeps payloads and terminators only.
            "loss_mask": [0, 0, 1, 1, 1, 1],
            "length": 6,
        }
    ]
    dataset.indices = np.array([0], dtype=np.int64)
    dataset.config = SimpleNamespace(
        tokenizer=_PretokenizedTokenizer(),
        sequence_length=8,
        reset_position_ids=False,
        context_parallel_size=1,
        create_attention_mask=False,
        reset_attention_mask=False,
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["tokens"], torch.tensor([10, 11, 12, 13, 14, 15, 99, 99]))
    torch.testing.assert_close(
        sample["labels"], torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 13, 14, 15, 99, 99, 99])
    )
    torch.testing.assert_close(
        sample["loss_mask"], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        sample["real_token_mask"], torch.tensor([True, True, True, True, True, True, False, False])
    )
    assert sample["cu_seqlens"][:2].tolist() == [0, 8]


def test_sft_dataset_rejects_parquet_mask_missing_canonical_targets():
    record = {"input_ids": [10, 11, 12, 13, 14, 15], "loss_mask": [0, 0, 1, 1, 0, 1], "length": 6}
    with pytest.raises(ValueError, match="excludes 1 canonical assistant targets"):
        SFTDataset._prepare_pretokenized_record(record, _PretokenizedTokenizer())


def test_sft_dataset_truncation_preserves_the_next_token_target():
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ]
    ]
    dataset.indices = np.array([0], dtype=np.int64)
    dataset.config = SimpleNamespace(
        tokenizer=_AllTargetTokenizer(),
        sequence_length=4,
        reset_position_ids=False,
        context_parallel_size=1,
        create_attention_mask=False,
        reset_attention_mask=False,
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["tokens"], torch.tensor([10, 11, 12, 13]))
    torch.testing.assert_close(sample["labels"], torch.tensor([11, 12, 13, 14]))
    torch.testing.assert_close(sample["loss_mask"], torch.ones(4))
    torch.testing.assert_close(sample["real_token_mask"], torch.ones(4, dtype=torch.bool))


def test_gpt_oss_masks_non_assistant_blocks_and_keeps_all_assistant_channels():
    start, message, end, return_token, call = 1, 2, 3, 4, 5
    assistant, system, user, tool, developer = 6, 7, 8, 9, 14
    channel, analysis, final, recipient = 10, 11, 12, 13
    rendered = [
        start,
        system,
        message,
        20,
        end,
        start,
        developer,
        message,
        22,
        end,
        start,
        user,
        message,
        21,
        end,
        start,
        assistant,
        channel,
        analysis,
        message,
        30,
        31,
        end,
        start,
        assistant,
        channel,
        final,
        message,
        40,
        return_token,
        start,
        assistant,
        recipient,
        message,
        50,
        call,
        start,
        tool,
        message,
        60,
        end,
    ]
    tokenizer = SFTTokenizer.__new__(SFTTokenizer)
    tokenizer._tokenizer = _HarmonyTokenizer(rendered)
    tokenizer._prompt_format = "gpt-oss"
    tokenizer._prompt_config = SimpleNamespace(
        has_system_role=True, custom_chat_template="native-harmony-template"
    )

    tokens, targets = tokenizer.tokenize_conversation(
        [{"role": "user", "content": "ignored by fake tokenizer"}],
        return_target=True,
        add_generation_prompt=False,
    )

    expected = np.full(len(rendered), IGNORE_INDEX, dtype=np.int64)
    expected[20:23] = [30, 31, end]
    expected[28:30] = [40, return_token]
    expected[34:36] = [50, call]
    np.testing.assert_array_equal(tokens, np.asarray(rendered))
    np.testing.assert_array_equal(targets, expected)


def test_gpt_oss_rejects_unterminated_assistant_message():
    tokenizer = SFTTokenizer.__new__(SFTTokenizer)
    tokenizer._tokenizer = _HarmonyTokenizer([1, 6, 2, 30])

    with pytest.raises(ValueError, match="missing its terminator"):
        tokenizer._mask_gpt_oss_assistant_targets(tokenizer._tokenizer.rendered_tokens)


def test_gpt_oss_native_tokenizer_masks_conversation_roles_and_assistant_channels():
    tokenizer_path = os.environ.get("GPT_OSS_TOKENIZER_PATH")
    if tokenizer_path is None:
        pytest.skip("GPT_OSS_TOKENIZER_PATH is required for native tokenizer validation")

    tokenizer = SFTTokenizer(tokenizer_path, "gpt-oss")

    def validate(conversation, required_text, forbidden_text, required_terminators):
        tokens, targets = tokenizer.tokenize_conversation(
            conversation, return_target=True, add_generation_prompt=False
        )
        supervised = targets != IGNORE_INDEX
        assert supervised.any()
        np.testing.assert_array_equal(targets[supervised], tokens[supervised])

        supervised_text = tokenizer._tokenizer.decode(
            targets[supervised].tolist(), skip_special_tokens=False
        )
        for text in required_text:
            assert text in supervised_text
        for text in forbidden_text:
            assert text not in supervised_text
        for terminator in required_terminators:
            terminator_id = tokenizer._tokenizer.convert_tokens_to_ids(terminator)
            assert np.count_nonzero(targets[supervised] == terminator_id) > 0

    validate(
        [
            {"role": "developer", "content": "developer-private-text"},
            {"role": "user", "content": "user-private-text"},
            {
                "role": "assistant",
                "thinking": "assistant-analysis-text",
                "content": "assistant-final-text",
            },
        ],
        required_text=("assistant-analysis-text", "assistant-final-text"),
        forbidden_text=("developer-private-text", "user-private-text"),
        required_terminators=("<|end|>", "<|return|>"),
    )

    validate(
        [
            {"role": "user", "content": "tool-user-private-text"},
            {
                "role": "assistant",
                "thinking": "tool-analysis-text",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
        ],
        required_text=("tool-analysis-text", "Paris"),
        forbidden_text=("tool-user-private-text",),
        required_terminators=("<|end|>", "<|call|>"),
    )

    validate(
        [
            {"role": "user", "content": "round-trip-user-private-text"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "tool-result-private-text"},
            {
                "role": "assistant",
                "thinking": "round-trip-analysis-text",
                "content": "round-trip-final-text",
            },
        ],
        required_text=("Paris", "round-trip-analysis-text", "round-trip-final-text"),
        forbidden_text=("round-trip-user-private-text", "tool-result-private-text"),
        required_terminators=("<|call|>", "<|end|>", "<|return|>"),
    )


def test_intermediate_pp_real_token_mask_uses_document_cp_partition():
    real_token_mask = torch.tensor([[True, True, False, False, True, False, True, False]])
    batch = {
        "tokens": None,
        "labels": None,
        "loss_mask": None,
        "position_ids": None,
        "real_token_mask": real_token_mask.clone(),
        "cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
        "cu_seqlens_padded": None,
        "max_seqlen": torch.tensor([4], dtype=torch.int32),
    }
    selected_indices = torch.tensor([1, 2, 5, 6], dtype=torch.int64)

    with (
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=1),
        patch("megatron.core.utils.tex.thd_get_partitioned_indices", return_value=selected_indices),
    ):
        result = _get_batch_on_this_cp_rank_per_document_balancing(batch, cp_group=object())

    torch.testing.assert_close(
        result["real_token_mask"], real_token_mask.index_select(1, selected_indices)
    )
    assert result["tokens"] is None
