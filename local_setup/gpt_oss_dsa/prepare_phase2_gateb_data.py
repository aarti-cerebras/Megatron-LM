"""Create a deterministic short-prefix Parquet view for the Phase 2 Gate B run."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible_rows(
    path: Path,
    message_token_id: int,
    terminator_token_ids: set[int],
    sequence_length: int,
    min_target_tokens: int,
) -> list[dict]:
    eligible = []
    row_offset = 0
    parquet_file = pq.ParquetFile(path)
    columns = ["input_ids", "prefix_tokens", "domain", "bucket", "length", "prompt_sha256"]
    for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
        rows = batch.to_pydict()
        for local_index, token_ids in enumerate(rows["input_ids"]):
            prefix = int(rows["prefix_tokens"][local_index])
            header_end = min(len(token_ids), prefix + 16)
            try:
                message_index = token_ids.index(message_token_id, prefix, header_end)
            except ValueError:
                continue
            first_target = message_index + 1
            block_end = next(
                (
                    index
                    for index in range(first_target, len(token_ids))
                    if token_ids[index] in terminator_token_ids
                ),
                None,
            )
            if block_end is None:
                continue
            last_visible_target = min(sequence_length, block_end)
            visible_selected_response_tokens = max(0, last_visible_target - first_target + 1)
            if visible_selected_response_tokens < min_target_tokens:
                continue
            eligible.append(
                {
                    "index": row_offset + local_index,
                    "domain": rows["domain"][local_index],
                    "bucket": rows["bucket"][local_index],
                    "length": int(rows["length"][local_index]),
                    "prompt_sha256": rows["prompt_sha256"][local_index],
                    "prefix_tokens": prefix,
                    "first_target": first_target,
                    "visible_selected_response_tokens": visible_selected_response_tokens,
                }
            )
        row_offset += batch.num_rows
    return eligible


def _stratified_sample(rows: list[dict], sample_count: int, seed: int) -> list[dict]:
    if sample_count <= 0 or sample_count >= len(rows):
        return sorted(rows, key=lambda row: row["index"])

    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    domains = sorted(by_domain)
    rng = np.random.default_rng(seed)
    selected = []
    for domain_index, domain in enumerate(domains):
        quota = sample_count // len(domains) + (domain_index < sample_count % len(domains))
        candidates = by_domain[domain]
        take = min(quota, len(candidates))
        selected.extend(
            candidates[index] for index in rng.choice(len(candidates), take, replace=False)
        )

    if len(selected) < sample_count:
        selected_indices = {row["index"] for row in selected}
        remaining = [row for row in rows if row["index"] not in selected_indices]
        take = min(sample_count - len(selected), len(remaining))
        selected.extend(
            remaining[index] for index in rng.choice(len(remaining), take, replace=False)
        )
    return sorted(selected, key=lambda row: row["index"])


def _write_rows(source_path: Path, output_path: Path, selected_rows: list[dict]) -> None:
    selected_indices = [row["index"] for row in selected_rows]
    selected_cursor = 0
    row_offset = 0
    writer = None
    try:
        for batch in pq.ParquetFile(source_path).iter_batches(batch_size=256):
            batch_end = row_offset + batch.num_rows
            local_indices = []
            while (
                selected_cursor < len(selected_indices)
                and selected_indices[selected_cursor] < batch_end
            ):
                local_indices.append(selected_indices[selected_cursor] - row_offset)
                selected_cursor += 1
            if local_indices:
                table = pa.Table.from_batches([batch]).take(pa.array(local_indices))
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
            row_offset = batch_end
    finally:
        if writer is not None:
            writer.close()
    if selected_cursor != len(selected_indices):
        raise RuntimeError(
            f"Only wrote {selected_cursor} of {len(selected_indices)} selected rows from {source_path}."
        )


def _selection_summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "tokens": sum(row["length"] for row in rows),
        "visible_selected_response_tokens": sum(
            row["visible_selected_response_tokens"] for row in rows
        ),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "buckets": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "max_prefix_tokens": max(row["prefix_tokens"] for row in rows),
        "min_visible_selected_response_tokens": min(
            row["visible_selected_response_tokens"] for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("tokenizer_path", type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--min-target-tokens", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.min_target_tokens <= 0 or args.min_target_tokens > args.sequence_length:
        raise ValueError("min-target-tokens must be in [1, sequence-length].")

    source_paths = {split: args.source_dir / f"{split}-00000.parquet" for split in ("train", "val")}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    message_token_id = tokenizer.convert_tokens_to_ids("<|message|>")
    if message_token_id is None or message_token_id < 0:
        raise ValueError("The tokenizer does not provide the GPT-OSS <|message|> token.")
    terminator_token_ids = {
        tokenizer.convert_tokens_to_ids(token) for token in ("<|end|>", "<|return|>", "<|call|>")
    }
    if any(token_id is None or token_id < 0 for token_id in terminator_token_ids):
        raise ValueError("The tokenizer does not provide all GPT-OSS message terminator tokens.")

    eligible = {
        split: _eligible_rows(
            path,
            message_token_id,
            terminator_token_ids,
            args.sequence_length,
            args.min_target_tokens,
        )
        for split, path in source_paths.items()
    }
    selected = {
        "train": _stratified_sample(eligible["train"], args.train_samples, args.seed),
        "val": _stratified_sample(eligible["val"], args.val_samples, args.seed + 1),
    }
    if not selected["train"] or not selected["val"]:
        raise RuntimeError("Gate B selection produced an empty train or validation split.")

    args.output_dir.mkdir(parents=True)
    output_paths = {split: args.output_dir / f"{split}-00000.parquet" for split in ("train", "val")}
    for split in ("train", "val"):
        _write_rows(source_paths[split], output_paths[split], selected[split])

    source_manifest = args.source_dir / "MANIFEST.json"
    manifest = {
        "kind": "gpt_oss_dsa_phase2_gateb_short_prefix_split",
        "source_dir": str(args.source_dir),
        "source_manifest_sha256": _sha256(source_manifest) if source_manifest.is_file() else None,
        "tokenizer_path": str(args.tokenizer_path),
        "sequence_length": args.sequence_length,
        "min_target_tokens": args.min_target_tokens,
        "seed": args.seed,
        "selection": {split: _selection_summary(rows) for split, rows in selected.items()},
        "eligible_rows": {split: len(rows) for split, rows in eligible.items()},
        "source_shards": {
            split: {"path": str(path), "sha256": _sha256(path)}
            for split, path in source_paths.items()
        },
        "output_shards": {
            split: {"path": path.name, "sha256": _sha256(path)}
            for split, path in output_paths.items()
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
