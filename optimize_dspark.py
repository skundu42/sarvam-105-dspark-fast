from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _classify_token(tok: str) -> str:
    s = tok.replace("▁", "").replace("Ġ", "").replace("Ċ", "")
    if not s:
        return "space"
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in s)
    if has_indic:
        return "indic"
    has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x30FF or 0xAC00 <= ord(c) <= 0xD7AF for c in s)
    if has_cjk:
        return "cjk"
    has_arab = any(0x0600 <= ord(c) <= 0x06FF for c in s)
    if has_arab:
        return "arab"
    has_cyr = any(0x0400 <= ord(c) <= 0x04FF for c in s)
    if has_cyr:
        return "cyr"
    if all(ord(c) < 128 for c in s):
        return "ascii"
    return "other"


def select_frspec_vocab_ids(
    tokenizer_json_path: Path,
    draft_vocab_size: int,
    vocab_size: int = 262144,
    prioritize_en_code_indic: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Select sorted target token IDs of length `draft_vocab_size` for lossless FR-Spec DSpark.

    Always preserves:
    - All special / added tokens (EOS, think tags, tool call tags, chat template tokens)
    - Base token range [0, 8192) (all UTF-8 bytes, ASCII, core multilingual BPE roots)
    - Highest-frequency BPE tokens prioritized for English, Code, Math, and Indic scripts.
    """
    data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    vocab: Dict[str, int] = data["model"]["vocab"]
    inv: Dict[int, str] = {idx: tok for tok, idx in vocab.items()}

    selected: Set[int] = set()

    # 1. Always keep all added / special tokens (chat template, <think>, tool calls, EOS/BOS)
    for item in data.get("added_tokens", []):
        tid = int(item["id"])
        if 0 <= tid < vocab_size:
            selected.add(tid)

    # 2. Always keep core first 8192 tokens (bytes, single chars, top multilingual roots)
    for tid in range(min(8192, vocab_size)):
        selected.add(tid)

    if not prioritize_en_code_indic:
        for tid in range(vocab_size):
            if len(selected) >= draft_vocab_size:
                break
            selected.add(tid)
    else:
        # 3. Prioritize English/Code/ASCII/Space/Math/Indic BPE tokens by merge rank (ascending ID)
        preferred = {"ascii", "indic", "space", "other"}
        for tid in range(vocab_size):
            if len(selected) >= draft_vocab_size:
                break
            if tid in selected:
                continue
            tok = inv.get(tid, "")
            if _classify_token(tok) in preferred:
                selected.add(tid)

        # 4. Fill any remaining slots in ascending BPE ID order
        for tid in range(vocab_size):
            if len(selected) >= draft_vocab_size:
                break
            selected.add(tid)

    sorted_ids = sorted(selected)[:draft_vocab_size]
    assert len(sorted_ids) == draft_vocab_size, f"Expected {draft_vocab_size}, got {len(sorted_ids)}"

    breakdown: Dict[str, int] = {}
    for tid in sorted_ids:
        k = _classify_token(inv.get(tid, ""))
        breakdown[k] = breakdown.get(k, 0) + 1

    return torch.tensor(sorted_ids, dtype=torch.int64), breakdown


def build_dspark_variant(
    src_model_dir: Path,
    out_dspark_dir: Path,
    draft_vocab_size: int | None = None,
    share_target_embed: bool = True,
    share_target_lm_head: bool = False,
    prioritize_en_code_indic: bool = True,
) -> Dict[str, object]:
    """Build a 100% lossless optimized DSpark draft checkpoint compatible with stock vLLM 0.34.0.

    Modes:
    - Shared full-vocab (`draft_vocab_size=None, share_target_embed=True, share_target_lm_head=True`):
      Omits duplicate 4.3GB BF16 embed_tokens & lm_head so vLLM aliases the target's copies.
    - FR-Spec shortlisted (`draft_vocab_size=32768 or 65536, share_target_embed=True`):
      Slices lm_head and markov_w2 to the selected shortlist and stores exact `d2t = selected_ids - arange(V_draft)`.
      Keeps markov_w1 at full target vocab_size (262144) and shares target embed_tokens.
    """
    t0 = time.perf_counter()
    src_dspark_dir = src_model_dir / "dspark"
    out_dspark_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (tokenizer/config metadata if any)
    for item in src_dspark_dir.iterdir():
        if item.name in ("model.safetensors", "config.json"):
            continue
        dst = out_dspark_dir / item.name
        if item.is_file():
            shutil.copy2(item, dst)

    cfg = json.loads((src_dspark_dir / "config.json").read_text(encoding="utf-8"))
    full_vocab = int(cfg.get("draft_vocab_size") or cfg["transformer_layer_config"]["vocab_size"])

    selected_ids: torch.Tensor | None = None
    breakdown: Dict[str, int] = {}
    if draft_vocab_size is not None and draft_vocab_size < full_vocab:
        assert not share_target_lm_head, "Cannot share full target lm_head when draft_vocab_size < vocab_size"
        selected_ids, breakdown = select_frspec_vocab_ids(
            src_model_dir / "tokenizer.json",
            draft_vocab_size=draft_vocab_size,
            vocab_size=full_vocab,
            prioritize_en_code_indic=prioritize_en_code_indic,
        )
        cfg["draft_vocab_size"] = int(draft_vocab_size)
        # Ensure transformer_layer_config keeps full vocab_size for embed_tokens & markov_w1
        cfg["transformer_layer_config"]["vocab_size"] = full_vocab
    else:
        cfg["draft_vocab_size"] = full_vocab

    (out_dspark_dir / "config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )

    out_tensors: Dict[str, torch.Tensor] = {}
    omitted: List[str] = []
    sliced: List[str] = []

    with safe_open(str(src_dspark_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            if share_target_embed and key == "embed_tokens.weight":
                omitted.append(key)
                continue
            if share_target_lm_head and key == "lm_head.weight":
                omitted.append(key)
                continue

            tensor = f.get_tensor(key)
            if selected_ids is not None and key in ("lm_head.weight", "markov_head.markov_w2.weight"):
                tensor = tensor.index_select(0, selected_ids).contiguous()
                sliced.append(f"{key} -> {list(tensor.shape)}")
            out_tensors[key] = tensor

    if selected_ids is not None:
        # Exact vLLM Qwen3DSparkForCausalLM contract:
        # map_draft_to_target(draft_ids) = draft_ids + d2t[draft_ids]
        d2t = (selected_ids - torch.arange(len(selected_ids), dtype=torch.int64)).contiguous()
        out_tensors["d2t"] = d2t

    out_st = out_dspark_dir / "model.safetensors"
    save_file(out_tensors, str(out_st), metadata=metadata)

    orig_mb = (src_dspark_dir / "model.safetensors").stat().st_size / 1e6
    new_mb = out_st.stat().st_size / 1e6
    summary = {
        "out_dir": str(out_dspark_dir),
        "draft_vocab_size": cfg["draft_vocab_size"],
        "orig_size_mb": round(orig_mb, 1),
        "new_size_mb": round(new_mb, 1),
        "size_reduction_x": round(orig_mb / max(new_mb, 1e-6), 2),
        "omitted": omitted,
        "sliced": sliced,
        "vocab_breakdown": breakdown,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    print(f"[optimize_dspark] Built {out_dspark_dir.name}: {json.dumps(summary, indent=2)}", flush=True)
    return summary
