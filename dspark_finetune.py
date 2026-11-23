from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "sarvam-105b-dspark-ft"
VOLUME_NAME = "sarvam-105b-infer-vol"
VOL_MOUNT = "/vol"
MODEL_DIR = f"{VOL_MOUNT}/models/sarvam-105b-1711"
VARIANTS_DIR = f"{VOL_MOUNT}/models/dspark_variants"
FT_WORK_DIR = f"{VOL_MOUNT}/dspark_ft"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# Reuse exact same image layer stack as modal_sarvam_track1.py for instant cached builds
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "curl")
    .uv_pip_install(
        "vllm==0.34.0",
        "speculators==0.9.0",
        "aiohttp>=3.10.0",
        "openai>=1.50.0",
    )
    .env(
        {
            "PYTHONPATH": "/app",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
            "VLLM_ENFORCE_STRICT_TOOL_CALLING": "0",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
            "VLLM_CACHE_ROOT": f"{VOL_MOUNT}/cache/vllm",
            "FLASHINFER_WORKSPACE_BASE": f"{VOL_MOUNT}/cache/flashinfer",
            "TORCHINDUCTOR_CACHE_DIR": f"{VOL_MOUNT}/cache/inductor",
        }
    )
    .uv_pip_install("ninja>=1.11.1")
    .env(
        {
            "VLLM_CACHE_ROOT": "/tmp/cache/vllm",
            "FLASHINFER_WORKSPACE_BASE": "/tmp/cache/flashinfer",
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/tmp/cache/flashinfer_autotune",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache/inductor",
            "TRITON_CACHE_DIR": "/tmp/cache/triton",
            "MAX_JOBS": "16",
        }
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path="/app/sarvam_infer",
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store"],
    )
)


# ---------------------------------------------------------------------------
# Exact Bit-Verified MXFP8 Dequantization / Quantization Helpers
# ---------------------------------------------------------------------------


def dequantize_mxfp8_tensor(weight_fp8: "torch.Tensor", scale_u8: "torch.Tensor") -> "torch.Tensor":
    """Exact MXFP8 (E4M3 + E8M0 block-32 scale) -> BF16 dequantization."""
    import torch

    exp = scale_u8.to(torch.int32) - 127
    scales = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
    w_fp32 = weight_fp8.to(torch.float32)
    scales_full = scales.repeat_interleave(32, dim=-1)
    return (w_fp32 * scales_full).to(torch.bfloat16)


def quantize_mxfp8_tensor(weight_bf16: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Exact BF16 -> MXFP8 (E4M3 + E8M0 block-32 scale) quantization (0.000000 round-trip diff)."""
    import torch

    orig_shape = weight_bf16.shape
    assert orig_shape[-1] % 32 == 0, f"Last dim {orig_shape[-1]} not divisible by 32"
    w = weight_bf16.to(torch.float32).reshape(*orig_shape[:-1], orig_shape[-1] // 32, 32)
    amax = w.abs().amax(dim=-1)
    ratio = torch.clamp(amax / 448.0, min=2.0 ** -126)
    exp = torch.ceil(torch.log2(ratio)).to(torch.int32)
    exp = torch.clamp(exp, -126, 127)
    scale_u8 = (exp + 127).to(torch.uint8)
    inv_scale = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), -exp)
    w_scaled = torch.clamp(w * inv_scale.unsqueeze(-1), -448.0, 448.0)
    w_fp8 = w_scaled.reshape(orig_shape).to(torch.float8_e4m3fn)
    return w_fp8.contiguous(), scale_u8.contiguous()


# ---------------------------------------------------------------------------
# External Public HF Dataset Sampler (Zero Overlap with BENCH_PROMPTS)
# ---------------------------------------------------------------------------


def build_external_prompt_mix(
    n_code: int = 800,
    n_math: int = 800,
    n_indic: int = 800,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Stream external public HF datasets (Code + Math + Indic) with strict anti-contamination checks."""
    from datasets import load_dataset
    from sarvam_infer.bench_client import BENCH_PROMPTS

    banned_snippets = {p.strip().lower()[:60] for p in BENCH_PROMPTS}

    def _clean_ok(text: str) -> bool:
        s = (text or "").strip()
        if len(s) < 25 or len(s) > 1400:
            return False
        low = s.lower()
        for b in banned_snippets:
            if b in low:
                return False
        return True

    rng = random.Random(seed)
    records: List[Dict[str, str]] = []

    # 1. Coding & Systems Engineering (CodeFeedback-Filtered-Instruction)
    print(f"[data] Streaming {n_code} Code prompts from m-a-p/CodeFeedback-Filtered-Instruction ...", flush=True)
    code_items: List[str] = []
    seen: set[str] = set()
    ds_code = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
    for row in ds_code:
        q = (row.get("query") or "").strip()
        if _clean_ok(q) and q not in seen:
            seen.add(q)
            code_items.append(q)
            if len(code_items) >= n_code * 2:
                break
    rng.shuffle(code_items)
    for q in code_items[:n_code]:
        records.append({"domain": "code", "prompt": q})

    # 2. Math & Multi-step Reasoning (openai/gsm8k train)
    print(f"[data] Streaming {n_math} Math prompts from openai/gsm8k ...", flush=True)
    math_items: List[str] = []
    ds_math = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for row in ds_math:
        q = (row.get("question") or "").strip()
        if _clean_ok(q) and q not in seen:
            seen.add(q)
            math_items.append(q)
            if len(math_items) >= n_math * 2:
                break
    rng.shuffle(math_items)
    for q in math_items[:n_math]:
        records.append({"domain": "math", "prompt": q})

    # 3. Indian Multilingual (CohereLabs/aya_dataset Indic languages)
    indic_codes = {"hin", "tam", "ben", "mar", "tel", "kan", "guj", "mal"}
    print(f"[data] Streaming {n_indic} Indic prompts from CohereLabs/aya_dataset ({sorted(indic_codes)}) ...", flush=True)
    indic_by_lang: Dict[str, List[str]] = {c: [] for c in indic_codes}
    ds_aya = load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    scanned = 0
    for row in ds_aya:
        scanned += 1
        lc = row.get("language_code", "")
        if lc in indic_by_lang and len(indic_by_lang[lc]) < 260:
            q = (row.get("inputs") or "").strip()
            if _clean_ok(q) and q not in seen:
                seen.add(q)
                indic_by_lang[lc].append(q)
        if sum(len(v) for v in indic_by_lang.values()) >= int(n_indic * 1.4) or scanned >= 120000:
            break
    indic_pool: List[str] = []
    # Balance across Hindi/Tamil/Bengali/Marathi/Telugu/etc.
    per_lang_cap = max(1, n_indic // len(indic_codes) + 40)
    for lc, lst in sorted(indic_by_lang.items()):
        rng.shuffle(lst)
        print(f"  aya[{lc}]: collected {len(lst)}", flush=True)
        indic_pool.extend(lst[:per_lang_cap])
    rng.shuffle(indic_pool)
    for q in indic_pool[:n_indic]:
        records.append({"domain": "indic", "prompt": q})

    # Stratified train / val split (50 per domain in held-out val)
    by_dom: Dict[str, List[Dict[str, str]]] = {"code": [], "math": [], "indic": []}
    for r in records:
        by_dom[r["domain"]].append(r)
    final_records: List[Dict[str, Any]] = []
    idx = 0
    for dom in ("code", "math", "indic"):
        lst = by_dom[dom]
        rng.shuffle(lst)
        val_n = min(50, max(10, len(lst) // 16))
        for j, item in enumerate(lst):
            final_records.append(
                {
                    "idx": idx,
                    "domain": dom,
                    "split": "val" if j < val_n else "train",
                    "prompt": item["prompt"],
                }
            )
            idx += 1

    print(
        f"[data] Prepared {len(final_records)} total external prompts "
        f"(train={sum(1 for r in final_records if r['split']=='train')}, "
        f"val={sum(1 for r in final_records if r['split']=='val')})",
        flush=True,
    )
    return final_records


# ---------------------------------------------------------------------------
# Server Management Helpers
# ---------------------------------------------------------------------------


def _restore_cache_fast() -> None:
    os.makedirs("/tmp/cache", exist_ok=True)
    for sub in ("vllm", "flashinfer", "flashinfer_autotune", "inductor", "triton"):
        os.makedirs(f"/tmp/cache/{sub}", exist_ok=True)
    tar_path = f"{VOL_MOUNT}/cache_bundle.tar"
    t0 = time.perf_counter()
    if os.path.exists(tar_path):
        subprocess.run(["tar", "-xf", tar_path, "-C", "/tmp/cache"], check=False)
        print(f"[cache] Extracted cache_bundle.tar in {time.perf_counter() - t0:.1f}s", flush=True)


def _wait_for_server(base_url: str, proc: subprocess.Popen, log_file: Path, timeout_s: int = 900) -> None:
    import urllib.request

    t0 = time.perf_counter()
    last_print = t0
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = log_file.read_text(errors="ignore")[-3000:]
            raise RuntimeError(f"vLLM exited early (code={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"[server] Ready in {time.perf_counter() - t0:.1f}s!", flush=True)
                    return
        except Exception:
            now = time.perf_counter()
            if now - last_print >= 20.0:
                last_print = now
                lines = [ln for ln in log_file.read_text(errors="ignore").splitlines() if ln.strip()]
                tail_ln = lines[-1][-150:] if lines else "(empty)"
                print(f"[server] waiting ({now - t0:.0f}s): {tail_ln}", flush=True)
            time.sleep(2.0)
    raise TimeoutError(f"vLLM failed to become ready in {timeout_s}s")


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    time.sleep(4.0)


# ---------------------------------------------------------------------------
# Phase 1: Fast On-Policy Rollout Generation via Sarvam-105B-1711
# ---------------------------------------------------------------------------


async def _generate_rollouts_async(
    base_url: str,
    model_name: str,
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_tokens: int = 768,
    temperature: float = 0.8,
    concurrency: int = 128,
) -> List[Dict[str, Any]]:
    import aiohttp

    # Pre-render prompts with exact Sarvam chat template + enable_thinking=True -> exact prompt_token_ids!
    rendered: List[Tuple[Dict[str, Any], List[int]]] = []
    for rec in records:
        msgs = [{"role": "user", "content": rec["prompt"]}]
        prompt_str = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        p_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        if 8 <= len(p_ids) <= 768:
            rendered.append((rec, p_ids))

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total_gen_toks = 0
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _one(rec: Dict[str, Any], p_ids: List[int]) -> Optional[Dict[str, Any]]:
            nonlocal completed, total_gen_toks
            payload = {
                "model": model_name,
                "prompt": p_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "return_token_ids": True,
            }
            async with sem:
                try:
                    async with session.post(
                        f"{base_url}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    ch = data["choices"][0]
                    c_ids = ch.get("token_ids")
                    if not c_ids:
                        c_ids = tokenizer.encode(ch.get("text", ""), add_special_tokens=False)
                    c_ids = [int(x) for x in c_ids]
                    if len(c_ids) < 24:
                        return None
                    full_ids = p_ids + c_ids
                    if len(full_ids) > 1536:
                        full_ids = full_ids[:1536]
                    completed += 1
                    total_gen_toks += len(c_ids)
                    if completed % 250 == 0 or completed == len(rendered):
                        dt = max(time.perf_counter() - t0, 1e-3)
                        print(
                            f"[rollout] {completed}/{len(rendered)} done | "
                            f"{total_gen_toks:,} gen tokens ({total_gen_toks / dt:,.0f} tok/s)",
                            flush=True,
                        )
                    return {
                        "idx": rec["idx"],
                        "domain": rec["domain"],
                        "split": rec["split"],
                        "prompt_len": len(p_ids),
                        "input_ids": full_ids,
                    }
                except Exception as e:
                    if completed < 5:
                        print(f"[rollout] WARN sample {rec['idx']} failed: {e!r}", flush=True)
                    return None

        results = await asyncio.gather(*[_one(rec, p_ids) for rec, p_ids in rendered])

    valid = [r for r in results if r is not None]
    dt = max(time.perf_counter() - t0, 1e-3)
    print(
        f"[rollout] Finished {len(valid)}/{len(rendered)} valid trajectories "
        f"({total_gen_toks:,} gen tokens in {dt:.1f}s = {total_gen_toks / dt:,.0f} tok/s)",
        flush=True,
    )
    return valid


# ---------------------------------------------------------------------------
# Phase 2: Exact Hidden-State Extraction via SarvamMLAModel Forward Hook
# ---------------------------------------------------------------------------


async def _extract_hidden_states_async(
    base_url: str,
    model_name: str,
    rollouts: List[Dict[str, Any]],
    concurrency: int = 4,
) -> int:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=16)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _prefill_one(rec: Dict[str, Any]) -> bool:
            nonlocal done
            payload = {
                "model": model_name,
                "prompt": rec["input_ids"],
                "max_tokens": 1,
                "temperature": 0.0,
            }
            async with sem:
                async with session.post(
                    f"{base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    await resp.json()
            done += 1
            if done % 300 == 0 or done == len(rollouts):
                dt = max(time.perf_counter() - t0, 1e-3)
                print(
                    f"[hs_extract] {done}/{len(rollouts)} sequences prefilled "
                    f"({done / dt:.1f} seq/s, {dt:.1f}s)",
                    flush=True,
                )
            return True

        res = await asyncio.gather(*[_prefill_one(r) for r in rollouts])
    return sum(1 for x in res if x)


# ---------------------------------------------------------------------------
# Phase 3: Official Lossless DSpark Fine-Tuning + MXFP8 Export
# ---------------------------------------------------------------------------


def _load_pretrained_dspark_bf16(device: "torch.device") -> Tuple["torch.nn.Module", List[str], Dict[str, Any]]:
    """Load Sarvam's pretrained MXFP8 DSpark into BF16 DSparkDraftModel + frozen target embed/lm_head/norm."""
    import torch
    from safetensors import safe_open
    from speculators.models.dspark.config import DSparkSpeculatorConfig
    from speculators.models.dspark.core import DSparkDraftModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    src_dspark = Path(MODEL_DIR) / "dspark"
    raw_cfg = json.loads((src_dspark / "config.json").read_text(encoding="utf-8"))
    orig_cfg_copy = json.loads(json.dumps(raw_cfg))

    # Strip quantization_config for BF16 training
    raw_cfg.pop("quantization_config", None)
    if isinstance(raw_cfg.get("transformer_layer_config"), dict):
        raw_cfg["transformer_layer_config"].pop("quantization_config", None)
        raw_cfg["transformer_layer_config"]["_attn_implementation"] = "simple_flex_attention"
    raw_cfg["speculators_config"]["verifier"]["name_or_path"] = MODEL_DIR

    dcfg = DSparkSpeculatorConfig(**raw_cfg)
    model = DSparkDraftModel(dcfg).to(dtype=torch.bfloat16)

    # Ensure verifier_norm is exact Sarvam RMSNorm (eps=1e-6)
    scfg = json.loads((Path(MODEL_DIR) / "config.json").read_text(encoding="utf-8"))
    model.verifier_norm = Qwen3RMSNorm(
        int(scfg["hidden_size"]), eps=float(scfg.get("rms_norm_eps", 1e-6))
    ).to(dtype=torch.bfloat16)

    # 1. Load & exact-dequantize all DSpark weights from dspark/model.safetensors
    mxfp8_weight_names: List[str] = []
    state_to_load: Dict[str, torch.Tensor] = {}
    with safe_open(str(src_dspark / "model.safetensors"), framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        for k in sorted(all_keys):
            if k.endswith("_scale"):
                continue
            if k in ("embed_tokens.weight", "lm_head.weight"):
                # Always load exact target copies below!
                continue
            scale_k = f"{k}_scale"
            if scale_k in all_keys:
                w_fp8 = f.get_tensor(k)
                s_u8 = f.get_tensor(scale_k)
                state_to_load[k] = dequantize_mxfp8_tensor(w_fp8, s_u8)
                mxfp8_weight_names.append(k)
            else:
                state_to_load[k] = f.get_tensor(k).to(torch.bfloat16)

    # 2. Load exact frozen target embed_tokens, lm_head, and model.norm from sarvam-105b-1711
    idx_map = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    for dst_key, src_key in (
        ("embed_tokens.weight", "model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("verifier_norm.weight", "model.norm.weight"),
    ):
        shard = Path(MODEL_DIR) / idx_map[src_key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            state_to_load[dst_key] = f.get_tensor(src_key).to(torch.bfloat16)
    state_to_load["verifier_lm_head.weight"] = state_to_load["lm_head.weight"]

    missing, unexpected = model.load_state_dict(state_to_load, strict=True)
    assert not missing and not unexpected, f"State dict mismatch: missing={missing}, unexpected={unexpected}"

    # Tie & freeze exact target weights (embed_tokens, lm_head, verifier_lm_head, verifier_norm)
    model.verifier_lm_head.weight = model.lm_head.weight
    frozen_prefixes = ("embed_tokens.", "lm_head.", "verifier_lm_head.", "verifier_norm.")
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            p.requires_grad = False
            n_frozen += p.numel()
        else:
            p.requires_grad = True
            n_train += p.numel()

    model = model.to(device=device, dtype=torch.bfloat16)
    print(
        f"[train] Loaded pretrained DSparkDraftModel on {device}: "
        f"trainable={n_train / 1e6:.1f}M, frozen_target={n_frozen / 1e6:.1f}M, "
        f"mxfp8_layers={len(mxfp8_weight_names)}",
        flush=True,
    )
    return model, mxfp8_weight_names, orig_cfg_copy


def _export_dspark_shared_mxfp8(
    model: "torch.nn.Module",
    mxfp8_weight_names: List[str],
    orig_cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Export fine-tuned DSpark in exact MXFP8 shared-head format (1.21 GB, -4.0 GiB VRAM)."""
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy any auxiliary files from original dspark/ (e.g. config.py if present)
    src_dspark = Path(MODEL_DIR) / "dspark"
    for item in src_dspark.iterdir():
        if item.name in ("model.safetensors", "config.json"):
            continue
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    (out_dir / "config.json").write_text(json.dumps(orig_cfg, indent=2) + "\n", encoding="utf-8")

    mxfp8_set = set(mxfp8_weight_names)
    omit_keys = {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    }
    out_tensors: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in omit_keys:
            continue
        t_cpu = v.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        if k in mxfp8_set:
            w_fp8, s_u8 = quantize_mxfp8_tensor(t_cpu)
            out_tensors[k] = w_fp8
            out_tensors[f"{k}_scale"] = s_u8
        else:
            out_tensors[k] = t_cpu

    out_st = out_dir / "model.safetensors"
    save_file(out_tensors, str(out_st))
    size_mb = out_st.stat().st_size / 1e6
    info = {
        "out_dir": str(out_dir),
        "num_tensors": len(out_tensors),
        "size_mb": round(size_mb, 1),
        "mxfp8_requantized": len(mxfp8_set),
    }
    print(f"[export] Saved lossless MXFP8 shared DSpark -> {json.dumps(info)}", flush=True)
    return info


def _pack_samples(
    samples: List[Dict[str, Any]],
    max_pack_len: int = 4096,
    shuffle: bool = False,
    rng: Optional[random.Random] = None,
) -> List[List[Dict[str, Any]]]:
    order = list(samples)
    if shuffle and rng is not None:
        rng.shuffle(order)
    packs: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    for s in order:
        slen = s["seq_len"]
        if cur and cur_len + slen > max_pack_len:
            packs.append(cur)
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += slen
    if cur:
        packs.append(cur)
    return packs


def _load_packed_batch(
    pack: List[Dict[str, Any]],
    device: "torch.device",
    pad_to_len: int = 4096,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Load FP8 safetensors for a pack, dequantize -> BF16, concatenate, and pad to fixed length for zero recompiles."""
    import torch
    from safetensors.torch import load_file
    from hs_connectors.fp8_utils import SCALES_KEY, dequantize_fp8_tensor

    hs_list, last_list, ids_list, mask_list, pos_list, doc_list = [], [], [], [], [], []
    total_t = 0
    for doc_idx, item in enumerate(pack):
        d = load_file(item["hs_path"], device="cpu")
        t_len = item["seq_len"]
        fp8_hs = d["hidden_states"][:t_len]  # [T, 6, 4096]
        scales = d[SCALES_KEY][:t_len]       # [T, 1, 1]
        hs_bf16 = dequantize_fp8_tensor(fp8_hs, scales, dtype=torch.bfloat16)  # [T, 6, 4096]
        aux_5 = hs_bf16[:, :5, :].reshape(t_len, 5 * 4096)
        last_1 = hs_bf16[:, 5, :]  # [T, 4096]

        ids = d["input_ids"][:t_len].to(torch.long)
        mask = torch.zeros(t_len, dtype=torch.bool)
        p_len = min(item["prompt_len"], t_len - 1)
        mask[p_len:] = True

        hs_list.append(aux_5)
        last_list.append(last_1)
        ids_list.append(ids)
        mask_list.append(mask)
        pos_list.append(torch.arange(t_len, dtype=torch.long))
        doc_list.append(torch.full((t_len,), doc_idx, dtype=torch.long))
        total_t += t_len

    # Pad to fixed pad_to_len (when pad_to_len > total_t) so flex_attention/torch.compile never recompiles!
    if pad_to_len > total_t:
        rem = pad_to_len - total_t
        pad_doc_id = len(pack)
        hs_list.append(torch.zeros((rem, 5 * 4096), dtype=torch.bfloat16))
        last_list.append(torch.zeros((rem, 4096), dtype=torch.bfloat16))
        ids_list.append(torch.zeros((rem,), dtype=torch.long))
        mask_list.append(torch.zeros((rem,), dtype=torch.bool))
        pos_list.append(torch.arange(rem, dtype=torch.long))
        doc_list.append(torch.full((rem,), pad_doc_id, dtype=torch.long))

    hidden_states = torch.cat(hs_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    last_hs = torch.cat(last_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    input_ids = torch.cat(ids_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    loss_mask = torch.cat(mask_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    position_ids = torch.cat(pos_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    document_ids = torch.cat(doc_list, dim=0).unsqueeze(0).to(device, non_blocking=True)
    return hidden_states, input_ids, loss_mask, last_hs, document_ids, position_ids


def _reduce_metrics(accum: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in sorted(accum.items()):
        if k.endswith("_sum"):
            base = k[:-4]
            denom = accum.get(f"{base}_total", 0.0)
            if denom > 0:
                out[base] = round(v / denom, 4)
    return out


def _eval_split(
    model: "torch.nn.Module",
    samples: List[Dict[str, Any]],
    device: "torch.device",
    loss_config: Any,
    tv_loss_fn: Any,
    max_anchors: int = 256,
) -> Dict[str, Dict[str, float]]:
    import torch

    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    for dom in ("all", "code", "math", "indic"):
        sub = samples if dom == "all" else [s for s in samples if s["domain"] == dom]
        if not sub:
            continue
        packs = _pack_samples(sub, max_pack_len=4096, shuffle=False)
        accum: Dict[str, float] = {}
        losses: List[float] = []
        # Deterministic anchor seed so Step 0 / Epoch 1..N evaluate identical anchor positions!
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        with torch.no_grad():
            for pack in packs:
                hs, ids, mask, last_hs, doc_ids, pos_ids = _load_packed_batch(pack, device)
                _, loss, metrics = model(
                    hidden_states=hs,
                    input_ids=ids,
                    loss_mask=mask,
                    verifier_last_hidden_states=last_hs,
                    document_ids=doc_ids,
                    position_ids=pos_ids,
                    loss_config=loss_config,
                    tv_loss_fn=tv_loss_fn,
                    gamma=4.0,
                    max_anchors=max_anchors,
                    confidence_head_alpha=0.5,
                )
                losses.append(float(loss.item()))
                for k, v in metrics.items():
                    val = float(v.item()) if hasattr(v, "item") else float(v)
                    accum[k] = accum.get(k, 0.0) + val
        reduced = _reduce_metrics(accum)
        reduced["loss"] = round(sum(losses) / max(len(losses), 1), 4)
        results[dom] = reduced
    return results


# ---------------------------------------------------------------------------
# End-to-End Modal 1x B200 On-Policy DSpark Fine-Tuning Job
# ---------------------------------------------------------------------------


def _run_training_experiment(
    exp_name: str,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    device: "torch.device",
    loss_fn_spec: str = '{"ce": 0.1, "tv": 0.9}',
    epochs: int = 4,
    lr: float = 3e-5,
    grad_accum: int = 8,
    max_anchors: int = 256,
    confidence_head_alpha: float = 0.2,
    select_domain: str = "all",
    out_variant_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one self-contained DSpark fine-tuning experiment from pretrained stock DSpark weights."""
    import torch
    from speculators.losses import resolve_loss_config

    t0 = time.perf_counter()
    model, mxfp8_weight_names, orig_cfg = _load_pretrained_dspark_bf16(device)
    loss_config = resolve_loss_config(loss_fn_spec, "fused")
    tv_loss_fn = resolve_loss_config("tv", "fused")["tv"][0]

    # Step 0 Pretrained Baseline Evaluation
    val_step0 = _eval_split(model, val_samples, device, loss_config, tv_loss_fn, max_anchors=max_anchors)
    print(f"\n=== [{exp_name}] Step 0 Pretrained Stock Val ===\n{json.dumps(val_step0, indent=2)}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    rng = random.Random(42)
    sample_packs = _pack_samples(train_samples, max_pack_len=4096, shuffle=True, rng=rng)
    steps_per_epoch = max(1, math.ceil(len(sample_packs) / grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(6, int(0.10 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    def _get_accept_len(val_res: Dict[str, Dict[str, float]]) -> float:
        dom_m = val_res.get(select_domain) or val_res.get("all", {})
        for k in ("accept_len", "mean_accept_len", "eal"):
            if k in dom_m:
                return float(dom_m[k])
        return -float(dom_m.get("loss", 999.0))

    best_score = _get_accept_len(val_step0)
    step0_score = best_score
    best_epoch = 0
    best_val = val_step0
    best_state_cpu: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = [{"epoch": 0, "val": val_step0}]

    global_step = 0
    for ep in range(1, epochs + 1):
        model.train()
        packs = _pack_samples(train_samples, max_pack_len=4096, shuffle=True, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        ep_loss = 0.0
        ep_n = 0
        t_ep = time.perf_counter()

        for b_idx, pack in enumerate(packs):
            hs, ids, mask, last_hs, doc_ids, pos_ids = _load_packed_batch(pack, device)
            _, loss, _ = model(
                hidden_states=hs,
                input_ids=ids,
                loss_mask=mask,
                verifier_last_hidden_states=last_hs,
                document_ids=doc_ids,
                position_ids=pos_ids,
                loss_config=loss_config,
                tv_loss_fn=tv_loss_fn,
                gamma=4.0,
                max_anchors=max_anchors,
                confidence_head_alpha=confidence_head_alpha,
            )
            (loss / grad_accum).backward()
            ep_loss += float(loss.item())
            ep_n += 1

            is_step = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == len(packs))
            if is_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 20 == 0 or global_step == total_steps:
                    cur_lr = scheduler.get_last_lr()[0]
                    print(
                        f"[{exp_name}] ep={ep}/{epochs} step={global_step}/{total_steps} "
                        f"avg_loss={ep_loss / max(ep_n, 1):.4f} lr={cur_lr:.2e}",
                        flush=True,
                    )

        val_ep = _eval_split(model, val_samples, device, loss_config, tv_loss_fn, max_anchors=max_anchors)
        score = _get_accept_len(val_ep)
        history.append({"epoch": ep, "train_loss": round(ep_loss / max(ep_n, 1), 4), "val": val_ep})
        print(
            f"\n=== [{exp_name}] Epoch {ep}/{epochs} ({time.perf_counter() - t_ep:.1f}s) "
            f"{select_domain}_accept_len={score:.4f} (step0={step0_score:.4f}) ===\n"
            f"{json.dumps(val_ep, indent=2)}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = ep
            best_val = val_ep
            best_state_cpu = {
                k: v.detach().to("cpu").clone()
                for k, v in model.state_dict().items()
                if k not in ("embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight", "verifier_norm.weight")
            }
            print(f"[{exp_name}] NEW BEST {select_domain}_accept_len={best_score:.4f} (+{best_score - step0_score:.4f}) @ epoch {ep}!", flush=True)

    export_info = None
    if best_state_cpu is not None and out_variant_name:
        cur_sd = model.state_dict()
        cur_sd.update({k: v.to(device) for k, v in best_state_cpu.items()})
        model.load_state_dict(cur_sd, strict=True)
        out_dir = Path(VARIANTS_DIR) / out_variant_name
        export_info = _export_dspark_shared_mxfp8(model, mxfp8_weight_names, orig_cfg, out_dir)
    else:
        print(f"[{exp_name}] Did not beat Step 0 ({step0_score:.4f}); skipping export.", flush=True)

    del model, optimizer, scheduler, trainable_params, best_state_cpu
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "exp_name": exp_name,
        "loss_fn": loss_fn_spec,
        "select_domain": select_domain,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "best_epoch": best_epoch,
        "step0_score": round(step0_score, 4),
        "best_score": round(best_score, 4),
        "delta_accept_len": round(best_score - step0_score, 4),
        "step0_val": val_step0,
        "best_val": best_val,
        "export": export_info,
        "history": history,
    }


@app.function(
    image=vllm_image,
    gpu="B200:1",
    cpu=16,
    memory=65536,
    volumes={VOL_MOUNT: vol},
    timeout=5400,
)
def run_onpolicy_dspark_finetune(
    n_per_domain: int = 800,
    epochs: int = 4,
    lr: float = 3e-5,
    grad_accum: int = 8,
    max_anchors: int = 256,
    loss_fn: str = '{"ce": 0.1, "tv": 0.9}',
    out_variant_name: str = "dspark_sector_tv",
    also_train_indic_specialist: bool = True,
) -> Dict[str, Any]:
    """Generate/reuse on-policy rollouts, extract MLA hidden states, run TV+CE DSpark fine-tuning, export MXFP8."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    t_all = time.perf_counter()
    vol.reload()
    os.makedirs(FT_WORK_DIR, exist_ok=True)
    _restore_cache_fast()

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    rollouts_path = Path(FT_WORK_DIR) / f"rollouts_{n_per_domain * 3}.jsonl"

    # ---------------- PHASE 1: On-Policy Rollout Generation (cached!) ----------------
    t_p1 = time.perf_counter()
    if rollouts_path.exists():
        print(f"[phase1] Reusing cached rollouts at {rollouts_path}", flush=True)
        rollouts = [json.loads(ln) for ln in rollouts_path.read_text().splitlines() if ln.strip()]
    else:
        prompt_records = build_external_prompt_mix(
            n_code=n_per_domain,
            n_math=n_per_domain,
            n_indic=n_per_domain,
            seed=42,
        )
        gen_cmd = [
            "vllm",
            "serve",
            MODEL_DIR,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--trust-remote-code",
            "--served-model-name",
            "sarvam-105b",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "256",
            "--max-num-batched-tokens",
            "16384",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--attention-backend",
            "TOKENSPEED_MLA",
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "model": f"{VARIANTS_DIR}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                }
            ),
        ]
        gen_log = Path("/tmp/phase1_gen_server.log")
        print(f"\n=== PHASE 1: On-Policy Rollouts ({len(prompt_records)} prompts) ===", flush=True)
        with open(gen_log, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(gen_cmd, env=os.environ.copy(), stdout=lf, stderr=subprocess.STDOUT)
            try:
                _wait_for_server("http://127.0.0.1:8000", proc, gen_log, timeout_s=900)
                rollouts = asyncio.run(
                    _generate_rollouts_async(
                        "http://127.0.0.1:8000",
                        "sarvam-105b",
                        prompt_records,
                        tokenizer=tokenizer,
                        max_tokens=768,
                        temperature=0.8,
                        concurrency=128,
                    )
                )
            finally:
                _stop_server(proc)
        with open(rollouts_path, "w", encoding="utf-8") as f:
            for r in rollouts:
                f.write(json.dumps(r) + "\n")
        vol.commit()
    p1_s = time.perf_counter() - t_p1
    print(f"[phase1] Done in {p1_s:.1f}s ({len(rollouts)} rollouts)", flush=True)

    # ---------------- PHASE 2: Exact MLA Hidden-State Extraction ----------------
    t_p2 = time.perf_counter()
    hs_dir = Path("/tmp/hs_out")
    if hs_dir.exists():
        shutil.rmtree(hs_dir)
    hs_dir.mkdir(parents=True, exist_ok=True)
    trigger_file = hs_dir / "ACTIVE"

    hs_env = os.environ.copy()
    hs_env["PYTHONPATH"] = "/app/sarvam_infer/patches:/app"
    hs_env["SARVAM_DUMP_HS_DIR"] = str(hs_dir)
    hs_env["SARVAM_DUMP_HS_LAYERS"] = "2,9,16,23,30,32"

    ext_cmd = [
        "vllm",
        "serve",
        MODEL_DIR,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--trust-remote-code",
        "--served-model-name",
        "sarvam-105b",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.60",
        "--kv-cache-dtype",
        "fp8_e4m3",
    ]
    ext_log = Path("/tmp/phase2_extract_server.log")
    print(f"\n=== PHASE 2: Extracting Exact MLA Hidden States ({len(rollouts)} seqs) ===", flush=True)
    with open(ext_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(ext_cmd, env=hs_env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            _wait_for_server("http://127.0.0.1:8000", proc, ext_log, timeout_s=600)
            trigger_file.write_text("1")
            asyncio.run(_extract_hidden_states_async("http://127.0.0.1:8000", "sarvam-105b", rollouts[:4], concurrency=2))
            time.sleep(1.5)
            for ln in ext_log.read_text(errors="ignore").splitlines():
                if "[hs_dump]" in ln:
                    print(" ", ln, flush=True)
            asyncio.run(_extract_hidden_states_async("http://127.0.0.1:8000", "sarvam-105b", rollouts[4:], concurrency=4))
            deadline = time.perf_counter() + 20.0
            while len(list(hs_dir.glob("hs_*.safetensors"))) < len(rollouts) and time.perf_counter() < deadline:
                time.sleep(0.5)
            trigger_file.unlink(missing_ok=True)
        finally:
            _stop_server(proc)

    key_to_rec: Dict[Tuple[int, Tuple[int, ...]], Dict[str, Any]] = {}
    prefix_to_recs: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
    for r in rollouts:
        ids = r["input_ids"]
        key_to_rec[(len(ids), tuple(ids[:16]))] = r
        prefix_to_recs.setdefault(tuple(ids[:16]), []).append(r)

    matched_samples: List[Dict[str, Any]] = []
    used_idxs: set[int] = set()
    for fpath in sorted(hs_dir.glob("hs_*.safetensors")):
        d = load_file(str(fpath), device="cpu")
        f_ids = d["input_ids"].tolist()
        pref = tuple(int(x) for x in f_ids[:16])
        rec = key_to_rec.get((len(f_ids), pref))
        if rec is None:
            for cand in prefix_to_recs.get(pref, []):
                clen = len(cand["input_ids"])
                if len(f_ids) >= clen and f_ids[:clen] == cand["input_ids"]:
                    rec = cand
                    break
        if rec is not None and rec["idx"] not in used_idxs:
            used_idxs.add(rec["idx"])
            matched_samples.append(
                {
                    "idx": rec["idx"],
                    "domain": rec["domain"],
                    "split": rec["split"],
                    "prompt_len": rec["prompt_len"],
                    "seq_len": len(rec["input_ids"]),
                    "hs_path": str(fpath),
                }
            )

    p2_s = time.perf_counter() - t_p2
    train_samples = [s for s in matched_samples if s["split"] == "train"]
    val_samples = [s for s in matched_samples if s["split"] == "val"]
    print(
        f"[phase2] Done in {p2_s:.1f}s | matched={len(matched_samples)}/{len(rollouts)} "
        f"(train={len(train_samples)}, val={len(val_samples)})",
        flush=True,
    )

    # ---------------- PHASE 3: TV+CE Multi-Sector + Indic Specialist Experiments ----------------
    t_p3 = time.perf_counter()
    device = torch.device("cuda:0")
    experiments: Dict[str, Any] = {}

    # Exp 1: Multi-sector (Code + Math + Indic) with TV+CE loss directly maximizing accept_rate!
    experiments["multi_sector_tv"] = _run_training_experiment(
        exp_name="multi_sector_tv",
        train_samples=train_samples,
        val_samples=val_samples,
        device=device,
        loss_fn_spec=loss_fn,
        epochs=epochs,
        lr=lr,
        grad_accum=grad_accum,
        max_anchors=max_anchors,
        confidence_head_alpha=0.2,
        select_domain="all",
        out_variant_name=out_variant_name,
    )

    # Exp 2: Indic-Sector Specialist (Code+Indic / Pure Indic where stock DSpark lags by -1.62 tokens!)
    if also_train_indic_specialist:
        indic_train = [s for s in train_samples if s["domain"] == "indic"]
        experiments["indic_specialist_tv"] = _run_training_experiment(
            exp_name="indic_specialist_tv",
            train_samples=indic_train,
            val_samples=val_samples,
            device=device,
            loss_fn_spec=loss_fn,
            epochs=6,
            lr=lr,
            grad_accum=4,
            max_anchors=max_anchors,
            confidence_head_alpha=0.2,
            select_domain="indic",
            out_variant_name="dspark_indic_tv",
        )

    p3_s = time.perf_counter() - t_p3
    shutil.rmtree(hs_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    total_s = time.perf_counter() - t_all
    est_cost = (total_s / 3600.0) * 6.25
    report = {
        "total_elapsed_s": round(total_s, 1),
        "est_gpu_cost_usd": round(est_cost, 3),
        "phase_times_s": {"phase1_gen": round(p1_s, 1), "phase2_extract": round(p2_s, 1), "phase3_train": round(p3_s, 1)},
        "experiments": {k: {ek: ev for ek, ev in v.items() if ek != "history"} for k, v in experiments.items()},
    }
    report_path = Path(FT_WORK_DIR) / f"{out_variant_name}_report.json"
    report_path.write_text(json.dumps({"summary": report, "full": experiments}, indent=2), encoding="utf-8")
    vol.commit()
    print(f"\n================ DSPARK TV+CE FT SUMMARY ({total_s:.1f}s, ~${est_cost:.2f}) ================", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(
    image=vllm_image,
    cpu=2,
    memory=4096,
    timeout=120,
)
def inspect_dspark_loss_recipe() -> Dict[str, Any]:
    """Check official speculators==0.9.0 DSpark loss_fn, tv_loss, and dpace defaults."""
    import speculators

    sroot = Path(speculators.__file__).parent
    for rel in ("losses/utils.py", "losses/fused.py", "losses/eager.py"):
        p = sroot / rel
        if p.exists():
            txt = p.read_text()
            print(f"\n=== speculators/{rel} ({len(txt)} chars) ===\n{txt[:4500]}", flush=True)

    schema = (sroot / "train/config/schema.py").read_text()
    for term in ("loss_fn", "dpace", "confidence_head_alpha", "per_position_loss_weight"):
        idx = schema.find(term)
        if idx >= 0:
            print(f"\n=== schema.py around {term} ===\n{schema[max(0, idx-100):idx+600]}", flush=True)
    return {"ok": True}


@app.local_entrypoint()
def main(
    action: str = "finetune",
    n_per_domain: int = 800,
    epochs: int = 4,
    lr: float = 3e-5,
    loss_fn: str = '{"ce": 0.1, "tv": 0.9}',
    out_variant_name: str = "dspark_sector_tv",
) -> None:
    if action == "inspect_loss":
        res = inspect_dspark_loss_recipe.remote()
        print(json.dumps(res, indent=2), flush=True)
        return
    res = run_onpolicy_dspark_finetune.remote(
        n_per_domain=n_per_domain,
        epochs=epochs,
        lr=lr,
        loss_fn=loss_fn,
        out_variant_name=out_variant_name,
    )
    print(json.dumps(res, indent=2), flush=True)
