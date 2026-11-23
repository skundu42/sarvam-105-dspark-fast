from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

APP_NAME = "sarvam-105b-track1"
VOLUME_NAME = "sarvam-105b-infer-vol"
VOL_MOUNT = "/vol"
MODEL_REPO = "sarvamai/sarvam-105b-1711"
MODEL_DIR = f"{VOL_MOUNT}/models/sarvam-105b-1711"
BENCH_DIR = f"{VOL_MOUNT}/bench"
CACHE_DIR = f"{VOL_MOUNT}/cache"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

CAPTURE_SIZES = (
    "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,"
    "192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,"
    "480,496,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024,1088,1152,"
    "1216,1280,1344,1408,1472,1536,1600,1664,1728,1792,1856,1920,1984,2048,2176,2304,2432,"
    "2560,2688,2816,2944,3072,3200,3328,3456,3584,3712,3840,3968,4096,4352,4608,4864,5120,"
    "5376,5632,5888,6144,6400,6656,6912,7168,7424,7680,7936,8192]"
)

# Lightweight CPU image for caching 70GB weights once (~$0.05)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub>=0.26.0",
        "safetensors>=0.4.5",
        "numpy>=1.26.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Blackwell B200 vLLM 0.34.0 + Speculators 0.9.0 image (bundles torch 2.14.0+cu13, flashinfer, cutlass-dsl)
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
            "VLLM_CACHE_ROOT": f"{CACHE_DIR}/vllm",
            "FLASHINFER_WORKSPACE_BASE": f"{CACHE_DIR}/flashinfer",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor",
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
# 1. Cache Sarvam-105B-1711 (70GB) onto persistent Modal Volume (CPU-only!)
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    volumes={VOL_MOUNT: vol},
    cpu=8,
    memory=32768,
    timeout=3600,
)
def cache_model() -> Dict[str, Any]:
    """Download sarvamai/sarvam-105b-1711 + dspark/ once onto Modal Volume & inspect embeddings."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import numpy as np

    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[cache] Downloading {MODEL_REPO} -> {MODEL_DIR}", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        max_workers=8,
    )

    files = sorted(Path(MODEL_DIR).rglob("*"))
    total_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9

    # Check if dspark/embed_tokens.weight is bitwise identical to target embed_tokens.weight!
    embed_same = None
    try:
        idx_path = Path(MODEL_DIR) / "model.safetensors.index.json"
        with open(idx_path, "r") as f:
            wm = json.load(f)["weight_map"]
        emb_key = next(k for k in wm if "embed_tokens" in k)
        emb_shard = Path(MODEL_DIR) / wm[emb_key]
        with safe_open(str(emb_shard), framework="numpy") as f1, safe_open(
            f"{MODEL_DIR}/dspark/model.safetensors", framework="numpy"
        ) as f2:
            s1 = f1.get_slice(emb_key)[0:16, :]
            s2 = f2.get_slice("embed_tokens.weight")[0:16, :]
            embed_same = bool(np.array_equal(s1, s2))
            # Also check lm_head vs target lm_head!
            lm_key = next((k for k in wm if "lm_head" in k), None)
            lm_same = None
            if lm_key:
                with safe_open(str(Path(MODEL_DIR) / wm[lm_key]), framework="numpy") as f3:
                    l1 = f3.get_slice(lm_key)[0:16, :]
                    l2 = f2.get_slice("lm_head.weight")[0:16, :]
                    lm_same = bool(np.array_equal(l1, l2))
    except Exception as e:
        embed_same = f"ERR: {e}"
        lm_same = None

    vol.commit()
    res = {
        "model_dir": MODEL_DIR,
        "total_gb": round(total_gb, 2),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "dspark_embed_matches_target": embed_same,
        "dspark_lm_head_matches_target": lm_same,
    }
    print("[cache] Done:", json.dumps(res, indent=2), flush=True)
    return res


# ---------------------------------------------------------------------------
# 2. Inspect vLLM 0.34.0 DSpark + Blackwell MoE/MLA internals (CPU-only, ~$0.002!)
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    cpu=2,
    memory=8192,
    timeout=300,
)
def inspect_vllm_dspark() -> Dict[str, Any]:
    """Read exact vLLM 0.34.0 DSpark, SpeculativeConfig, and Blackwell kernel knobs."""
    import vllm
    import inspect

    vllm_root = Path(vllm.__file__).parent
    dspark_files = sorted(str(p.relative_to(vllm_root)) for p in vllm_root.rglob("*dspark*"))
    dflash_files = sorted(str(p.relative_to(vllm_root)) for p in vllm_root.rglob("*dflash*"))
    sarvam_files = sorted(str(p.relative_to(vllm_root)) for p in vllm_root.rglob("*sarvam*"))

    # Inspect SpeculativeConfig fields
    from vllm.config import SpeculativeConfig

    spec_fields = {}
    for k, v in SpeculativeConfig.__dataclass_fields__.items():
        if any(
            kw in k.lower()
            for kw in (
                "spec",
                "draft",
                "dspark",
                "adapt",
                "conf",
                "reject",
                "vocab",
                "markov",
                "method",
                "sample",
                "quant",
                "token",
                "model",
            )
        ):
            spec_fields[k] = str(v.default if v.default is not v.default_factory else "<factory>")

    # Dump key snippets from vLLM's dspark implementation files
    snippets: Dict[str, str] = {}
    for rel in dspark_files[:6]:
        p = vllm_root / rel
        if p.is_file() and p.suffix == ".py":
            txt = p.read_text(encoding="utf-8", errors="ignore")
            snippets[rel] = txt[:4000]

    # Check vllm.envs for FP4 / MoE / MLA / DSpark flags
    import vllm.envs as envs

    env_flags = [
        k
        for k in dir(envs)
        if any(
            w in k.upper()
            for w in ("FP4", "FP8", "MLA", "MOE", "DSPARK", "SPEC", "FLASH", "CUTE", "QUANT", "GRAPH")
        )
    ]

    info = {
        "vllm_version": vllm.__version__,
        "dspark_files": dspark_files,
        "dflash_files": dflash_files,
        "sarvam_files": sarvam_files,
        "spec_config_fields": spec_fields,
        "relevant_env_flags": env_flags,
        "dspark_snippets": snippets,
    }
    print(json.dumps({k: v for k, v in info.items() if k != "dspark_snippets"}, indent=2), flush=True)
    for k, v in snippets.items():
        print(f"\n================ vllm/{k} ================\n{v}", flush=True)
    return info


@app.function(
    image=vllm_image,
    volumes={VOL_MOUNT: vol},
    cpu=4,
    memory=16384,
    timeout=300,
)
def inspect_dspark_deep() -> Dict[str, Any]:
    """Deep CPU-only inspection of vLLM 0.34.0 DSpark proposer, config, Sarvam model, and weights."""
    import vllm
    import torch
    from safetensors import safe_open

    vol.reload()
    vllm_root = Path(vllm.__file__).parent
    out_dir = Path(BENCH_DIR) / "vllm_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check whether dspark embed_tokens & lm_head match target weights bitwise (using torch!)
    weight_checks: Dict[str, Any] = {}
    try:
        idx_path = Path(MODEL_DIR) / "model.safetensors.index.json"
        wm = json.loads(idx_path.read_text())["weight_map"]
        emb_key = next(k for k in wm if "embed_tokens" in k)
        lm_key = next((k for k in wm if "lm_head" in k), None)
        dspark_st = f"{MODEL_DIR}/dspark/model.safetensors"
        with safe_open(str(Path(MODEL_DIR) / wm[emb_key]), framework="pt") as f1, safe_open(
            dspark_st, framework="pt"
        ) as f2:
            t1 = f1.get_slice(emb_key)[0:64, :]
            t2 = f2.get_slice("embed_tokens.weight")[0:64, :]
            weight_checks["embed_shape_target"] = list(f1.get_slice(emb_key).get_shape())
            weight_checks["embed_shape_dspark"] = list(f2.get_slice("embed_tokens.weight").get_shape())
            weight_checks["embed_exact_match_first64"] = bool(torch.equal(t1, t2))
            weight_checks["embed_max_abs_diff_first64"] = float((t1.float() - t2.float()).abs().max().item())
            if lm_key:
                with safe_open(str(Path(MODEL_DIR) / wm[lm_key]), framework="pt") as f3:
                    l1 = f3.get_slice(lm_key)[0:64, :]
                    l2 = f2.get_slice("lm_head.weight")[0:64, :]
                    weight_checks["lm_head_shape_target"] = list(f3.get_slice(lm_key).get_shape())
                    weight_checks["lm_head_shape_dspark"] = list(f2.get_slice("lm_head.weight").get_shape())
                    weight_checks["lm_head_exact_match_first64"] = bool(torch.equal(l1, l2))
                    weight_checks["lm_head_max_abs_diff_first64"] = float(
                        (l1.float() - l2.float()).abs().max().item()
                    )
                    # Check if dspark lm_head is closer to target lm_head or distinct
                    cos = torch.nn.functional.cosine_similarity(
                        l1.float().flatten(), l2.float().flatten(), dim=0
                    ).item()
                    weight_checks["lm_head_cosine_first64"] = round(cos, 6)
    except Exception as e:
        weight_checks["error"] = repr(e)

    # 2. List all files inside v1/worker/gpu/spec_decode/dspark and dump key source files
    dspark_pkg = vllm_root / "v1/worker/gpu/spec_decode/dspark"
    dspark_pkg_files = sorted(str(p.relative_to(vllm_root)) for p in dspark_pkg.rglob("*.py"))

    target_files = [
        "model_executor/models/qwen3_dspark.py",
        "model_executor/models/sarvam.py",
        "config/speculative.py",
        *dspark_pkg_files,
    ]
    print("=== WEIGHT CHECKS ===", flush=True)
    print(json.dumps(weight_checks, indent=2), flush=True)
    print("=== DSPARK PKG FILES ===", flush=True)
    print(json.dumps(dspark_pkg_files, indent=2), flush=True)

    for rel in target_files:
        p = vllm_root / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"\n================ FILE: vllm/{rel} ({len(txt)} chars) ================", flush=True)
            print(txt, flush=True)

    # 3. Check CLI flags related to moe, attention, mla, speculative, async, fp4, quant
    help_out = subprocess.check_output(["vllm", "serve", "--help"], text=True, stderr=subprocess.STDOUT)
    interesting_lines = []
    lines = help_out.splitlines()
    for i, ln in enumerate(lines):
        if any(
            kw in ln.lower()
            for kw in (
                "attention-backend",
                "moe-backend",
                "all2all",
                "async-scheduling",
                "speculative",
                "kv-cache",
                "quantization",
                "compilation",
                "cudagraph",
                "scheduler",
                "mla",
                "fp4",
                "fp8",
                "batch",
                "max-num",
                "enable-chunked",
            )
        ):
            interesting_lines.append(ln)
    print("\n================ VLLM SERVE RELEVANT CLI FLAGS ================", flush=True)
    print("\n".join(interesting_lines[:200]), flush=True)

    return {"weight_checks": weight_checks, "dspark_pkg_files": dspark_pkg_files}


@app.function(
    image=vllm_image,
    cpu=2,
    memory=8192,
    timeout=300,
)
def inspect_knobs() -> Dict[str, Any]:
    """Inspect adaptive verification, batch-size spec schedule, MoE/MLA backends, and eagle utils."""
    import vllm
    from vllm.config.kernel import MoEBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    vllm_root = Path(vllm.__file__).parent
    print("=== MoEBackend ===", [m for m in dir(MoEBackend) if not m.startswith("_")], MoEBackend, flush=True)
    try:
        from typing import get_args
        print("MoEBackend args:", get_args(MoEBackend), flush=True)
    except Exception:
        pass
    print("=== AttentionBackendEnum ===", [e.name for e in AttentionBackendEnum], flush=True)

    # Search for enable_adaptive_verification and num_speculative_tokens_per_batch_size across vllm/v1
    v1_root = vllm_root / "v1"
    for term in (
        "enable_adaptive_verification",
        "draft_token_confidence_probs",
        "num_speculative_tokens_per_batch_size",
        "adaptive_verif",
        "share_target_lm_head",
    ):
        hits = []
        for p in sorted(v1_root.rglob("*.py")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if term in txt:
                rel = str(p.relative_to(vllm_root))
                lines = [
                    f"  L{i+1}: {ln}"
                    for i, ln in enumerate(txt.splitlines())
                    if term in ln
                ]
                hits.append((rel, lines))
        print(f"\n=== GREP: {term} ({len(hits)} files) ===", flush=True)
        for rel, lines in hits:
            print(f"FILE {rel}:", flush=True)
            for l in lines[:25]:
                print(l, flush=True)

    # Dump adaptive verification implementation & eagle/utils.py & draft quant utils
    for rel in (
        "v1/worker/gpu/spec_decode/eagle/utils.py",
        "v1/worker/gpu/spec_decode/dflash/speculator.py",
    ):
        p = vllm_root / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"\n================ FILE: vllm/{rel} ({len(txt)} chars) ================", flush=True)
            print(txt[:16000], flush=True)

    # Find any file with "adaptive" in its name
    adaptive_files = sorted(str(p.relative_to(vllm_root)) for p in v1_root.rglob("*adapt*"))
    print("\n=== ADAPTIVE FILES ===", adaptive_files, flush=True)
    for rel in adaptive_files:
        p = vllm_root / rel
        if p.is_file() and p.suffix == ".py":
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"\n================ FILE: vllm/{rel} ({len(txt)} chars) ================", flush=True)
            print(txt[:20000], flush=True)

    return {"adaptive_files": adaptive_files}


@app.function(
    image=vllm_image,
    volumes={VOL_MOUNT: vol},
    cpu=4,
    memory=24576,
    timeout=600,
)
def build_dspark_variants() -> Dict[str, Any]:
    """Build 100% lossless optimized DSpark checkpoints on CPU and verify vLLM config parsing."""
    from sarvam_infer.optimize_dspark import build_dspark_variant
    from vllm.config import ModelConfig

    vol.reload()
    src = Path(MODEL_DIR)
    base_out = Path(VOL_MOUNT) / "models" / "dspark_variants"

    results = {}
    # 1. Shared target embed_tokens + lm_head (full 262K vocab, -4.3GB VRAM)
    results["dspark_shared"] = build_dspark_variant(
        src,
        base_out / "dspark_shared",
        draft_vocab_size=None,
        share_target_embed=True,
        share_target_lm_head=True,
    )
    # 2. FR-Spec 64K shortlist (4x smaller lm_head & base_logits, -3.76GB VRAM, 100% lossless)
    results["dspark_frspec_64k"] = build_dspark_variant(
        src,
        base_out / "dspark_frspec_64k",
        draft_vocab_size=65536,
        share_target_embed=True,
        share_target_lm_head=False,
        prioritize_en_code_indic=True,
    )
    # 3. FR-Spec 32K shortlist (8x smaller lm_head & base_logits, -4.03GB VRAM, 100% lossless)
    results["dspark_frspec_32k"] = build_dspark_variant(
        src,
        base_out / "dspark_frspec_32k",
        draft_vocab_size=32768,
        share_target_embed=True,
        share_target_lm_head=False,
        prioritize_en_code_indic=True,
    )

    # Validate vLLM HF config parsing for each variant on CPU!
    from vllm.transformers_utils.config import get_config

    for name in results:
        d = str(base_out / name)
        hf_cfg = get_config(d, trust_remote_code=True)
        results[name]["parsed_hf_vocab_size"] = getattr(hf_cfg, "vocab_size", None)
        results[name]["parsed_hf_draft_vocab_size"] = getattr(hf_cfg, "draft_vocab_size", None)
        results[name]["parsed_hf_arch"] = getattr(hf_cfg, "architectures", None)
        results[name]["parsed_hf_conf_head"] = getattr(hf_cfg, "enable_confidence_head", None)

    vol.commit()
    print("=== BUILT ALL DSPARK VARIANTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return results


@app.function(
    image=vllm_image,
    cpu=2,
    memory=8192,
    timeout=300,
)
def inspect_wave1_findings() -> Dict[str, Any]:
    """Inspect CompilationMode, num_speculative_tokens_per_batch_size, online lm_head quant, and tactic=-1."""
    import vllm

    vllm_root = Path(vllm.__file__).parent
    files = [
        ("v1/spec_decode/dynamic/utils.py", 0, 8000),
        ("model_executor/layers/quantization/online/lm_head.py", 0, 12000),
        ("config/compilation.py", 0, 10000),
    ]
    for rel, s, e in files:
        p = vllm_root / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            print(f"\n================ FILE: vllm/{rel} ({len(txt)} chars) ================", flush=True)
            print(txt[s:e], flush=True)

    # Check scheduler.py around line 324 (num_speculative_tokens_per_batch_size)
    sched_p = vllm_root / "v1/core/sched/scheduler.py"
    if sched_p.exists():
        lines = sched_p.read_text(encoding="utf-8", errors="ignore").splitlines()
        print("\n=== scheduler.py L310..L370 ===", flush=True)
        for i in range(309, min(375, len(lines))):
            print(f"L{i+1}: {lines[i]}", flush=True)

    # Check where "outside the tuning bucket range" or tuning_buckets is defined
    for root_dir in (vllm_root, Path("/usr/local/lib/python3.12/site-packages/flashinfer")):
        if not root_dir.exists():
            continue
        for p in sorted(root_dir.rglob("*.py")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "outside the tuning bucket" in txt or "VLLM_USE_V2_MODEL_RUNNER" in txt:
                rel = str(p)
                hits = [f"  L{i+1}: {ln}" for i, ln in enumerate(txt.splitlines()) if any(k in ln for k in ("tuning_bucket", "outside the tuning", "CompilationMode", "enable_adaptive", "mode"))]
                if hits:
                    print(f"\nMATCH {rel}:", flush=True)
                    print("\n".join(hits[:25]), flush=True)
    return {"ok": True}


@app.function(
    image=vllm_image,
    cpu=2,
    memory=8192,
    timeout=300,
)
def inspect_w2() -> Dict[str, Any]:
    """Check why CompilationMode reset to NONE under MRV2, dynamic_sd_lookup usage, and KernelConfig CLI."""
    import vllm

    vllm_root = Path(vllm.__file__).parent
    for term in ("CompilationMode.NONE", "dynamic_sd_lookup", "monokernel", "TOKENSPEED_MLA", "FLASHINFER_CUTEDSL"):
        print(f"\n=== GREP: {term} ===", flush=True)
        n = 0
        for p in sorted(vllm_root.rglob("*.py")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if term in txt:
                rel = str(p.relative_to(vllm_root))
                for i, ln in enumerate(txt.splitlines()):
                    if term in ln:
                        print(f"  {rel}:{i+1}: {ln.strip()}", flush=True)
                        n += 1
                        if n >= 35:
                            break
            if n >= 35:
                break

    # Check KernelConfig fields & CLI flags in engine/arg_utils.py
    from vllm.config.kernel import KernelConfig
    print("\n=== KernelConfig fields ===", flush=True)
    for k, f in KernelConfig.__dataclass_fields__.items():
        print(f"  {k}: default={f.default}", flush=True)
    return {"ok": True}


@app.function(
    image=vllm_image,
    volumes={VOL_MOUNT: vol},
    cpu=4,
    memory=8192,
    timeout=600,
)
def prep_wave5_and_check_finetune() -> Dict[str, Any]:
    """Pack /vol/cache into a single fast-restore tarball + check async_scheduling, pass_manager, and hs_connectors."""
    import vllm
    import speculators
    import hs_connectors

    vol.reload()
    vroot = Path(vllm.__file__).parent
    sroot = Path(speculators.__file__).parent
    hroot = Path(hs_connectors.__file__).parent

    # 1. Pack /vol/cache -> /vol/cache_bundle.tar on CPU so B200 restores in ~5s instead of 325s!
    tar_path = f"{VOL_MOUNT}/cache_bundle.tar"
    t0 = time.perf_counter()
    if os.path.exists(CACHE_DIR):
        subprocess.run(["tar", "-cf", tar_path, "-C", CACHE_DIR, "."], check=True)
        size_mb = os.path.getsize(tar_path) / 1e6
        print(f"[tar] Packed {tar_path} ({size_mb:.1f} MB) in {time.perf_counter() - t0:.1f}s", flush=True)
        vol.commit()

    # 2. Inspect async_scheduling logic in vllm/config/vllm.py (L1585..L1650)
    vllm_cfg_lines = (vroot / "config/vllm.py").read_text(encoding="utf-8", errors="ignore").splitlines()
    print("\n=== vllm/config/vllm.py L1585..1650 (async_scheduling) ===", flush=True)
    for i in range(1584, min(1650, len(vllm_cfg_lines))):
        print(f"L{i+1}: {vllm_cfg_lines[i]}", flush=True)

    # 3. Inspect pass_manager.py (L140..255) for fuse_norm_quant / fuse_act_quant / fuse_attn_quant
    pm_lines = (vroot / "compilation/passes/pass_manager.py").read_text(encoding="utf-8", errors="ignore").splitlines()
    print("\n=== vllm/compilation/passes/pass_manager.py L140..255 ===", flush=True)
    for i in range(139, min(255, len(pm_lines))):
        print(f"L{i+1}: {pm_lines[i]}", flush=True)

    # 4. Inspect hs_connectors/fp8_hidden_states_connector.py (first 120 lines) & speculators train from_pretrained
    fp8_conn = (hroot / "fp8_hidden_states_connector.py").read_text(encoding="utf-8", errors="ignore").splitlines()
    print("\n=== hs_connectors/fp8_hidden_states_connector.py L1..100 ===", flush=True)
    for i, ln in enumerate(fp8_conn[:100]):
        print(f"L{i+1}: {ln}", flush=True)

    cli_txt = (sroot / "train/cli.py").read_text(encoding="utf-8", errors="ignore")
    idx = cli_txt.find("from_pretrained")
    if idx >= 0:
        print("\n=== speculators/train/cli.py around from_pretrained ===", flush=True)
        print(cli_txt[max(0, idx - 300):idx + 2500], flush=True)

    return {"tar_path": tar_path}


# ---------------------------------------------------------------------------
# 3. Standardized 1x B200 vLLM Benchmark Runner (A/B testable across configs!)
# ---------------------------------------------------------------------------


def _sync_dir(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["cp", "-ru", f"{src}/.", dst], check=False)


def _restore_cache_fast() -> None:
    """Restore /tmp/cache from single /vol/cache_bundle.tar (~3-5s) or fallback to _sync_dir."""
    tar_path = f"{VOL_MOUNT}/cache_bundle.tar"
    t0 = time.perf_counter()
    if os.path.exists(tar_path):
        print(f"[cache] Extracting {tar_path} -> /tmp/cache ...", flush=True)
        subprocess.run(["tar", "-xf", tar_path, "-C", "/tmp/cache"], check=False)
        print(f"[cache] Extracted in {time.perf_counter() - t0:.1f}s", flush=True)
    else:
        print("[cache] Restoring compiled kernel caches from /vol/cache -> /tmp/cache ...", flush=True)
        _sync_dir(CACHE_DIR, "/tmp/cache")
        print(f"[cache] Restored in {time.perf_counter() - t0:.1f}s", flush=True)


def _persist_cache_fast() -> None:
    """Persist /tmp/cache -> /vol/cache_bundle.tar in one sequential write (~5-8s)."""
    tar_path = f"{VOL_MOUNT}/cache_bundle.tar"
    tmp_tar = "/tmp/cache_bundle_new.tar"
    t0 = time.perf_counter()
    subprocess.run(["tar", "-cf", tmp_tar, "-C", "/tmp/cache", "."], check=False)
    if os.path.exists(tmp_tar):
        subprocess.run(["cp", "-f", tmp_tar, tar_path], check=False)
        os.remove(tmp_tar)
    vol.commit()
    print(f"[cache] Persisted cache_bundle.tar in {time.perf_counter() - t0:.1f}s", flush=True)


def _wait_for_server(
    base_url: str,
    proc: subprocess.Popen,
    log_file: Path,
    timeout_s: int = 1200,
) -> None:
    import urllib.request

    t0 = time.perf_counter()
    last_print = t0
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"[server] Ready in {time.perf_counter() - t0:.1f}s!", flush=True)
                    return
        except Exception:
            now = time.perf_counter()
            if now - last_print >= 25.0:
                last_print = now
                try:
                    lines = [ln for ln in log_file.read_text(errors="ignore").splitlines() if ln.strip()]
                    tail_ln = lines[-1][-160:] if lines else "(empty log)"
                except Exception:
                    tail_ln = "(reading log)"
                print(f"[server] waiting ({now - t0:.0f}s): {tail_ln}", flush=True)
            time.sleep(2.0)
    raise TimeoutError(f"vLLM server failed to become ready in {timeout_s}s")


def _run_one_server_bench(
    run_name: str,
    model_path: str = MODEL_DIR,
    dspark_path: Optional[str] = f"{MODEL_DIR}/dspark",
    spec_config_override: Optional[Dict[str, Any]] = None,
    compilation_config_override: Optional[Dict[str, Any]] = None,
    extra_vllm_args: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    concurrencies: Optional[List[int]] = None,
    max_tokens: int = 384,
    temperature: float = 1.0,
    gpu_memory_utilization: Optional[float] = 0.92,
    max_num_batched_tokens: int = 8192,
    save_as_greedy_ref: bool = False,
    run_reasoning_suite: bool = False,
    run_indic_suite: bool = False,
    wait_timeout_s: int = 1800,
) -> Dict[str, Any]:
    from sarvam_infer.bench_client import run_full_suite

    t_cfg = time.perf_counter()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    if compilation_config_override is not None:
        comp_cfg = json.dumps(compilation_config_override)
    else:
        comp_cfg = (
            '{"use_inductor_graph_partition":false,'
            '"pass_config":{"fuse_rope_kvcache_cat_mla":true},'
            f'"cudagraph_capture_sizes":{CAPTURE_SIZES},'
            '"max_cudagraph_capture_size":8192}'
        )

    cmd = [
        "vllm",
        "serve",
        model_path,
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
        "131072",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm45",
        "--reasoning-parser",
        "glm45",
        "--enable-prompt-tokens-details",
        "--compilation-config",
        comp_cfg,
    ]
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    if spec_config_override is not None:
        cmd.extend(["--speculative-config", json.dumps(spec_config_override)])
    elif dspark_path is not None:
        default_spec = {
            "method": "dspark",
            "model": dspark_path,
            "num_speculative_tokens": 5,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        }
        cmd.extend(["--speculative-config", json.dumps(default_spec)])

    if extra_vllm_args:
        cmd.extend(extra_vllm_args)

    print(f"\n=== [{run_name}] Launching vLLM ===\n{' '.join(cmd)}", flush=True)
    log_file = Path(BENCH_DIR) / f"{run_name}_server.log"
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            _wait_for_server("http://127.0.0.1:8000", proc, log_file=log_file, timeout_s=wait_timeout_s)
            tiers_list = concurrencies or [1, 8, 16, 32, 64, 128]
            ref_path = Path(BENCH_DIR) / "baseline_greedy_ref.json"
            suite = asyncio.run(
                run_full_suite(
                    base_url="http://127.0.0.1:8000",
                    model_name="sarvam-105b",
                    concurrencies=tiers_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    ref_path=ref_path,
                    save_as_ref=save_as_greedy_ref,
                    run_reasoning_suite=run_reasoning_suite,
                    run_indic_suite=run_indic_suite,
                )
            )
            elapsed_s = time.perf_counter() - t_cfg
            est_cost = (elapsed_s / 3600.0) * 6.25
            report = {
                "run_name": run_name,
                "elapsed_s": round(elapsed_s, 1),
                "est_gpu_cost_usd": round(est_cost, 3),
                "spec_config": spec_config_override or (dspark_path and "default_k5"),
                "extra_vllm_args": extra_vllm_args,
                "extra_env": extra_env,
                **suite,
            }
            out_path = Path(BENCH_DIR) / f"{run_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            lf.flush()
            log_text = log_file.read_text(encoding="utf-8", errors="ignore")
            highlights = [
                ln
                for ln in log_text.splitlines()
                if any(
                    kw in ln.lower()
                    for kw in (
                        "backend",
                        "moe",
                        "mla",
                        "dspark",
                        "speculative",
                        "kv cache",
                        "cudagraph",
                        "cuda graph",
                        "gpu memory",
                        "loaded weights",
                        "quantization",
                        "fp4",
                        "fp8",
                        "adaptive",
                        "lm_head",
                        "markov",
                        "async",
                        "cost tables",
                        "profile",
                    )
                )
                and "cubin_loader" not in ln.lower()
                and "capturing cuda graphs" not in ln.lower()
                and "[autotuner]: tuning" not in ln.lower()
            ]
            print(f"\n=== [{run_name}] SERVER LOG HIGHLIGHTS ({len(highlights)} lines) ===", flush=True)
            print("\n".join(highlights[:100]), flush=True)
            print(
                f"\n=== [{run_name}] DONE in {elapsed_s:.1f}s (est. ~${est_cost:.2f} @ $6.25/hr) ===",
                flush=True,
            )
            return report
        except Exception:
            lf.flush()
            tail = log_file.read_text(encoding="utf-8", errors="ignore")[-4000:]
            print(f"\n=== [{run_name}] SERVER LOG TAIL ===\n{tail}", flush=True)
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            # Small pause so GPU VRAM fully frees before next server launch in same container
            time.sleep(3.0)


@app.function(
    image=vllm_image,
    gpu="B200:1",
    cpu=16,
    memory=65536,
    volumes={VOL_MOUNT: vol},
    timeout=3600,
)
def run_bench_batch(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run multiple vLLM configs sequentially in ONE 1x B200 container sharing hot NVMe caches."""
    t_all = time.perf_counter()
    vol.reload()
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for sub in ("vllm", "flashinfer", "flashinfer_autotune", "inductor", "triton"):
        os.makedirs(f"/tmp/cache/{sub}", exist_ok=True)
    _restore_cache_fast()

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True
    )

    reports: Dict[str, Any] = {}
    adaptive_failed = False

    for cfg in configs:
        cfg = dict(cfg)
        name = cfg["run_name"]
        # If an earlier config showed adaptive verification is unsupported by the MLA backend,
        # automatically flip it off for subsequent configs that requested it so we still get data!
        spec = cfg.get("spec_config_override")
        if adaptive_failed and isinstance(spec, dict) and spec.get("enable_adaptive_verification"):
            spec = dict(spec)
            spec["enable_adaptive_verification"] = False
            cfg["spec_config_override"] = spec
            name = f"{name}_noadapt"
            cfg["run_name"] = name
            print(f"[batch] Auto-disabling adaptive_verification for {name} after earlier failure", flush=True)

        try:
            rep = _run_one_server_bench(**cfg)
            reports[name] = rep
        except Exception as e:
            err_s = repr(e)
            reports[name] = {"error": err_s}
            print(f"[batch] Config {name} failed: {err_s}", flush=True)
            if isinstance(spec, dict) and spec.get("enable_adaptive_verification"):
                adaptive_failed = True
                # Immediately retry this config with enable_adaptive_verification=False!
                fallback = dict(cfg)
                fb_spec = dict(spec)
                fb_spec["enable_adaptive_verification"] = False
                fallback["spec_config_override"] = fb_spec
                fallback["run_name"] = f"{name}_noadapt"
                print(f"[batch] Retrying {fallback['run_name']} without adaptive_verification...", flush=True)
                try:
                    reports[fallback["run_name"]] = _run_one_server_bench(**fallback)
                except Exception as e2:
                    reports[fallback["run_name"]] = {"error": repr(e2)}
                    print(f"[batch] Fallback {fallback['run_name']} also failed: {e2!r}", flush=True)
        finally:
            print(f"[cache] Persisting caches & results after {name} ...", flush=True)
            vol.commit()

    _persist_cache_fast()

    total_s = time.perf_counter() - t_all
    total_cost = (total_s / 3600.0) * 6.25
    summary = {
        "total_elapsed_s": round(total_s, 1),
        "total_est_gpu_cost_usd": round(total_cost, 3),
        "reports": reports,
    }
    print(f"\n================ BATCH SUMMARY ({total_s:.1f}s, ~${total_cost:.2f}) ================", flush=True)
    for rname, rep in reports.items():
        if "tiers" not in rep:
            print(f"  {rname}: ERROR {rep.get('error')}", flush=True)
            continue
        tps = " | ".join(
            f"B={t['concurrency']}:{t['total_throughput_tok_s']:.0f}({t['spec_mean_accepted_len']:.2f})"
            for t in rep["tiers"]
        )
        print(f"  {rname:34s} | {tps} | lossless={rep.get('lossless_check')}", flush=True)
    return summary


@app.function(
    image=vllm_image,
    gpu="B200:1",
    cpu=16,
    memory=65536,
    volumes={VOL_MOUNT: vol},
    timeout=3600,
)
def benchmark_vllm_config(
    run_name: str,
    model_path: str = MODEL_DIR,
    dspark_path: Optional[str] = f"{MODEL_DIR}/dspark",
    spec_config_override: Optional[Dict[str, Any]] = None,
    extra_vllm_args: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    concurrencies: Optional[List[int]] = None,
    max_tokens: int = 384,
    save_as_greedy_ref: bool = False,
) -> Dict[str, Any]:
    """Single-config wrapper calling _run_one_server_bench."""
    t_start = time.perf_counter()
    vol.reload()
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for sub in ("vllm", "flashinfer", "flashinfer_autotune", "inductor", "triton"):
        os.makedirs(f"/tmp/cache/{sub}", exist_ok=True)
    _sync_dir(CACHE_DIR, "/tmp/cache")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True
    )
    try:
        return _run_one_server_bench(
            run_name=run_name,
            model_path=model_path,
            dspark_path=dspark_path,
            spec_config_override=spec_config_override,
            extra_vllm_args=extra_vllm_args,
            extra_env=extra_env,
            concurrencies=concurrencies,
            max_tokens=max_tokens,
            save_as_greedy_ref=save_as_greedy_ref,
        )
    finally:
        _sync_dir("/tmp/cache", CACHE_DIR)
        vol.commit()


# ---------------------------------------------------------------------------
# Local CLI Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(action: str = "prep", run_name: str = "baseline") -> None:
    if action == "prep":
        print(">>> Running CPU-only model cache + vLLM 0.34.0 DSpark inspection in parallel...", flush=True)
        h_cache = cache_model.spawn()
        h_insp = inspect_vllm_dspark.spawn()
        r_insp = h_insp.get()
        r_cache = h_cache.get()
        print("\n=== PREP COMPLETE ===", flush=True)
        print("Cache:", json.dumps(r_cache, indent=2), flush=True)
        print("vLLM version:", r_insp["vllm_version"], flush=True)

    elif action == "inspect":
        res = inspect_dspark_deep.remote()
        print("\n=== DEEP INSPECT DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "inspect_knobs":
        res = inspect_knobs.remote()
        print("\n=== INSPECT KNOBS DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "build_dspark":
        res = build_dspark_variants.remote()
        print("\n=== BUILD DSPARK DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "inspect_w1":
        res = inspect_wave1_findings.remote()
        print("\n=== INSPECT W1 DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "inspect_w2":
        res = inspect_w2.remote()
        print("\n=== INSPECT W2 DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "inspect_speculators":
        res = prep_wave5_and_check_finetune.remote()
        print("\n=== PREP WAVE5 & CHECK FINETUNE DONE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "final_ab":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        capture_list = json.loads(CAPTURE_SIZES)
        inductor_env = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
        comp_o3_graphpart = {
            "mode": 3,
            "use_inductor_graph_partition": True,
            "pass_config": {
                "fuse_rope_kvcache_cat_mla": True,
                "fuse_attn_quant": True,
                "enable_qk_norm_rope_fusion": True,
            },
            "cudagraph_capture_sizes": capture_list,
            "max_cudagraph_capture_size": 8192,
        }
        configs = [
            {
                "run_name": "final_0_official_baseline",
                "dspark_path": f"{MODEL_DIR}/dspark",
                "gpu_memory_utilization": 0.92,
                "max_num_batched_tokens": 8192,
                "save_as_greedy_ref": True,
                "run_reasoning_suite": True,
                "run_indic_suite": True,
            },
            {
                "run_name": "final_1_opt_balanced_sector_tv",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_graphpart,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "extra_vllm_args": ["--attention-backend", "TOKENSPEED_MLA"],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_sector_tv",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 256, 5],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
                "run_reasoning_suite": True,
                "run_indic_suite": True,
            },
            {
                "run_name": "final_2_opt_indic_tv",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_graphpart,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "extra_vllm_args": ["--attention-backend", "TOKENSPEED_MLA"],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_indic_tv",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 256, 5],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
                "run_reasoning_suite": True,
                "run_indic_suite": True,
            },
            {
                "run_name": "final_3_opt_peak_tput",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_graphpart,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 8192,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_sector_tv",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
                "run_reasoning_suite": True,
                "run_indic_suite": True,
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "wave5":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        capture_list = json.loads(CAPTURE_SIZES)
        inductor_env = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
        comp_o3_fullfuse = {
            "mode": 3,
            "use_inductor_graph_partition": True,
            "pass_config": {
                "fuse_rope_kvcache_cat_mla": True,
                "fuse_attn_quant": True,
                "fuse_norm_quant": True,
                "fuse_act_quant": True,
                "enable_qk_norm_rope_fusion": True,
            },
            "cudagraph_capture_sizes": capture_list,
            "max_cudagraph_capture_size": 8192,
        }
        configs = [
            {
                "run_name": "w5_1_fullfuse_fi_fa_k5",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_fullfuse,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
            {
                "run_name": "w5_2_fullfuse_fi_fa_dynk75",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_fullfuse,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 256, 5],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
            {
                "run_name": "w5_3_fullfuse_tsmla_fa_dynk75",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_fullfuse,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "extra_vllm_args": ["--attention-backend", "TOKENSPEED_MLA"],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 256, 5],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
            {
                "run_name": "w5_4_fullfuse_fi_fa_dynk754",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_fullfuse,
                "gpu_memory_utilization": None,
                "max_num_batched_tokens": 16384,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 32, 5],
                        [33, 256, 4],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "wave4":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        capture_list = json.loads(CAPTURE_SIZES)
        inductor_env = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
        comp_o3_piecewise = {
            "mode": 3,
            "use_inductor_graph_partition": False,
            "pass_config": {
                "fuse_rope_kvcache_cat_mla": True,
                "fuse_attn_quant": False,
                "enable_qk_norm_rope_fusion": True,
            },
            "cudagraph_capture_sizes": capture_list,
            "max_cudagraph_capture_size": 8192,
        }
        comp_o3_graphpart = {
            "mode": 3,
            "use_inductor_graph_partition": True,
            "pass_config": {
                "fuse_rope_kvcache_cat_mla": True,
                "fuse_attn_quant": True,
                "enable_qk_norm_rope_fusion": True,
            },
            "cudagraph_capture_sizes": capture_list,
            "max_cudagraph_capture_size": 8192,
        }
        configs = [
            {
                "run_name": "w4_1_inductor_o3_piecewise_k5",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_piecewise,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
            },
            {
                "run_name": "w4_2_inductor_o3_graphpart_k5",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_graphpart,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
            },
            {
                "run_name": "w4_3_stack_inductor_tsmla_dynk_7_5_3",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_piecewise,
                "extra_vllm_args": [
                    "--attention-backend",
                    "TOKENSPEED_MLA",
                    "--max-num-batched-tokens",
                    "16384",
                ],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 32, 5],
                        [33, 256, 3],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
            {
                "run_name": "w4_4_stack_inductor_tsmla_dynk_7_5_3_2",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_piecewise,
                "extra_vllm_args": [
                    "--attention-backend",
                    "TOKENSPEED_MLA",
                    "--max-num-batched-tokens",
                    "16384",
                ],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 24, 5],
                        [25, 64, 3],
                        [65, 256, 2],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                },
            },
            {
                "run_name": "w4_5_stack_inductor_tsmla_dynk_topk512",
                "extra_env": inductor_env,
                "compilation_config_override": comp_o3_piecewise,
                "extra_vllm_args": [
                    "--attention-backend",
                    "TOKENSPEED_MLA",
                    "--max-num-batched-tokens",
                    "16384",
                ],
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 32, 5],
                        [33, 256, 3],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "attention_backend": "FLASH_ATTN",
                    "dspark_draft_topk": 512,
                },
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "baseline":
        res = benchmark_vllm_config.remote(
            run_name="baseline_k5",
            save_as_greedy_ref=True,
        )
        print(json.dumps(res, indent=2), flush=True)

    elif action == "wave1":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        configs = [
            {
                "run_name": "baseline_k5",
                "dspark_path": f"{MODEL_DIR}/dspark",
                "save_as_greedy_ref": True,
            },
            {
                "run_name": "opt1_shared_adapt_k8_topk128",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "enable_adaptive_verification": True,
                    "dspark_draft_topk": 128,
                },
            },
            {
                "run_name": "opt2_frspec64k_adapt_k8_topk128",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_frspec_64k",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "enable_adaptive_verification": True,
                    "dspark_draft_topk": 128,
                },
            },
            {
                "run_name": "opt3_frspec32k_adapt_k8_topk128",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_frspec_32k",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "enable_adaptive_verification": True,
                    "dspark_draft_topk": 128,
                },
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "wave2":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        patch_env = {
            "PYTHONPATH": "/app/sarvam_infer/patches:/app",
            "SARVAM_ENABLE_QWEN3_DSPARK_LM_HEAD_QUANT": "1",
        }
        capture_list = json.loads(CAPTURE_SIZES)
        configs = [
            {
                "run_name": "w2_0_baseline_k5",
                "dspark_path": f"{MODEL_DIR}/dspark",
                "save_as_greedy_ref": True,
            },
            {
                "run_name": "w2_1_shared_k8",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
            },
            {
                "run_name": "w2_2_shared_k8_topk512",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "dspark_draft_topk": 512,
                },
            },
            {
                "run_name": "w2_3_shared_k8_lmhead_nvfp4",
                "extra_env": patch_env,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "draft_lm_head_quantization": "nvfp4",
                },
            },
            {
                "run_name": "w2_4_shared_k8_lmhead_fp8",
                "extra_env": patch_env,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 8,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "draft_lm_head_quantization": "fp8",
                },
            },
            {
                "run_name": "w2_5_compile_o3_k5",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
                "compilation_config_override": {
                    "mode": 3,
                    "use_inductor_graph_partition": False,
                    "pass_config": {
                        "fuse_rope_kvcache_cat_mla": True,
                    },
                    "cudagraph_capture_sizes": capture_list,
                    "max_cudagraph_capture_size": 8192,
                },
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    elif action == "wave3":
        variants = f"{VOL_MOUNT}/models/dspark_variants"
        patch_env = {
            "PYTHONPATH": "/app/sarvam_infer/patches:/app",
            "SARVAM_ENABLE_QWEN3_DSPARK_LM_HEAD_QUANT": "1",
        }
        capture_list = json.loads(CAPTURE_SIZES)
        configs = [
            {
                "run_name": "w3_1_shared_k3",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 3,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
            },
            {
                "run_name": "w3_2_shared_dynk_7_5_4_3",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 7,
                    "num_speculative_tokens_per_batch_size": [
                        [1, 8, 7],
                        [9, 24, 5],
                        [25, 64, 4],
                        [65, 256, 3],
                    ],
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
            },
            {
                "run_name": "w3_3_moe_cutedsl_k5",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
                "extra_vllm_args": ["--moe-backend", "flashinfer_cutedsl"],
            },
            {
                "run_name": "w3_4_attn_tokenspeed_mla_k5",
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
                "extra_vllm_args": ["--attention-backend", "TOKENSPEED_MLA"],
            },
            {
                "run_name": "w3_5_k5_draft_fa_topk512_fp8head",
                "extra_env": patch_env,
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                    "dspark_draft_topk": 512,
                    "draft_lm_head_quantization": "fp8",
                    "attention_backend": "FLASH_ATTN",
                },
                "extra_vllm_args": ["--max-num-batched-tokens", "16384"],
            },
            {
                "run_name": "w3_6_inductor_o3_nobreakable_k5",
                "extra_env": {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
                "spec_config_override": {
                    "method": "dspark",
                    "model": f"{variants}/dspark_shared",
                    "num_speculative_tokens": 5,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "block",
                },
                "compilation_config_override": {
                    "mode": 3,
                    "use_inductor_graph_partition": False,
                    "pass_config": {
                        "fuse_rope_kvcache_cat_mla": True,
                        "fuse_attn_quant": True,
                        "enable_qk_norm_rope_fusion": True,
                    },
                    "cudagraph_capture_sizes": capture_list,
                    "max_cudagraph_capture_size": 8192,
                },
            },
        ]
        res = run_bench_batch.remote(configs)
        print(json.dumps(res, indent=2), flush=True)

    else:
        raise ValueError(f"Unknown action: {action!r}")
