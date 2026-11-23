from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import modal

APP_NAME = "sarvam-105b-hf-publish"
VOLUME_NAME = "sarvam-105b-infer-vol"
VOL_MOUNT = "/vol"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

pub_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub>=0.26.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path="/app/sarvam_infer",
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store"],
    )
)


@app.function(
    image=pub_image,
    volumes={VOL_MOUNT: vol},
    cpu=4,
    memory=8192,
    timeout=600,
)
def publish_dspark_to_hf(
    repo_id: str,
    hf_token: str,
    variant_dir: str = "/vol/models/dspark_variants/dspark_sector_tv",
) -> Dict[str, Any]:
    """Assemble and upload the 1.21 GB weight-shared MXFP8 DSpark checkpoint, charts, README, and scripts to HF."""
    from huggingface_hub import HfApi

    t0 = time.perf_counter()
    vol.reload()
    src = Path(variant_dir)
    assert src.exists(), f"Variant directory not found: {src}"

    # Verify config.json has no local /vol paths
    cfg_text = (src / "config.json").read_text(encoding="utf-8")
    assert "/vol/" not in cfg_text, "Found local /vol/ path in config.json!"

    # Verify README has zero em/en dashes
    readme_src = Path("/app/sarvam_infer/HF_README.md")
    readme_text = readme_src.read_text(encoding="utf-8")
    assert "\u2014" not in readme_text and "\u2013" not in readme_text, "Found em/en dash in README!"

    stage = Path(tempfile.mkdtemp(prefix="hf_stage_"))
    try:
        # 1. Model checkpoint files at repo root
        for item in sorted(src.iterdir()):
            if item.is_file():
                shutil.copy2(item, stage / item.name)
                print(f"[stage] root/{item.name} ({item.stat().st_size / 1e6:.2f} MB)", flush=True)

        # 2. Model card README.md
        shutil.copy2(readme_src, stage / "README.md")

        # 3. Benchmark charts under assets/
        assets_dst = stage / "assets"
        assets_dst.mkdir(parents=True, exist_ok=True)
        for png in ("throughput_gains.png", "latency_and_kv.png"):
            shutil.copy2(Path("/app/sarvam_infer/assets") / png, assets_dst / png)

        # 4. Clean reproducible scripts under scripts/ (never including publish_hf.py or tokens)
        scripts_dst = stage / "scripts"
        scripts_dst.mkdir(parents=True, exist_ok=True)
        for fname in (
            "serve_sarvam_track1.sh",
            "optimize_dspark.py",
            "bench_client.py",
            "dspark_finetune.py",
        ):
            shutil.copy2(Path("/app/sarvam_infer") / fname, scripts_dst / fname)
        patches_dst = scripts_dst / "patches"
        patches_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("/app/sarvam_infer/patches/sitecustomize.py"), patches_dst / "sitecustomize.py")

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        commit_info = api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload 1.21 GB lossless weight-shared MXFP8 DSpark checkpoint, charts, and vLLM serving recipe",
        )
        uploaded_files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        res = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "commit_url": str(getattr(commit_info, "commit_url", "")),
            "files": uploaded_files,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        print("[hf] Upload complete:", json.dumps(res, indent=2), flush=True)
        return res
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@app.local_entrypoint()
def main(repo_id: str = "skundu42/sarvam-105b-1711-dspark-fast") -> None:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("Set HF_TOKEN in environment before running publish_hf.py")
    res = publish_dspark_to_hf.remote(repo_id=repo_id, hf_token=tok)
    print(json.dumps(res, indent=2), flush=True)
