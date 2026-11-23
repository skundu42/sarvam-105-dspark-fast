#!/usr/bin/env bash
# Production-ready 100% lossless Track 1 vLLM 0.34.0 launch script for
# sarvamai/sarvam-105b-1711 on 1x NVIDIA B200 (180GB).
#
# Usage:
#   MODE=balanced ./sarvam_infer/serve_sarvam_track1.sh /path/to/sarvam-105b-1711 /path/to/dspark_sector_tv
#   MODE=peak_tput ./sarvam_infer/serve_sarvam_track1.sh /path/to/sarvam-105b-1711 /path/to/dspark_sector_tv
#
# Build the 1.21 GB weight-shared DSpark checkpoint (if using stock Sarvam weights):
#   python -c "from pathlib import Path; from sarvam_infer.optimize_dspark import build_dspark_variant; \
#     build_dspark_variant(Path('/path/to/sarvam-105b-1711'), Path('/path/to/dspark_shared'), \
#     draft_vocab_size=None, share_target_embed=True, share_target_lm_head=True)"

set -euo pipefail

MODEL_DIR="${1:-sarvamai/sarvam-105b-1711}"
DSPARK_DIR="${2:-skundu42/sarvam-105b-1711-dspark-fast}"
PORT="${PORT:-8000}"
MODE="${MODE:-peak_tput}" # "peak_tput" (FLASHINFER_MLA + K=5) | "balanced" (zero-stall TOKENSPEED_MLA + dyn-K)

CAPTURE_SIZES="[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024,1088,1152,1216,1280,1344,1408,1472,1536,1600,1664,1728,1792,1856,1920,1984,2048,2176,2304,2432,2560,2688,2816,2944,3072,3200,3328,3456,3584,3712,3840,3968,4096,4352,4608,4864,5120,5376,5632,5888,6144,6400,6656,6912,7168,7424,7680,7936,8192]"

COMP_CFG="{\"mode\":3,\"use_inductor_graph_partition\":true,\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":true,\"fuse_attn_quant\":true,\"enable_qk_norm_rope_fusion\":true},\"cudagraph_capture_sizes\":${CAPTURE_SIZES},\"max_cudagraph_capture_size\":8192}"

# Unlock torch.compile Inductor mode=3 + graph-partitioned FULL_AND_PIECEWISE CUDA graphs
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ENFORCE_STRICT_TOOL_CALLING=0
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

if [[ "${MODE}" == "balanced" ]]; then
  SPEC_CFG="{\"method\":\"dspark\",\"model\":\"${DSPARK_DIR}\",\"num_speculative_tokens\":7,\"num_speculative_tokens_per_batch_size\":[[1,8,7],[9,256,5]],\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"attention_backend\":\"FLASH_ATTN\"}"
  exec vllm serve "${MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --served-model-name sarvam-105b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend TOKENSPEED_MLA \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-prompt-tokens-details \
    --compilation-config "${COMP_CFG}" \
    --speculative-config "${SPEC_CFG}"
elif [[ "${MODE}" == "peak_tput" ]]; then
  SPEC_CFG="{\"method\":\"dspark\",\"model\":\"${DSPARK_DIR}\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
  exec vllm serve "${MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --served-model-name sarvam-105b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-prompt-tokens-details \
    --compilation-config "${COMP_CFG}" \
    --speculative-config "${SPEC_CFG}"
else
  echo "Unknown MODE=${MODE} (expected 'balanced' or 'peak_tput')" >&2
  exit 1
fi
