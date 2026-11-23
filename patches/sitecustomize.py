"""Lazy import hook enabling native vLLM draft_lm_head_quantization for Qwen3DSparkModel.

Active ONLY when SARVAM_ENABLE_QWEN3_DSPARK_LM_HEAD_QUANT=1 is set in the environment.
vLLM 0.34.0 already implements online NVFP4/FP8 lm_head quantization in
`share_target_lm_head` -> `quantized_lm_head_copy`, and Qwen3DSparkForCausalLM
only accesses `self.lm_head` through `self.logits_processor(self.lm_head, hidden_states)`.
This hook solely adds `"Qwen3DSparkModel"` to DSparkSpeculatorConfig's architecture check.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_TARGET_MOD = "vllm.v1.worker.gpu.spec_decode.dspark.config"


def _patch_dspark_config(module) -> None:
    from vllm.config.speculative import (
        _QWEN3_OMNI_DSPARK_ARCHITECTURE,
        _validate_qwen3_omni_dspark,
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.config import (
        check_draft_lm_head_quantization,
    )

    cls = module.DSparkSpeculatorConfig

    @classmethod
    def _patched_apply_options(cls_, spec) -> None:
        draft = spec.draft_model_config
        hf_config = draft.hf_config
        dspark_draft_topk = spec.dspark_draft_topk
        if dspark_draft_topk is None:
            dspark_draft_topk = getattr(hf_config, "dspark_draft_topk", None)
        if dspark_draft_topk is not None:
            draft_vocab_size = (
                getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
            )
            if not 1 <= dspark_draft_topk <= draft_vocab_size:
                raise ValueError(
                    "dspark_draft_topk must be between 1 and the "
                    f"draft vocabulary size ({draft_vocab_size})"
                )
            if (
                "Qwen3DSparkModel" not in draft.architectures
                and _QWEN3_OMNI_DSPARK_ARCHITECTURE not in draft.architectures
            ):
                raise ValueError(
                    "dspark_draft_topk is only supported by "
                    "Qwen3DSparkModel and Qwen3OmniDSparkModel"
                )
            hf_config.dspark_draft_topk = dspark_draft_topk

        if spec.draft_lm_head_quantization is not None:
            if not {
                "DSparkDraftModel",
                "DSparkV41DraftModel",
                "Qwen3DSparkModel",
            } & set(draft.architectures):
                raise ValueError(
                    "draft_lm_head_quantization is only supported by "
                    "DSparkDraftModel, DSparkV41DraftModel, and Qwen3DSparkModel"
                )
            check_draft_lm_head_quantization(spec)

        assert spec.target_model_config is not None
        _validate_qwen3_omni_dspark(
            spec.target_model_config, draft, spec.num_speculative_tokens
        )

    cls.apply_options = _patched_apply_options


class _DSparkPatchLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self._orig_loader = orig_loader

    def create_module(self, spec):
        if hasattr(self._orig_loader, "create_module"):
            return self._orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        _patch_dspark_config(module)


class _DSparkPatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._active = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MOD or self._active:
            return None
        self._active = True
        try:
            spec = importlib.util.find_spec(fullname)
            if spec is not None and spec.loader is not None:
                spec.loader = _DSparkPatchLoader(spec.loader)
            return spec
        finally:
            self._active = False


if os.environ.get("SARVAM_ENABLE_QWEN3_DSPARK_LM_HEAD_QUANT") == "1":
    sys.meta_path.insert(0, _DSparkPatchFinder())


# ---------------------------------------------------------------------------
# Optional MLA Hidden-State Extractor Hook (active ONLY when SARVAM_DUMP_HS_DIR is set)
# ---------------------------------------------------------------------------
# vLLM 0.34.0's ExampleHiddenStatesConnector does not yet support MLA verifiers
# because MLA shares a single KV-cache group. When SARVAM_DUMP_HS_DIR is set during
# offline DSpark distillation data generation (enforce-eager, max_num_seqs=1),
# this hook taps SarvamMLAModel.forward directly, extracts layers (2,9,16,23,30,32),
# FP8-quantizes them via hs_connectors.fp8_utils, and writes atomic safetensors
# while returning the normal single hidden_states tensor to vLLM.

_SARVAM_MOD = "vllm.model_executor.models.sarvam"


def _patch_sarvam_hs_dump(module) -> None:
    import concurrent.futures
    import torch
    from safetensors.torch import save_file
    from hs_connectors.fp8_utils import SCALES_KEY, quantize_tensor_to_fp8

    dump_dir = os.environ["SARVAM_DUMP_HS_DIR"]
    os.makedirs(dump_dir, exist_ok=True)
    trigger_file = os.path.join(dump_dir, "ACTIVE")
    layer_ids = tuple(
        int(x)
        for x in os.environ.get("SARVAM_DUMP_HS_LAYERS", "2,9,16,23,30,32").split(",")
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="sarvam-hs")
    state = {"counter": 0, "verified_norm": False}

    def _write_atomic(out_path: str, ids_cpu: torch.Tensor, fp8_cpu: torch.Tensor, sc_cpu: torch.Tensor) -> None:
        tmp_path = out_path + ".tmp"
        save_file(
            {
                "input_ids": ids_cpu,
                "hidden_states": fp8_cpu,
                SCALES_KEY: sc_cpu,
            },
            tmp_path,
        )
        os.replace(tmp_path, out_path)

    cls = module.SarvamMLAModel
    orig_forward = cls.forward

    def _wrapped_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        active = (
            os.path.exists(trigger_file)
            and input_ids is not None
            and input_ids.numel() >= 16
        )
        if not active:
            return orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

        prev_layers = tuple(getattr(self, "aux_hidden_state_layers", ()))
        if hasattr(self, "set_aux_hidden_state_layers"):
            self.set_aux_hidden_state_layers(layer_ids)
        else:
            self.aux_hidden_state_layers = layer_ids
        try:
            out = orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        finally:
            if hasattr(self, "set_aux_hidden_state_layers"):
                self.set_aux_hidden_state_layers(prev_layers)
            else:
                self.aux_hidden_state_layers = prev_layers

        if isinstance(out, tuple) and len(out) == 2:
            hidden_states, aux_list = out
            if len(aux_list) == len(layer_ids):
                if not state["verified_norm"]:
                    state["verified_norm"] = True
                    with torch.no_grad():
                        normed_last = self.norm(aux_list[-1].clone())
                        diff = (normed_last.float() - hidden_states.float()).abs().max().item()
                        print(
                            f"[hs_dump] SELF-TEST norm(aux[-1]) vs hidden_states "
                            f"max_abs_diff={diff:.6f} shape={list(hidden_states.shape)}",
                            flush=True,
                        )
                stacked = torch.stack(aux_list, dim=1)  # [T, 6, 4096]
                fp8_hs, scales = quantize_tensor_to_fp8(stacked)
                ids_cpu = input_ids.detach().to("cpu", non_blocking=False).contiguous()
                fp8_cpu = fp8_hs.detach().to("cpu", non_blocking=False).contiguous()
                sc_cpu = scales.detach().to("cpu", non_blocking=False).contiguous()
                idx = state["counter"]
                state["counter"] += 1
                out_path = os.path.join(dump_dir, f"hs_{idx:06d}.safetensors")
                executor.submit(_write_atomic, out_path, ids_cpu, fp8_cpu, sc_cpu)
            return hidden_states
        return out

    cls.forward = _wrapped_forward


class _SarvamHSDumpLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self._orig_loader = orig_loader

    def create_module(self, spec):
        if hasattr(self._orig_loader, "create_module"):
            return self._orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        _patch_sarvam_hs_dump(module)


class _SarvamHSDumpFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._active = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _SARVAM_MOD or self._active:
            return None
        self._active = True
        try:
            spec = importlib.util.find_spec(fullname)
            if spec is not None and spec.loader is not None:
                spec.loader = _SarvamHSDumpLoader(spec.loader)
            return spec
        finally:
            self._active = False


if os.environ.get("SARVAM_DUMP_HS_DIR"):
    sys.meta_path.insert(0, _SarvamHSDumpFinder())
