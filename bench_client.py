from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Diverse reasoning, coding, tool-calling, math, and Indic prompts matching Sarvam-105B workloads
BENCH_PROMPTS: List[str] = [
    "Prove that there are infinitely many prime numbers congruent to 3 modulo 4. Show all steps clearly.",
    "Write a production-ready Python async rate limiter using a token bucket algorithm with per-key limits and thread/task safety.",
    "A farmer has 17 sheep; all but 9 die. Then he buys twice as many sheep as remain, and sells a third of his total flock. How many sheep does he have now? Explain step by step.",
    "Explain how Multi-Head Latent Attention (MLA) compresses the KV cache compared to GQA, including the matrix absorption trick during decoding.",
    "भारत में UPI (Unified Payments Interface) की सफलता के मुख्य तकनीकी और आर्थिक कारण क्या हैं? संक्षेप में बिंदुओं में समझाएं।",
    "Design a PostgreSQL schema and write the SQL query to compute a 7-day rolling active user retention cohort table with zero double-counting.",
    "Find all positive integers n such that n^2 + 3n + 2 is divisible by n + 5. Verify each solution.",
    "Implement a lock-free ring buffer in C++20 with acquire-release memory ordering and explain why relaxed ordering alone is insufficient.",
    "தமிழ் இலக்கியத்தில் சங்க காலப் பாடல்களின் முக்கியத்துவத்தையும் அகத்திணை மற்றும் புறத்திணை வேறுபாடுகளையும் விளக்கவும்.",
    "You are debugging a vLLM server where TTFT p90 spikes from 80ms to 1.5s at 64 concurrent users while decode tok/s stays high. Diagnose the 4 most likely causes and exact mitigation flags.",
    "Solve the recurrence a_n = 3 a_{n-1} - 2 a_{n-2} + n with a_0 = 1, a_1 = 4. Derive the closed form and verify for n = 2, 3, 4.",
    "Write a concise JSON schema and Python Pydantic v2 model for an Indian GST e-Invoice payload including CGST/SGST/IGST validation rules.",
    "Compare FlashAttention-4 CuTe DSL on Blackwell SM100 against Triton FlexAttention for variable-length packed document sequences.",
    "বাংলা সাহিত্যে রবীন্দ্রনাথ ঠাকুরের ছোটগল্পের মূল বৈশিষ্ট্যগুলি উদাহরণসহ আলোচনা করুন।",
    "Implement Dijkstra's shortest path algorithm in Rust using BinaryHeap with a custom Reverse state struct and handle disconnected nodes cleanly.",
    "A fair 6-sided die is rolled repeatedly until two consecutive sixes appear. What is the expected number of rolls? Derive carefully.",
]

GREEDY_VERIFY_PROMPTS: List[str] = BENCH_PROMPTS[:8]

# Fresh unseen Indic evaluation prompts across 8 Indian languages (zero overlap with Aya / BENCH_PROMPTS)
INDIC_EVAL_PROMPTS: List[str] = [
    "डिजिटल पब्लिक इन्फ्रास्ट्रक्चर (DPI) और आधार-यूपीआई स्टैक ने भारत में वित्तीय समावेशन को कैसे बदला है? मुख्य बिंदुओं में विश्लेषण करें।",
    "இந்தியாவில் புதுப்பிக்கத்தக்க எரிசக்தி (சூரிய மற்றும் காற்றாலை) உற்பத்தியை அதிகரிப்பதில் உள்ள முக்கிய சவால்கள் மற்றும் தீர்வுகளை விளக்கவும்.",
    "ভারতের অর্থনীতিতে ক্ষুদ্র ও মাঝারি শিল্পের (MSME) অবদান এবং ডিজিটাল ঋণ ব্যবস্থার প্রভাব আলোচনা করুন।",
    "महाराष्ट्रातील शेती क्षेत्रात कृत्रिम बुद्धिमत्ता (AI) आणि सूक्ष्म सिंचन तंत्रज्ञानाचा वापर कसा फायदेशीर ठरू शकतो ते स्पष्ट करा.",
    "భారతదేశంలో అంతరిక్ష పరిశోధనలో ఇస్రో (ISRO) సాధించిన ముఖ్యమైన విజయాలు మరియు గగన్‌యాన్ మిషన్ ప్రాముఖ్యతను వివరించండి.",
    "ગુજરાતમાં ગિફ્ટ સિટી (GIFT City) ફિનટેક અને આંતરરાષ્ટ્રીય નાણાકીય સેવાઓ માટે કેવી રીતે મહત્વપૂર્ણ કેન્દ્ર બની રહ્યું છે?",
    "ಬೆಂಗಳೂರಿನಲ್ಲಿ ತಂತ್ರಜ್ಞಾನ ನವೋದ್ಯಮಗಳ (Startups) ಬೆಳವಣಿಗೆಗೆ ಕಾರಣವಾದ ಪ್ರಮುಖ ಅಂಶಗಳು ಯಾವುವು? ವಿವರಿಸಿ.",
    "കേരളത്തിലെ വിനോദസഞ്ചാര മേഖലയിൽ സുസ്ഥിര ടൂറിസത്തിന്റെ (Responsible Tourism) പ്രാധാന്യവും സാമ്പത്തിക നേട്ടങ്ങളും വിശദീകരിക്കുക.",
    "भारत के नए डेटा संरक्षण अधिनियम (DPDP Act 2023) के तहत तकनीकी कंपनियों की प्रमुख जिम्मेदारियाँ क्या हैं?",
    "தமிழ்நாட்டின் மென்பொருள் மற்றும் மின்னணு உற்பத்தித் துறைகளின் வளர்ச்சிக்கு அரசு மேற்கொண்ட முக்கிய முயற்சிகள் யாவை?",
    "ভারতীয় রেলে বন্দে ভারত এক্সপ्रेस এবং আধুনিক সিগন্যালিং ব্যবস্থা (Kavach) কীভাবে যাত্রী নিরাপত্তা বাড়িয়েছে?",
    "भारत में सेमीकंडक्टर विनिर्माण मिशन (India Semiconductor Mission) के रणनीतिक महत्व और आपूर्ति श्रृंखला लाभों को समझाएं।",
]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


async def _fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> Dict[str, float]:
    """Scrape vLLM Prometheus /metrics for speculative decoding & KV cache counters."""
    out: Dict[str, float] = {}
    try:
        async with session.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=10)) as r:
            text = await r.text()
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if any(
                k in line
                for k in (
                    "spec_decode",
                    "num_draft",
                    "num_accepted",
                    "kv_cache_usage",
                    "prefix_cache",
                    "generation_tokens_total",
                    "prompt_tokens_total",
                    "time_per_output_token",
                    "e2e_request_latency",
                    "iteration_tokens",
                    "dspark",
                    "adaptive",
                )
            ):
                m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9eE.+\-]+)$", line.strip())
                if m:
                    name, val = m.group(1), float(m.group(2))
                    out[name] = out.get(name, 0.0) + val
    except Exception:
        pass
    return out


async def _single_stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if seed is not None:
        payload["seed"] = seed

    t0 = time.perf_counter()
    first_tok_t: Optional[float] = None
    chunks = 0
    completion_tokens = 0
    prompt_tokens = 0
    text_parts: List[str] = []
    reasoning_parts: List[str] = []

    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if evt.get("usage"):
                completion_tokens = int(evt["usage"].get("completion_tokens", completion_tokens))
                prompt_tokens = int(evt["usage"].get("prompt_tokens", prompt_tokens))
            choices = evt.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if c or rc:
                    if first_tok_t is None:
                        first_tok_t = time.perf_counter()
                    chunks += 1
                    if c:
                        text_parts.append(c)
                    if rc:
                        reasoning_parts.append(rc)

    t1 = time.perf_counter()
    if completion_tokens == 0:
        completion_tokens = chunks
    ttft_s = (first_tok_t - t0) if first_tok_t is not None else (t1 - t0)
    decode_s = max(t1 - (first_tok_t or t0), 1e-6)
    total_s = max(t1 - t0, 1e-6)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_s * 1000.0,
        "total_s": total_s,
        "decode_s": decode_s,
        "user_tps": completion_tokens / decode_s if completion_tokens > 1 else completion_tokens / total_s,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
    }


async def run_concurrency_tier(
    base_url: str,
    model_name: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = True,
    prompts: Optional[List[str]] = None,
    tag_prefix: str = "Request",
) -> Dict[str, Any]:
    prompt_pool = prompts or BENCH_PROMPTS
    connector = aiohttp.TCPConnector(limit=max(256, concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        m_before = await _fetch_metrics(session, base_url)
        sem = asyncio.Semaphore(concurrency)

        async def _worker(idx: int) -> Dict[str, Any]:
            # Add tiny unique prefix tag so prefix caching doesn't artificially inflate scores
            prompt = f"[{tag_prefix} #{idx}] {prompt_pool[idx % len(prompt_pool)]}"
            async with sem:
                return await _single_stream_request(
                    session,
                    base_url,
                    model_name,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_worker(i) for i in range(num_requests)])
        wall_s = max(time.perf_counter() - t0, 1e-6)
        m_after = await _fetch_metrics(session, base_url)

    total_out_toks = sum(r["completion_tokens"] for r in results)
    total_in_toks = sum(r["prompt_tokens"] for r in results)
    ttfts = [r["ttft_ms"] for r in results]
    user_tps = [r["user_tps"] for r in results]

    # Compute exact speculative decoding acceptance stats from Prometheus counter deltas
    spec_deltas: Dict[str, float] = {}
    for k, v in m_after.items():
        if "spec_decode" in k or "draft" in k or "accepted" in k:
            spec_deltas[k] = round(v - m_before.get(k, 0.0), 3)

    num_drafts = sum(v for k, v in spec_deltas.items() if k.endswith("num_drafts_total"))
    num_draft_toks = sum(v for k, v in spec_deltas.items() if k.endswith("num_draft_tokens_total"))
    num_accepted_toks = sum(
        v for k, v in spec_deltas.items() if k.endswith("num_accepted_tokens_total")
    )
    accept_rate = (num_accepted_toks / num_draft_toks) if num_draft_toks > 0 else 0.0
    mean_accept_len = (1.0 + num_accepted_toks / num_drafts) if num_drafts > 0 else 1.0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_s": round(wall_s, 3),
        "total_prompt_tokens": total_in_toks,
        "total_output_tokens": total_out_toks,
        "total_throughput_tok_s": round(total_out_toks / wall_s, 1),
        "user_speed_median_tok_s": round(statistics.median(user_tps), 1),
        "user_speed_mean_tok_s": round(statistics.mean(user_tps), 1),
        "ttft_p50_ms": round(_percentile(ttfts, 50), 1),
        "ttft_p90_ms": round(_percentile(ttfts, 90), 1),
        "spec_acceptance_rate": round(accept_rate, 4),
        "spec_mean_accepted_len": round(mean_accept_len, 3),
        "spec_raw_deltas": spec_deltas,
    }


async def run_greedy_lossless_check(
    base_url: str,
    model_name: str,
    ref_path: Optional[Path] = None,
    save_as_ref: bool = False,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Run deterministic temperature=0 prompts and compare against baseline for lossless proof."""
    connector = aiohttp.TCPConnector(limit=16)
    outputs = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, p in enumerate(GREEDY_VERIFY_PROMPTS):
            res = await _single_stream_request(
                session,
                base_url,
                model_name,
                p,
                max_tokens=max_tokens,
                temperature=0.0,
                enable_thinking=True,
                seed=42,
            )
            outputs.append(
                {
                    "idx": i,
                    "prompt": p,
                    "reasoning": res["reasoning"],
                    "text": res["text"],
                    "completion_tokens": res["completion_tokens"],
                }
            )

    if save_as_ref and ref_path is not None:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        return {"saved_reference": str(ref_path), "count": len(outputs)}

    if ref_path is not None and ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        exact_matches = 0
        prefix_match_ratios = []
        for cur, base in zip(outputs, ref):
            cur_full = (cur["reasoning"] or "") + "\n" + (cur["text"] or "")
            base_full = (base["reasoning"] or "") + "\n" + (base["text"] or "")
            if cur_full == base_full:
                exact_matches += 1
                prefix_match_ratios.append(1.0)
            else:
                n_same = 0
                for a, b in zip(cur_full, base_full):
                    if a == b:
                        n_same += 1
                    else:
                        break
                denom = max(len(base_full), 1)
                prefix_match_ratios.append(round(n_same / denom, 3))
        return {
            "exact_matches": f"{exact_matches}/{len(outputs)}",
            "avg_prefix_match_ratio": round(sum(prefix_match_ratios) / max(len(prefix_match_ratios), 1), 3),
            "prefix_match_ratios": prefix_match_ratios,
        }

    return {"count": len(outputs), "reference_missing": True}


async def run_logprob_lossless_check(
    base_url: str,
    model_name: str,
    ref_path: Optional[Path] = None,
    save_as_ref: bool = False,
) -> Dict[str, Any]:
    """Verify exact target-model distribution preservation via top-5 logprobs on identical prefixes."""
    connector = aiohttp.TCPConnector(limit=8)
    prompts = BENCH_PROMPTS[:12]
    records = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, p in enumerate(prompts):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            choice = (data.get("choices") or [{}])[0]
            lp_content = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
            top_lps = [
                {"token": x.get("token"), "logprob": round(float(x.get("logprob", 0.0)), 5)}
                for x in (lp_content.get("top_logprobs") or [])
            ]
            records.append(
                {
                    "idx": i,
                    "top1_token": lp_content.get("token"),
                    "top1_logprob": round(float(lp_content.get("logprob", 0.0)), 5)
                    if lp_content.get("logprob") is not None
                    else None,
                    "top5": top_lps,
                }
            )

    if save_as_ref and ref_path is not None:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return {"saved_logprob_ref": str(ref_path), "count": len(records)}

    if ref_path is not None and ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        top1_matches = 0
        diffs = []
        top5_diffs = []
        for cur, base in zip(records, ref):
            if cur["top1_token"] == base["top1_token"]:
                top1_matches += 1
            if cur["top1_logprob"] is not None and base["top1_logprob"] is not None:
                diffs.append(abs(cur["top1_logprob"] - base["top1_logprob"]))
            cur_map = {x["token"]: x["logprob"] for x in (cur.get("top5") or [])}
            for bx in base.get("top5") or []:
                if bx["token"] in cur_map:
                    top5_diffs.append(abs(cur_map[bx["token"]] - bx["logprob"]))
        return {
            "top1_token_agreement": f"{top1_matches}/{len(records)}",
            "max_abs_top1_logprob_diff": round(max(diffs), 5) if diffs else None,
            "mean_abs_top1_logprob_diff": round(sum(diffs) / len(diffs), 5) if diffs else None,
            "max_abs_top5_logprob_diff": round(max(top5_diffs), 5) if top5_diffs else None,
            "top5_tokens_compared": len(top5_diffs),
        }
    return {"count": len(records), "reference_missing": True}


async def run_full_suite(
    base_url: str,
    model_name: str,
    concurrencies: List[int],
    max_tokens: int = 384,
    temperature: float = 1.0,
    ref_path: Optional[Path] = None,
    save_as_ref: bool = False,
    run_reasoning_suite: bool = False,
    run_indic_suite: bool = False,
) -> Dict[str, Any]:
    # Multi-batch warmup (B=1, B=8, B=32) so CUDA graphs & FlashInfer tactics are hot before timing
    connector = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            _single_stream_request(session, base_url, model_name, "Say hello.", 16, 0.6, False),
            _single_stream_request(session, base_url, model_name, "What is 17 * 23?", 24, 1.0, True),
        )
        await asyncio.gather(
            *[
                _single_stream_request(
                    session, base_url, model_name, f"[Warmup {i}] Explain MLA briefly.", 24, 1.0, True
                )
                for i in range(32)
            ]
        )
        await _single_stream_request(session, base_url, model_name, "What is 2 + 2?", 16, 0.0, False)
        await asyncio.sleep(0.3)

    tiers = []
    for c in concurrencies:
        await asyncio.sleep(0.3)
        # Scale request count so every tier runs long enough for stable throughput & spec stats
        n_req = max(c, 4) if c <= 8 else (c * 2 if c <= 32 else c)
        tier_res = await run_concurrency_tier(
            base_url=base_url,
            model_name=model_name,
            concurrency=c,
            num_requests=n_req,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=True,
        )
        print(
            f"  [B={c:3d}] total={tier_res['total_throughput_tok_s']:7.1f} tok/s | "
            f"per_user={tier_res['user_speed_median_tok_s']:6.1f} tok/s | "
            f"TTFT p50/p90={tier_res['ttft_p50_ms']:5.1f}/{tier_res['ttft_p90_ms']:6.1f} ms | "
            f"spec_acc={tier_res['spec_acceptance_rate']:.3f} (len={tier_res['spec_mean_accepted_len']:.2f})",
            flush=True,
        )
        tiers.append(tier_res)

    reasoning_tiers = []
    if run_reasoning_suite:
        print("  --- Sustained Reasoning Suite (max_tokens=1024, temp=0.6) ---", flush=True)
        for c in [1, 16, 64, 128]:
            await asyncio.sleep(0.3)
            n_req = max(c, 4) if c <= 1 else c
            r_res = await run_concurrency_tier(
                base_url=base_url,
                model_name=model_name,
                concurrency=c,
                num_requests=n_req,
                max_tokens=1024,
                temperature=0.6,
                enable_thinking=True,
                tag_prefix="Reason",
            )
            print(
                f"  [R B={c:3d}] total={r_res['total_throughput_tok_s']:7.1f} tok/s | "
                f"per_user={r_res['user_speed_median_tok_s']:6.1f} tok/s | "
                f"TTFT p50/p90={r_res['ttft_p50_ms']:5.1f}/{r_res['ttft_p90_ms']:6.1f} ms | "
                f"spec_acc={r_res['spec_acceptance_rate']:.3f} (len={r_res['spec_mean_accepted_len']:.2f})",
                flush=True,
            )
            reasoning_tiers.append(r_res)

    indic_tiers = []
    if run_indic_suite:
        print("  --- Unseen Indic Sector Suite (8 languages, max_tokens=384, temp=1.0) ---", flush=True)
        for c in [1, 16]:
            await asyncio.sleep(0.3)
            n_req = 6 if c == 1 else 16
            i_res = await run_concurrency_tier(
                base_url=base_url,
                model_name=model_name,
                concurrency=c,
                num_requests=n_req,
                max_tokens=384,
                temperature=1.0,
                enable_thinking=True,
                prompts=INDIC_EVAL_PROMPTS,
                tag_prefix="Indic",
            )
            print(
                f"  [I B={c:3d}] total={i_res['total_throughput_tok_s']:7.1f} tok/s | "
                f"per_user={i_res['user_speed_median_tok_s']:6.1f} tok/s | "
                f"TTFT p50/p90={i_res['ttft_p50_ms']:5.1f}/{i_res['ttft_p90_ms']:6.1f} ms | "
                f"spec_acc={i_res['spec_acceptance_rate']:.3f} (len={i_res['spec_mean_accepted_len']:.2f})",
                flush=True,
            )
            indic_tiers.append(i_res)

    lossless_greedy = await run_greedy_lossless_check(
        base_url=base_url,
        model_name=model_name,
        ref_path=ref_path,
        save_as_ref=save_as_ref,
        max_tokens=128,
    )
    lp_ref_path = ref_path.parent / "baseline_logprob_ref.json" if ref_path is not None else None
    lossless_logprob = await run_logprob_lossless_check(
        base_url=base_url,
        model_name=model_name,
        ref_path=lp_ref_path,
        save_as_ref=save_as_ref,
    )
    out: Dict[str, Any] = {
        "tiers": tiers,
        "lossless_check": {
            "logprob_equivalence": lossless_logprob,
            "greedy_match": lossless_greedy,
        },
    }
    if reasoning_tiers:
        out["reasoning_tiers"] = reasoning_tiers
    if indic_tiers:
        out["indic_tiers"] = indic_tiers
    return out
