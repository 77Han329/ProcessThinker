"""
Process-aware reward with stepwise external rollouts.
"""

from __future__ import annotations

import json
import math
import os
import re
import pdb
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union, Tuple

# ==================== Reward logging config ====================
REWARD_LOG_ENABLED = True  # Set to False to disable logging
REWARD_LOG_DIR = "/hnvme/workspace/v100dd13-reasoning_model/Processthinker/EasyR1/reward_logs"  # default path

# Global counter for tracking the training step
_global_step_counter = 0

def _get_log_file(prefix: str = "rewards", log_dir: str = None) -> str:
    """Return the log file path."""
    target_dir = log_dir if log_dir else REWARD_LOG_DIR
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d')}.jsonl")

def _log_reward(sample_idx: int, item: dict, result: dict, normalized_response: str = None, log_dir: str = None):
    """Save per-sample reward info to a jsonl file (thread-safe)."""
    if not REWARD_LOG_ENABLED:
        return

    global _global_step_counter

    try:
        raw_response = item.get("response", "")
        response = normalized_response if normalized_response else raw_response
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": _global_step_counter,
            "sample_idx": sample_idx,
            "ground_truth": item.get("ground_truth", ""),
            "problem": (item.get("problem", "") or "")[:500],  # truncated problem text
            "response": response,  # normalized response (used for reward computation)
            "raw_response": raw_response,  # original response (for debugging)
            "response_length_tokens": item.get("response_length", 0),  # token count (from framework)
            "response_length_chars": len(response),  # character count
            "rewards": result,
        }

        # Protect file writes with a lock
        with _log_file_lock:
            with open(_get_log_file("rewards", log_dir), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[_log_reward] save failed: {e}")

def _log_process_rollout(sample_idx: int, step_idx: int, prefix_steps: List[str],
                         outputs: List[str], accs: List[float], prefix_score: float, log_dir: str = None):
    """Log each continuation produced during process-reward computation (thread-safe)."""
    if not REWARD_LOG_ENABLED:
        return

    global _global_step_counter

    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": _global_step_counter,
            "sample_idx": sample_idx,
            "step_idx": step_idx,
            "prefix_steps": prefix_steps,  # accumulated prefix steps so far
            "continuations": [  # each continuation and its accuracy
                {"output": out, "accuracy": acc}
                for out, acc in zip(outputs, accs)
            ],
            "prefix_score": prefix_score,  # average score for this prefix
        }

        with _log_file_lock:
            with open(_get_log_file("process_rollouts", log_dir), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[_log_process_rollout] save failed: {e}")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from mathruler.grader import grade_answer as _grade_answer
except Exception:
    def _grade_answer(pred: str, gt: str) -> bool:
        return (pred or "").strip() == (gt or "").strip()


# ==================== Regex patterns ====================
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
ANSWER_CAPTURE_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# Global client cache: {(base_url, timeout): OpenAI client}
_client_cache: Dict[tuple, Any] = {}
_client_cache_lock = Lock()  # protects the client cache

# Global endpoint list for load balancing
_endpoint_list: List[str] = []
_endpoint_call_count: Dict[str, int] = {}  # number of calls per endpoint
_endpoint_count_lock = Lock()  # protects the call counter

# Lock for log file writes
_log_file_lock = Lock()


def _parse_endpoints(endpoint_str: str) -> List[str]:
    """Parse an endpoint string; supports comma-separated endpoints."""
    if not endpoint_str:
        return []
    endpoints = [e.strip() for e in endpoint_str.split(",") if e.strip()]
    return endpoints


def _select_endpoint(endpoints: List[str]) -> str:
    """Randomly pick one of the endpoints (load balancing)."""
    if not endpoints:
        raise ValueError("No endpoints available")
    if len(endpoints) == 1:
        return endpoints[0]

    # Random selection
    selected = random.choice(endpoints)

    # Track calls per endpoint (thread-safe)
    with _endpoint_count_lock:
        if selected not in _endpoint_call_count:
            _endpoint_call_count[selected] = 0
        _endpoint_call_count[selected] += 1

    return selected


def _get_openai_client(base_url: str, timeout: float) -> Any:
    """Return a cached OpenAI client (singleton, thread-safe)."""
    cache_key = (base_url, timeout)

    # Fast path: check without locking
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    # Lock on create to avoid duplicate instantiation
    with _client_cache_lock:
        # Double-check
        if cache_key not in _client_cache:
            if OpenAI is None:
                raise ImportError(
                    "OpenAI client is required. Please install it with: pip install openai"
                )
            print(f"[pg_reward] creating OpenAI client: {base_url}")
            _client_cache[cache_key] = OpenAI(
                api_key="EMPTY",  # vLLM does not require a real API key
                base_url=base_url,
                timeout=timeout,
            )
    return _client_cache[cache_key]


def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _normalize_tags(text: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", text or "")


def _extract_steps(response: str) -> List[str]:
    """
    Extract all <step> contents inside <think> tags.

    Important: only steps *inside* <think>...</think> are extracted.
    Anything after </think> (e.g. <answer>) is ignored.
    """
    # Step 1: normalize tags (strip whitespace around angle brackets)
    response = _normalize_tags(response)

    # Step 2: extract the content inside <think>...</think>
    match = THINK_RE.search(response)
    if not match:
        return []

    think = match.group(1) or ""

    # Step 3: find all well-formed <step>...</step> pairs inside the think content
    steps = [s.strip() for s in STEP_RE.findall(think) if s.strip()]

    return steps


# ==================== Format Reward ====================

def check_format_valid(response: str, step_min: int = 2, step_max: int = 6, verbose: bool = True) -> bool:
    """
    Check whether a response satisfies the format requirements.

    Format requirements (all must hold):
    1. Overall structure: <think>...</think><answer>...</answer>
    2. The <think> block must contain <step>...</step> tags.
    3. The step count must be within [step_min, step_max].

    Args:
        response: full model output
        step_min: minimum number of steps (default 2)
        step_max: maximum number of steps (default 6)
        verbose: print the failure reason when a check fails

    Returns:
        True if the format is valid, False otherwise.
    """
    # Check 1: overall <think>...</think><answer>...</answer> structure
    if not THINK_ANSWER_PATTERN.fullmatch(response or ""):
        if verbose:
            print(f"[format check] FAIL: overall structure is not <think>...</think><answer>...</answer>")
        return False

    # Check 2: can we extract <think> content?
    match = THINK_RE.search(response or "")
    if not match:
        if verbose:
            print(f"[format check] FAIL: cannot extract content inside <think>")
        return False

    think_content = match.group(1) or ""

    # Check 3: matched <step>/</step> tag counts
    open_step_count = len(re.findall(r'<step>', think_content))
    close_step_count = len(re.findall(r'</step>', think_content))

    if open_step_count != close_step_count:
        if verbose:
            print(f"[format check] FAIL: mismatched <step> and </step> counts ({open_step_count} vs {close_step_count})")
        return False

    # Reject consecutive </step></step> or <step><step>
    if re.search(r'</step>\s*</step>', think_content):
        if verbose:
            print(f"[format check] FAIL: found consecutive </step></step>")
        return False

    if re.search(r'<step>\s*<step>', think_content):
        if verbose:
            print(f"[format check] FAIL: found consecutive <step><step>")
        return False

    # Check 4: at least one <step> inside <think>
    steps = STEP_RE.findall(think_content)
    step_count = len([s for s in steps if s.strip()])

    if step_count == 0:
        if verbose:
            print(f"[format check] FAIL: no <step>...</step> inside <think>")
        return False

    # Check 5: step count must be in [step_min, step_max]
    if step_count < step_min:
        if verbose:
            print(f"[format check] FAIL: too few steps ({step_count} < {step_min})")
        return False

    if step_count > step_max:
        if verbose:
            print(f"[format check] FAIL: too many steps ({step_count} > {step_max})")
        return False

    if verbose:
        print(f"[format check] OK: {step_count} step(s) within [{step_min}, {step_max}]")
    return True


def compute_format_reward(
    response: str,
    response_length_tokens: int,
    step_min: int = 2,
    step_max: int = 6,
    l_min: int = 320,
    l_max: int = 520,
    r1_value: float = 0.5,
    beta_value: float = 0.5,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute Format Reward = r1 + beta.

    r1:
        - format valid (correct structure and step_count in [step_min, step_max]): r1 = r1_value (0.5)
        - format invalid: r1 = 0

    beta:
        - response length (in tokens) within [l_min, l_max]: beta = beta_value (0.5)
        - otherwise: beta = 0

    Args:
        response: full model output
        response_length_tokens: response length in tokens
        step_min: minimum step count (default 2)
        step_max: maximum step count (default 6)
        l_min: minimum token count (default 320)
        l_max: maximum token count (default 520)
        r1_value: r1 value when the format is valid (default 0.5)
        beta_value: beta value when the length is valid (default 0.5)
        verbose: print detailed info

    Returns:
        Dictionary with individual scores:
        {
            "format_reward": r1 + beta,
            "r1": r1,
            "beta": beta,
            "format_valid": 1.0 or 0.0,
            "length_valid": 1.0 or 0.0,
            "step_count": number of steps
        }
    """
    # Step 1: check the format
    format_valid = check_format_valid(response, step_min, step_max, verbose)

    # Step 2: compute r1
    r1 = r1_value if format_valid else 0.0

    # Step 3: compute beta (based on token length)
    length_valid = (l_min <= response_length_tokens <= l_max)
    beta = beta_value if length_valid else 0.0

    # Step 4: final format_reward
    format_reward = r1 + beta

    # Step count (for logging)
    step_count = len(_extract_steps(response))

    if verbose:
        print(f"[Format Reward] r1={r1} (format_valid={format_valid}), "
              f"beta={beta} (length={response_length_tokens}, range=[{l_min},{l_max}]), "
              f"total={format_reward}")

    return {
        "format_reward": format_reward,
        "r1": r1,
        "beta": beta,
        "format_valid": 1.0 if format_valid else 0.0,
        "length_valid": 1.0 if length_valid else 0.0,
        "step_count": float(step_count),
    }


# ==================== Accuracy Reward ====================

def accuracy_reward(response: str, ground_truth: str, data_type: str, problem_type: str) -> float:
    """
    Compute the accuracy reward (0.0 or 1.0).
    """
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response or "")
        given_answer = content_match.group(1).strip() if content_match else (response or "").strip()
        is_correct = _grade_answer(given_answer, (ground_truth or "").strip())
        return 1.0 if is_correct else 0.0
    except Exception:
        return 0.0


# ==================== vLLM communication ====================

def _extract_base_url(endpoint: str) -> str:
    """Extract the base URL from an endpoint for the OpenAI client."""
    endpoint = endpoint.rstrip("/")

    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[: -len("/chat/completions")]
    elif endpoint.endswith("/v1/chat/completions"):
        endpoint = endpoint[: -len("/v1/chat/completions")]

    if not endpoint.endswith("/v1"):
        endpoint = f"{endpoint.rstrip('/')}/v1"

    return endpoint


def _normalize_media_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://") or path.startswith("file://"):
        return path
    return f"file://{path}"


def _media_items(
    multi_modal_data: Optional[dict],
    media_format: str,
) -> List[dict]:
    """Convert multi-modal data to an OpenAI-compatible format for vLLM."""
    if not multi_modal_data or not isinstance(multi_modal_data, dict):
        return []

    items: List[dict] = []
    images = multi_modal_data.get("images") or []
    videos = multi_modal_data.get("videos") or []

    for image in images:
        items.append({"type": "image_url", "image_url": {"url": _normalize_media_url(image)}})
    for video in videos:
        items.append({"type": "video_url", "video_url": {"url": _normalize_media_url(video)}})

    return items


def _call_chat_completions_single(
    endpoint: str,
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    content: Any,
    assistant_prefix: Optional[str] = None,
) -> List[str]:
    """Single vLLM call (no retry)."""
    base_url = _extract_base_url(endpoint)
    client = _get_openai_client(base_url, timeout)

    outputs: List[str] = []

    if assistant_prefix:
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": assistant_prefix},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            extra_body={
                "add_generation_prompt": False,
                "continue_final_message": True,
            },
        )

        for choice in response.choices:
            if choice.message and choice.message.content:
                outputs.append(assistant_prefix + choice.message.content)

    else:
        messages = [{"role": "user", "content": content}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )

        for choice in response.choices:
            if choice.message and choice.message.content:
                outputs.append(choice.message.content)

    return outputs


def _call_chat_completions(
    endpoint: str,
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    multi_modal_data: Optional[dict],
    media_format: str,
    assistant_prefix: Optional[str] = None,
    fallback_endpoints: Optional[List[str]] = None,  # fallback endpoints
    max_retries: int = 2,  # max retries per endpoint
    retry_delay: float = 1.0,  # delay between retries (seconds)
) -> List[str]:
    """Call the chat-completions API via the OpenAI client (vLLM-compatible).

    Adds a fail-over mechanism:
    1. Try the primary endpoint first (up to max_retries times).
    2. If that fails, fall back through the other endpoints in order.
    """
    import time

    content_items = _media_items(multi_modal_data, media_format)

    if content_items:
        content_items.append({"type": "text", "text": prompt})
        content = content_items
    else:
        content = prompt

    # Build the endpoint list: primary + fallbacks
    endpoints_to_try = [endpoint]
    if fallback_endpoints:
        for fb in fallback_endpoints:
            if fb != endpoint and fb not in endpoints_to_try:
                endpoints_to_try.append(fb)

    last_error = None

    for ep_idx, current_endpoint in enumerate(endpoints_to_try):
        for attempt in range(max_retries):
            try:
                outputs = _call_chat_completions_single(
                    endpoint=current_endpoint,
                    model=model,
                    prompt=prompt,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    content=content,
                    assistant_prefix=assistant_prefix,
                )
                return outputs

            except Exception as e:
                last_error = e
                total_attempt = ep_idx * max_retries + attempt + 1
                total_max = len(endpoints_to_try) * max_retries

                if attempt < max_retries - 1:
                    # Same endpoint, retry available
                    print(f"[pg_reward] vLLM call failed ({total_attempt}/{total_max}): {type(e).__name__}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                elif ep_idx < len(endpoints_to_try) - 1:
                    # Current endpoint exhausted, switch to the next one
                    next_ep = endpoints_to_try[ep_idx + 1]
                    print(f"[pg_reward] vLLM {current_endpoint} failed, switching to {next_ep}...")
                    time.sleep(retry_delay)

    # All endpoints exhausted
    print(f"[pg_reward] vLLM call failed (all {len(endpoints_to_try)} endpoint(s) exhausted): {type(last_error).__name__}: {last_error}")
    return []


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_process_prompt(problem: str, prefix_steps: List[str]) -> tuple:
    """Build the message structure used for process-reward evaluation."""
    prefix = "".join([f"<step>{s}</step>" for s in prefix_steps])
    assistant_prefix = f"<think>{prefix}" if prefix else "<think>"

    user_prompt = (
        f"{problem}\n\n"
        "Please continue the reasoning from the given steps and provide the final answer with <answer>...</answer>."
    )

    return (user_prompt, assistant_prefix)


def _process_reward_for_sample(
    sample_idx: int,
    item: Dict[str, Any],
    ground_truth: str,
    data_type: str,
    problem_type: str,
    process_model_endpoints: List[str],  # supports multiple endpoints
    process_model_name: str,
    process_n: int,
    process_max_steps: int,
    process_temperature: float,
    process_top_p: float,
    process_max_tokens: int,
    process_timeout: float,
    process_media_format: str,
    process_use_prefill: bool = True,
    reward_log_dir: str = None,  # log directory
) -> float:
    """Compute the process reward: score each reasoning step via continuation solvability."""
    if not process_model_endpoints or process_n <= 0:
        return 0.0

    steps = _extract_steps(item.get("response", "") or "")

    if not steps:
        print(f"[Process Reward] no steps extracted, returning 0.0")
        return 0.0

    if process_max_steps > 0 and len(steps) > process_max_steps:
        steps = steps[:process_max_steps]

    problem = item.get("problem", "") or ""
    multi_modal_data = item.get("multi_modal_data")

    print(f"[Process Reward] scoring {len(steps)} step(s)")

    step_scores: List[float] = []
    prefix_steps: List[str] = []

    for i, step in enumerate(steps):
        prefix_steps.append(step)

        user_prompt, assistant_prefix = _build_process_prompt(problem, prefix_steps)

        # Load balancing: pick a primary endpoint at random.
        # The remaining endpoints are used as fallbacks on failure.
        selected_endpoint = _select_endpoint(process_model_endpoints)

        outputs = _call_chat_completions(
            endpoint=selected_endpoint,
            model=process_model_name,
            prompt=user_prompt,
            n=process_n,
            temperature=process_temperature,
            top_p=process_top_p,
            max_tokens=process_max_tokens,
            timeout=process_timeout,
            multi_modal_data=multi_modal_data,
            media_format=process_media_format,
            assistant_prefix=assistant_prefix if process_use_prefill else None,
            fallback_endpoints=process_model_endpoints,  # all endpoints serve as fallbacks
        )

        accs = [accuracy_reward(out, ground_truth, data_type, problem_type) for out in outputs]
        prefix_score = _mean(accs)
        step_scores.append(prefix_score)

        _log_process_rollout(
            sample_idx=sample_idx,
            step_idx=i,
            prefix_steps=list(prefix_steps),
            outputs=outputs,
            accs=accs,
            prefix_score=prefix_score,
            log_dir=reward_log_dir,
        )

    final_score = _mean(step_scores)
    print(f"[Process Reward] final score: {final_score:.3f}")

    return final_score


# ==================== Main entry ====================

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    # Format Reward (R1) parameters
    step_min: int = 2,
    step_max: int = 6,
    l_min: int = 320,
    l_max: int = 520,
    r1_value: float = 0.5,
    beta_value: float = 0.5,
    # Penalty parameters: applies to the CoT reward
    penalty: bool = True,      # True: if cot_reward < 0.5, cot_reward = -step_bonus (fixed penalty)
    # Step-Bonus parameters
    use_step_bonus: bool = True,  # whether step bonus is enabled
    alpha: float = 0.5,           # step-bonus scale
    # R2 parameters: acc_weight + cot_weight = 1
    acc_weight: float = 1.0,   # weight of the accuracy reward
    cot_weight: float = 0.0,   # weight of the CoT/process reward
    # Process Reward (CoT) parameters
    process_model_endpoint: Optional[str] = None,
    process_model_name: str = "",
    process_n: int = 4,
    process_max_steps: int = 6,
    process_temperature: float = 1.0,
    process_top_p: float = 1.0,
    process_max_tokens: int = 2048,
    process_timeout: float = 60.0,
    process_media_format: str = "openai",
    process_use_prefill: bool = True,
    # Reward log directory
    reward_log_dir: Optional[str] = None,  # None means use the default path
) -> List[Dict[str, float]]:
    """
    Compute the overall reward.

    ==================== Reward breakdown ====================

    1. Format Reward (R1) = r1 + beta
       - r1: r1_value (0.5) if format valid, else 0
       - beta: beta_value (0.5) if token length in [l_min, l_max], else 0
       - Valid format: <think><step>...</step>...</think><answer>...</answer>
                       and step_min <= step_count <= step_max

    2. Penalty (applied to the CoT reward)
       - If penalty = True and cot_reward < 0.5:
           cot_reward = -step_bonus (fixed penalty tied to step_bonus)
       - If penalty = False: cot_reward is left unchanged.

    3. Step Bonus (optional, enabled when use_step_bonus=True)
       - s(step) = (step_count - step_min) / (step_max - step_min)  # normalized
       - g(step) = sqrt(s(step))                                    # square root
       - step_bonus = alpha * g(step)                               # scale factor

    4. R2 = acc_weight * acc_reward + cot_weight * cot_reward + step_bonus
       - acc_weight + cot_weight = 1 (required)
       - acc_reward: 1.0 if the final answer is correct, else 0.0
       - cot_reward: step-wise quality score from continuation rollouts via vLLM
       - step_bonus: only added when use_step_bonus=True

       Fast paths (save compute):
       - acc_weight=1, cot_weight=0: only acc_reward is computed, cot_reward is skipped
       - acc_weight=0, cot_weight=1: only cot_reward is computed, acc_reward is skipped
       - otherwise: both are computed

    ==================== Final score ====================
    If format_valid = False:
        overall = 0  # invalid format => zero reward
    Else:
        overall = R1 + R2
                = (r1 + beta) + (acc_weight * acc_reward + cot_weight * cot_reward + step_bonus)

    Notes:
    - step_bonus is only included if use_step_bonus=True.
    - If penalty=True and cot_reward < 0.5, cot_reward is overwritten with -step_bonus.

    ==================== Returns ====================
    Each sample is returned as a dictionary:
    {
        "overall": overall score,
        "format_reward": format reward (r1 + beta),
        "r1": r1,
        "beta": beta,
        "format_valid": whether the format is valid,
        "length_valid": whether the length is valid,
        "step_count": number of steps,
        "acc_reward": accuracy reward,
        "cot_reward": CoT / process reward,
        "step_bonus": step bonus (when use_step_bonus=True),
        "r2": acc_weight * acc_reward + cot_weight * cot_reward + step_bonus,
    }
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    # Validate weight parameters
    assert abs(acc_weight + cot_weight - 1.0) < 1e-6, \
        f"acc_weight ({acc_weight}) + cot_weight ({cot_weight}) must equal 1.0"
    assert 0.0 <= acc_weight <= 1.0, f"acc_weight ({acc_weight}) must be within [0, 1]"
    assert 0.0 <= cot_weight <= 1.0, f"cot_weight ({cot_weight}) must be within [0, 1]"

    # Check penalty / step_bonus dependency
    assert not (penalty and not use_step_bonus), \
        f"penalty=True requires use_step_bonus=True (the penalty value equals -step_bonus)"

    global _global_step_counter
    _global_step_counter += 1

    # Parse endpoints (supports comma-separated list)
    process_model_endpoints = _parse_endpoints(process_model_endpoint or "")

    print(f"\n[compute_score] Step {_global_step_counter}: computing reward for {len(reward_inputs)} sample(s)")
    print(f"[compute_score] R1 params: step_min={step_min}, step_max={step_max}, "
          f"l_min={l_min}, l_max={l_max}, r1={r1_value}, beta={beta_value}")
    print(f"[compute_score] Penalty: {penalty} (if cot_reward < 0.5 -> cot_reward = -step_bonus)")
    print(f"[compute_score] Step Bonus: use_step_bonus={use_step_bonus}, alpha={alpha}")
    print(f"[compute_score] R2 params: acc_weight={acc_weight}, cot_weight={cot_weight}")
    if process_model_endpoints:
        print(f"[compute_score] vLLM endpoints ({len(process_model_endpoints)}): {process_model_endpoints}")

    results: List[Dict[str, float]] = []
    # pdb.set_trace()

    # DEBUG: dump the structure of reward_inputs
    if len(reward_inputs) > 0:
        print(f"\n[DEBUG] reward_inputs[0] keys: {list(reward_inputs[0].keys())}")
        print(f"[DEBUG] reward_inputs[0] response first 100 chars: {repr(str(reward_inputs[0].get('response', ''))[:100])}")
        if 'prompt' in reward_inputs[0]:
            print(f"[DEBUG] reward_inputs[0] prompt first 200 chars: {repr(str(reward_inputs[0].get('prompt', ''))[:200])}")

    # ========== Per-sample reward computation ==========
    def _compute_single_reward(idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """Compute the reward for a single sample and return (idx, result)."""
        try:
            # ========== 1. Extract and preprocess inputs ==========
            def safe_str(val) -> str:
                if val is None:
                    return ""
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="replace")
                try:
                    s = str(val)
                    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                except Exception:
                    return ""

            raw_response = safe_str(item.get("response", ""))
            response = _normalize_tags(raw_response)

            raw_gt = safe_str(item.get("ground_truth", ""))
            gt_extracted = extract_answer(raw_gt) or raw_gt

            data_type = safe_str(item.get("data_type", ""))
            problem_type = safe_str(item.get("problem_type", ""))

            response_length_tokens = item.get("response_length", 0)

            # DEBUG: dump info for the first sample only
            if idx == 0:
                print(f"\n[compute_score] sample 0 details:")
                print(f"  - response length: {len(raw_response)} chars, {response_length_tokens} tokens")
                print(f"  - ground_truth: {gt_extracted}")

            # ========== 2. Compute Format Reward ==========
            format_result = compute_format_reward(
                response=response,
                response_length_tokens=response_length_tokens,
                step_min=step_min,
                step_max=step_max,
                l_min=l_min,
                l_max=l_max,
                r1_value=r1_value,
                beta_value=beta_value,
                verbose=(idx == 0),
            )

            # ========== 3. Short-circuit on invalid format: return 0 and skip vLLM ==========
            if format_result["format_valid"] == 0.0:
                if idx == 0:
                    print(f"[compute_score] sample 0: format_valid=0, skipping further computation, overall=0")

                reward_result = {
                    "overall": 0.0,
                    "format_reward": 0.0,  # invalid format => zero format reward
                    "r1": 0.0,
                    "beta": 0.0,  # also zeroed for consistency
                    "format_valid": 0.0,
                    "length_valid": float(format_result["length_valid"]),
                    "step_count": float(format_result["step_count"]),
                    "acc_reward_raw": 0.0,
                    "acc_reward": 0.0,
                    "cot_reward_raw": 0.0,
                    "cot_reward": 0.0,
                    "step_bonus": 0.0,
                    "r2": 0.0,
                    # GDPO: per-component scores (used for independent normalization)
                    "gdpo_r1": 0.0,
                    "gdpo_beta": 0.0,
                    "gdpo_cot": 0.0,
                    "gdpo_step_bonus": 0.0,
                }
                _log_reward(idx, item, reward_result, normalized_response=response, log_dir=reward_log_dir)
                return (idx, reward_result)

            # ========== 4. Format valid, compute R2 = acc_weight * acc_reward + cot_weight * cot_reward ==========
            acc_reward_value = 0.0
            cot_reward_value = 0.0

            # Compute acc_reward only when acc_weight > 0
            if acc_weight > 0:
                acc_reward_value = accuracy_reward(response, gt_extracted, data_type, problem_type)
                if idx == 0:
                    print(f"[compute_score] sample 0 acc_reward: {acc_reward_value}")

            # Compute cot_reward only when cot_weight > 0 (saves vLLM round-trips)
            if cot_weight > 0 and process_model_endpoints:
                cot_reward_value = _process_reward_for_sample(
                    sample_idx=idx,
                    item=item,
                    ground_truth=gt_extracted,
                    data_type=data_type,
                    problem_type=problem_type,
                    process_model_endpoints=process_model_endpoints,
                    process_model_name=process_model_name,
                    process_n=process_n,
                    process_max_steps=process_max_steps,
                    process_temperature=process_temperature,
                    process_top_p=process_top_p,
                    process_max_tokens=process_max_tokens,
                    process_timeout=process_timeout,
                    process_media_format=process_media_format,
                    process_use_prefill=process_use_prefill,
                    reward_log_dir=reward_log_dir,
                )
                if idx == 0:
                    print(f"[compute_score] sample 0 cot_reward (raw): {cot_reward_value}")

            # ========== 4.5 Step Bonus (optional) ==========
            # Compute step_bonus first, since the penalty uses it.
            step_bonus_value = 0.0
            if use_step_bonus:
                step_count = int(format_result["step_count"])
                # s(step) = (step_count - step_min) / (step_max - step_min)
                if step_max > step_min:  # avoid division by zero
                    s_step = (step_count - step_min) / (step_max - step_min)
                    s_step = max(0.0, min(1.0, s_step))  # clip to [0, 1]
                else:
                    s_step = 0.0
                # g(step) = sqrt(s(step))
                g_step = math.sqrt(s_step)
                # step_bonus = alpha * g(step)
                step_bonus_value = alpha * g_step

                if idx == 0:
                    print(f"[compute_score] sample 0 step_bonus: s={s_step:.3f}, g={g_step:.3f}, bonus={step_bonus_value:.3f}")

            # ========== 4.6 Penalty gate ==========
            # Save raw values (pre-penalty)
            acc_reward_raw = acc_reward_value
            cot_reward_raw = cot_reward_value

            # Acc penalty: if penalty=True and acc_reward != 1 (wrong answer), set acc_reward = -step_bonus
            if acc_weight > 0 and penalty and acc_reward_value < 1.0:
                acc_reward_value = -step_bonus_value   # fixed penalty value = -step_bonus
                if idx == 0:
                    print(f"[compute_score] sample 0 acc_reward penalty: {acc_reward_raw} -> {acc_reward_value} (wrong answer, penalty=-step_bonus)")

            if idx == 0 and acc_weight > 0:
                print(f"[compute_score] sample 0 acc_reward (final): {acc_reward_value}")

            # CoT penalty: if penalty=True and cot_reward < 0.5, set cot_reward = -step_bonus
            if cot_weight > 0 and penalty and cot_reward_value < 0.5:
                cot_reward_value = -step_bonus_value * 1  # fixed penalty value = -step_bonus
                if idx == 0:
                    print(f"[compute_score] sample 0 cot_reward penalty: {cot_reward_raw} -> {cot_reward_value} (< 0.5, penalty=-step_bonus)")

            if idx == 0 and cot_weight > 0:
                print(f"[compute_score] sample 0 cot_reward (final): {cot_reward_value}")

            # R2 = acc_weight * acc_reward + cot_weight * cot_reward + step_bonus
            r2 = acc_weight * acc_reward_value + cot_weight * cot_reward_value + step_bonus_value

            # ========== 5. Final overall score ==========
            r1_total = format_result["format_reward"]  # R1 = r1 + beta
            overall = r1_total + r2

            if idx == 0:
                print(f"[compute_score] sample 0 final:")
                print(f"  - R1 (format_reward): {r1_total} (r1={format_result['r1']}, beta={format_result['beta']})")
                print(f"  - R2: {r2} (acc={acc_reward_value}, cot={cot_reward_value}, step_bonus={step_bonus_value})")
                print(f"  - overall: {overall}")

            reward_result = {
                "overall": float(overall),
                "format_reward": float(format_result["format_reward"]),
                "r1": float(format_result["r1"]),
                "beta": float(format_result["beta"]),
                "format_valid": float(format_result["format_valid"]),
                "length_valid": float(format_result["length_valid"]),
                "step_count": float(format_result["step_count"]),
                "acc_reward_raw": float(acc_reward_raw),  # raw acc_reward (pre-penalty)
                "acc_reward": float(acc_reward_value),    # final acc_reward (post-penalty)
                "cot_reward_raw": float(cot_reward_raw),  # raw cot_reward (pre-penalty)
                "cot_reward": float(cot_reward_value),    # final cot_reward (post-penalty)
                "step_bonus": float(step_bonus_value),
                "r2": float(r2),
                # GDPO: per-component scores (used for independent normalization)
                "gdpo_r1": float(format_result["r1"]),
                "gdpo_beta": float(format_result["beta"]),
                "gdpo_cot": float(cot_reward_value),  # post-penalty cot_reward
                "gdpo_step_bonus": float(step_bonus_value),
            }
            _log_reward(idx, item, reward_result, normalized_response=response, log_dir=reward_log_dir)
            return (idx, reward_result)

        except Exception as e:
            print(f"[compute_score] sample {idx} failed: {type(e).__name__}: {e}")
            error_result = {
                "overall": 0.0,
                "format_reward": 0.0,
                "r1": 0.0,
                "beta": 0.0,
                "format_valid": 0.0,
                "length_valid": 0.0,
                "step_count": 0.0,
                "acc_reward_raw": 0.0,
                "acc_reward": 0.0,
                "cot_reward_raw": 0.0,
                "cot_reward": 0.0,
                "step_bonus": 0.0,
                "r2": 0.0,
                # GDPO: per-component scores
                "gdpo_r1": 0.0,
                "gdpo_beta": 0.0,
                "gdpo_cot": 0.0,
                "gdpo_step_bonus": 0.0,
            }
            _log_reward(idx, item, error_result, log_dir=reward_log_dir)
            return (idx, error_result)

    # ========== Process all samples (possibly in parallel) ==========
    use_parallel = cot_weight > 0 and process_model_endpoints and len(reward_inputs) > 1

    if use_parallel:
        # Parallel mode: use ThreadPoolExecutor
        # Use at most 12 worker threads (aligned with typical batch sizes)
        max_workers = min(len(reward_inputs), 12)
        print(f"[compute_score] parallel mode: {max_workers} worker(s) for {len(reward_inputs)} sample(s)")

        # Pre-allocate the result list so we can preserve order
        results = [None] * len(reward_inputs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_reward, idx, item): idx
                for idx, item in enumerate(reward_inputs)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        assert all(r is not None for r in results), "some samples failed to compute"
    else:
        # Serial mode (no cot_reward needed, or only one sample)
        print(f"[compute_score] serial mode: processing {len(reward_inputs)} sample(s)")
        results = []
        for idx, item in enumerate(reward_inputs):
            _, result = _compute_single_reward(idx, item)
            results.append(result)

    print(f"[compute_score] done, processed {len(results)} sample(s)")

    # Print per-endpoint call statistics (load balance)
    if _endpoint_call_count:
        print(f"[compute_score] vLLM endpoint call stats: {dict(_endpoint_call_count)}")

    return results
