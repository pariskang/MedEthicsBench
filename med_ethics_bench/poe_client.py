from __future__ import annotations

"""
poe_client.py — Unified LLM client: Poe + Azure OpenAI backends.

Backend is selected via Config.backend:
  "poe"   → Poe /v1/responses  (default, current behaviour)
  "azure" → Azure OpenAI chat.completions  (new)

Both share the same respond() / respond_json() interface so retriever,
question_generator, and evaluator need zero changes.

High-concurrency robustness:
  - Manual retry loop with exponential back-off + jitter (no tenacity dep for this)
  - Rate-limit (429) handled separately with longer sleep
  - Per-request timeout passed to OpenAI client
  - Thread-safe telemetry counter
  - json-repair + 8-candidate local JSON recovery pipeline
  - Targeted repair: only missing keys re-requested (not full regeneration)
"""

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError

try:
    from json_repair import repair_json as _json_repair_fn
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

log = logging.getLogger(__name__)

_RETRYABLE_NETWORK = (APITimeoutError, APIConnectionError, OSError)

# ── Model capability registry ─────────────────────────────────────────────────
# Models that support the `reasoning` / `reasoning_effort` parameter.
# Any model NOT in this set will have reasoning params silently stripped.
# Checked via prefix-match so "claude-3-7-sonnet-20250219" matches "claude".
# Add new reasoning-capable models here as they become available.
_REASONING_MODEL_PREFIXES: tuple = (
    "claude",
    "o1", "o3", "o4",           # OpenAI o-series
    "deepseek-r",               # DeepSeek R-series
    "qwq",                      # Qwen QwQ
    "grok-3",                   # xAI Grok 3
)

# Per-model override cache: populated at runtime when a 400 "UnsupportedParams"
# is received, so any new model automatically gets stripped on the next call.
_NO_REASONING_MODELS: set = set()
_NO_REASONING_LOCK = threading.Lock()


def _supports_reasoning(model: str) -> bool:
    """Return True if this model is known to support reasoning_effort."""
    name = model.lower()
    with _NO_REASONING_LOCK:
        if name in _NO_REASONING_MODELS:
            return False
    return any(name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _mark_no_reasoning(model: str) -> None:
    """Permanently mark a model as not supporting reasoning params (for this process)."""
    with _NO_REASONING_LOCK:
        _NO_REASONING_MODELS.add(model.lower())
    log.info("已将 %s 标记为不支持 reasoning 参数，后续调用将自动跳过。", model)


def _is_unsupported_param_error(e: Exception) -> bool:
    """Detect 400 errors caused by unsupported parameters (not auth/quota issues)."""
    msg = str(e).lower()
    return isinstance(e, BadRequestError) and (
        "unsupportedparams" in msg
        or "unsupported" in msg and "param" in msg
        or "reasoning_effort" in msg
        or "drop_params" in msg
    )


# ── Telemetry ─────────────────────────────────────────────────────────────────
class _Telemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.errors = 0
        self.rate_limits = 0

    def record(self, *, error: bool = False, rate_limit: bool = False) -> None:
        with self._lock:
            self.total += 1
            if error:
                self.errors += 1
            if rate_limit:
                self.rate_limits += 1

    def summary(self) -> str:
        return (f"requests={self.total}, errors={self.errors}, "
                f"rate_limits={self.rate_limits}")


TELEMETRY = _Telemetry()


@dataclass
class WebSource:
    title: str = ""
    url: str = ""


@dataclass
class ResponseResult:
    text: str
    reasoning_summary: Optional[str] = None
    web_sources: List[WebSource] = field(default_factory=list)
    raw: Any = None


# ── Client ────────────────────────────────────────────────────────────────────
class PoeClient:
    """Unified LLM client — identical interface for Poe and Azure OpenAI."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.poe.com/v1",
        backend: str = "poe",
        request_timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        if not api_key:
            raise ValueError("PoeClient 需要非空 api_key。")
        self._backend = backend.lower()
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=0,   # retry logic is our own
        )

    # ------------------------------------------------------------------
    # respond() — public interface
    # ------------------------------------------------------------------
    def respond(
        self,
        *,
        model: str,
        system: str,
        user: str,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: str = "none",
        enable_web_search: bool = False,
        max_output_tokens: int = 1024,
    ) -> ResponseResult:
        # Strip reasoning params immediately for known-incompatible models
        effective_reasoning = reasoning_effort if _supports_reasoning(model) else None

        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                if self._backend == "azure":
                    result = self._respond_azure(
                        model=model, system=system, user=user,
                        reasoning_effort=effective_reasoning,
                        max_output_tokens=max_output_tokens,
                    )
                else:
                    result = self._respond_poe(
                        model=model, system=system, user=user,
                        reasoning_effort=effective_reasoning,
                        reasoning_summary=reasoning_summary,
                        enable_web_search=enable_web_search,
                        max_output_tokens=max_output_tokens,
                    )
                TELEMETRY.record()
                return result

            except BadRequestError as e:
                if _is_unsupported_param_error(e) and effective_reasoning is not None:
                    # Model doesn't support reasoning params → strip and retry immediately
                    _mark_no_reasoning(model)
                    effective_reasoning = None
                    log.warning(
                        "模型 %s 不支持 reasoning 参数，已自动剔除，立即重试（不计入重试次数）。",
                        model,
                    )
                    continue  # retry without sleeping
                # Other 400 errors (bad prompt, auth, etc.) — don't retry
                TELEMETRY.record(error=True)
                raise

            except RateLimitError as e:
                last_err = e
                TELEMETRY.record(error=True, rate_limit=True)
                wait = min(5 * (2 ** attempt) + random.uniform(0, 3), 120.0)
                log.warning("Rate limit (attempt %d/%d). Sleeping %.1fs …",
                            attempt + 1, self._max_retries, wait)
                time.sleep(wait)

            except _RETRYABLE_NETWORK as e:
                last_err = e
                TELEMETRY.record(error=True)
                wait = min(2 * (2 ** attempt) + random.uniform(0, 2), 60.0)
                log.warning("Network error (attempt %d/%d). Sleeping %.1fs → %s",
                            attempt + 1, self._max_retries, wait, e)
                time.sleep(wait)

            except Exception:
                TELEMETRY.record(error=True)
                raise

        raise RuntimeError(
            f"LLM 调用失败，已重试 {self._max_retries} 次。最后错误：{last_err}"
        ) from last_err

    # ------------------------------------------------------------------
    # Poe /v1/responses backend
    # ------------------------------------------------------------------
    def _respond_poe(
        self, *, model, system, user, reasoning_effort,
        reasoning_summary, enable_web_search, max_output_tokens,
    ) -> ResponseResult:
        # If reasoning is not needed, use the simpler responses.create interface.
        # For models that only support chat.completions (e.g. Gemma via Poe),
        # responses.create still works — Poe routes it appropriately.
        tools: List[Dict[str, Any]] = []
        include: List[str] = []
        if enable_web_search:
            tools.append({"type": "web_search_preview"})
            include.append("web_search_call.action.sources")

        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_output_tokens,
        }
        # Only send reasoning block when explicitly requested AND model supports it
        if reasoning_effort and reasoning_effort not in {"none", "off", "false", "0"}:
            kwargs["reasoning"] = {"effort": reasoning_effort,
                                   "summary": reasoning_summary}
        if tools:
            kwargs["tools"] = tools
        if include:
            kwargs["include"] = include

        resp = self._client.responses.create(**kwargs)
        text = _extract_response_text(resp)
        result = ResponseResult(text=text, raw=resp)

        for item in (_safe_get(resp, "output") or []):
            itype = _safe_get(item, "type")
            if itype == "web_search_call":
                for s in (_safe_get(_safe_get(item, "action"), "sources") or []):
                    result.web_sources.append(
                        WebSource(title=_safe_get(s, "title") or "",
                                  url=_safe_get(s, "url") or ""))
            elif itype == "reasoning":
                smry = _extract_reasoning_summary(_safe_get(item, "summary"))
                if smry:
                    result.reasoning_summary = smry

        if not result.reasoning_summary:
            for key in ("reasoning_summary", "summary"):
                smry = _extract_reasoning_summary(_safe_get(resp, key))
                if smry:
                    result.reasoning_summary = smry
                    break

        if not result.text:
            log.warning("Poe: 提取到空文本。preview=%s", _preview_response(resp))
        return result

    # ------------------------------------------------------------------
    # Azure OpenAI chat.completions backend
    # ------------------------------------------------------------------
    def _respond_azure(
        self, *, model, system, user, reasoning_effort, max_output_tokens,
    ) -> ResponseResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        }
        # o-series / reasoning models accept reasoning_effort
        if reasoning_effort and str(reasoning_effort).lower() not in {
            "none", "off", "false", "0"
        }:
            kwargs["reasoning_effort"] = reasoning_effort

        resp = self._client.chat.completions.create(**kwargs)
        text = ""
        reasoning_text: Optional[str] = None
        choice = resp.choices[0] if resp.choices else None
        finish_reason = choice.finish_reason if choice else "unknown"

        if choice:
            msg = choice.message
            text = (msg.content or "").strip()
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning_text = str(msg.reasoning_content).strip()

        if finish_reason == "length" and not text:
            # Reasoning model consumed all tokens in the thinking phase.
            # Auto-retry with 3× the token budget (once).
            expanded = min(max_output_tokens * 3, 16000)
            log.warning(
                "Azure: finish_reason=length, 文本为空（model=%s）。"
                "自动扩容至 %d tokens 重试。",
                model, expanded,
            )
            kwargs["max_completion_tokens"] = expanded
            resp2 = self._client.chat.completions.create(**kwargs)
            choice2 = resp2.choices[0] if resp2.choices else None
            if choice2:
                text = (choice2.message.content or "").strip()
                if hasattr(choice2.message, "reasoning_content") and choice2.message.reasoning_content:
                    reasoning_text = str(choice2.message.reasoning_content).strip()
            if text:
                log.info("Azure 扩容重试成功，获得 %d 字符。", len(text))
                return ResponseResult(text=text, reasoning_summary=reasoning_text, raw=resp2)

        if finish_reason == "length" and text:
            # Got partial content — pass it to the JSON repair pipeline
            log.warning(
                "Azure: finish_reason=length，内容截断（model=%s，获得 %d 字符）。"
                "将尝试 JSON 修复。",
                model, len(text),
            )

        if not text:
            log.warning(
                "Azure: 提取到空文本。model=%s, finish_reason=%s",
                model, finish_reason,
            )
        return ResponseResult(text=text, reasoning_summary=reasoning_text, raw=resp)

    # ------------------------------------------------------------------
    # respond_json()
    # ------------------------------------------------------------------
    def respond_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        enable_web_search: bool = False,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: str = "none",
        max_output_tokens: int = 1800,
        required_keys: Optional[Sequence[str]] = None,
        repair_on_fail: bool = True,
    ) -> Dict[str, Any]:
        required_keys = list(required_keys or [])

        res = self.respond(
            model=model, system=system, user=user,
            enable_web_search=enable_web_search,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_output_tokens,
        )

        # Pass 1: strict
        first_err: Optional[Exception] = None
        try:
            data = _extract_json_object(res.text, required_keys=required_keys)
            return _attach_meta(data, res)
        except Exception as e:
            first_err = e

        # Pass 2: lenient — detect missing keys
        partial_data: Optional[Dict[str, Any]] = None
        missing_keys: List[str] = []
        try:
            partial_data = _extract_json_object(res.text, required_keys=[])
            if isinstance(partial_data, dict):
                missing_keys = [k for k in required_keys if k not in partial_data]
        except Exception:
            pass

        if not repair_on_fail:
            raise ValueError(
                f"JSON 解析失败：{first_err}\n"
                f"preview: {_preview_text_or_response(res.text, res.raw)}"
            ) from first_err

        log.info("首次 JSON 解析失败（缺少键: %s），发起修复请求。",
                 missing_keys or "解析错误")

        repaired = self._repair_to_json(
            model=model, original_system=system, original_user=user,
            bad_output=res.text, required_keys=required_keys,
            missing_keys=missing_keys, partial_data=partial_data,
            max_output_tokens=max_output_tokens,
        )
        try:
            data = _extract_json_object(repaired.text, required_keys=required_keys)
            return _attach_meta(data, repaired)
        except Exception as e:
            raise ValueError(
                f"JSON 解析失败：无法解析为 JSON。最后错误：{e}\n"
                f"原文前 500 字：{_preview_text_or_response(repaired.text, repaired.raw)}"
            ) from e

    # ------------------------------------------------------------------
    # _repair_to_json
    # ------------------------------------------------------------------
    def _repair_to_json(
        self, *, model, original_system, original_user, bad_output,
        required_keys, missing_keys=None, partial_data=None, max_output_tokens,
    ) -> ResponseResult:
        repair_tokens = max(max_output_tokens, 5000)
        missing_keys = missing_keys or []

        if missing_keys and partial_data:
            repair_system = (
                "You are a JSON completion specialist. "
                "Output ONLY a JSON object containing the specified missing fields. "
                "No markdown, no explanations, no extra keys."
            )
            existing = json.dumps(
                {k: v for k, v in partial_data.items()
                 if k not in ("_web_sources", "_reasoning_summary")},
                ensure_ascii=False,
            )[:2000]
            repair_user = (
                f"Missing fields to generate: {missing_keys}\n\n"
                f"Existing partial JSON (context only):\n{existing}\n\n"
                "Output ONLY a JSON object with the missing fields. "
                "First char '{{', last char '}}'.\n"
                f"Example:\n{_missing_keys_example(missing_keys)}"
            )
        else:
            repair_system = (
                "You are a JSON repair specialist. "
                "Output ONE valid, complete JSON object only. "
                "No markdown fences, no text outside the JSON."
            )
            all_keys = ", ".join(required_keys) if required_keys else "stem, reference_answer, key_points"
            repair_user = (
                f"Required top-level JSON keys: {all_keys}\n\n"
                "The JSON below is MALFORMED or TRUNCATED. "
                "Repair into a valid, complete JSON object. "
                "Complete truncated strings plausibly.\n\n"
                f"=== Malformed ===\n{bad_output[:4500]}\n=== End ===\n\n"
                "Return ONLY the repaired JSON. First '{{', last '}}'."
            )

        return self.respond(
            model=model, system=repair_system, user=repair_user,
            enable_web_search=False, reasoning_effort=None,
            reasoning_summary="none", max_output_tokens=repair_tokens,
        )


# ── JSON extraction helpers ───────────────────────────────────────────────────
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_TRAILING_COMMAS = re.compile(r",\s*([}\]])")


def _extract_json_object(
    text: str, required_keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("模型返回为空字符串。")
    required_keys = list(required_keys or [])

    raw = text.strip()
    unfenced = _CODE_FENCE.sub("", raw).strip()
    candidates = [raw, unfenced]
    obj = _slice_first_balanced_json_object(unfenced)
    if obj:
        candidates.append(obj)
    left, right = unfenced.find("{"), unfenced.rfind("}")
    if left != -1 and right > left:
        candidates.append(unfenced[left: right + 1])

    seen: set = set()
    last_err: Optional[Exception] = None
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        for variant in _repair_json_candidates(cand):
            try:
                data = json.loads(variant)
                if not isinstance(data, dict):
                    raise ValueError("顶层不是 JSON object。")
                missing = [k for k in required_keys if k not in data]
                if missing:
                    raise ValueError(f"缺少顶层键：{missing}")
                return data
            except Exception as e:
                last_err = e
    raise ValueError(f"无法解析为 JSON。最后错误：{last_err}")


def _repair_json_candidates(s: str) -> Iterable[str]:
    if _HAS_JSON_REPAIR:
        try:
            r = _json_repair_fn(s, return_objects=False)
            if r and isinstance(r, str):
                yield r
        except Exception:
            pass
    yield s
    s1 = s.replace("\ufeff", "").strip()
    yield s1
    s2 = _TRAILING_COMMAS.sub(r"\1", s1)
    yield s2
    s3 = _close_unbalanced_json(s2)
    yield s3
    s4 = _escape_unescaped_quotes_in_json_strings(s3)
    yield s4
    s5 = _TRAILING_COMMAS.sub(r"\1", s4)
    yield s5
    s6 = _close_unbalanced_json(s5)
    yield s6
    if _HAS_JSON_REPAIR:
        try:
            r2 = _json_repair_fn(s6, return_objects=False)
            if r2 and isinstance(r2, str):
                yield r2
        except Exception:
            pass


def _close_unbalanced_json(s: str) -> str:
    stack: list = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    suffix = '"' if in_string else ""
    for opener in reversed(stack):
        suffix += "}" if opener == "{" else "]"
    return s + suffix


def _escape_unescaped_quotes_in_json_strings(s: str) -> str:
    out: List[str] = []
    in_string = escape = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escape = False
            i += 1
            continue
        if escape:
            out.append(ch); escape = False; i += 1; continue
        if ch == "\\":
            out.append(ch); escape = True; i += 1; continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            nxt = s[j] if j < n else ""
            if nxt in {",", "}", "]", ":", ""}:
                out.append(ch); in_string = False
            else:
                out.append(r'\"')
            i += 1; continue
        out.append(r"\n" if ch == "\n" else r"\r" if ch == "\r" else r"\t" if ch == "\t" else ch)
        i += 1
    if in_string:
        out.append('"')
    return "".join(out)


def _slice_first_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = in_string = escape = False
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _missing_keys_example(missing_keys: List[str]) -> str:
    examples: Dict[str, Any] = {
        "expected_principles": ["自主原则", "不伤害原则", "公正原则"],
        "expected_stakeholders": ["患者", "家属", "主治医师", "伦理委员会"],
        "common_pitfalls": [
            "把美国/国际框架直接套用到本国语境",
            "把法律允许等同于伦理最优",
            "忽略医院伦理委员会的程序义务",
            "忽视患者自主权与家属代理权的层级关系",
        ],
        "requires_reasoning_trace": True,
        "stem": "题干文本...",
        "context": "背景材料...",
        "reference_answer": "参考答案...",
        "key_points": [{"id": "kp1", "content": "...", "weight": 1.0,
                        "required": True, "category": "principle"}],
        "options": None,
    }
    return json.dumps({k: examples[k] for k in missing_keys if k in examples},
                      ensure_ascii=False, indent=2)


def _attach_meta(data: Dict[str, Any], res: ResponseResult) -> Dict[str, Any]:
    data.setdefault("_web_sources",
                    [{"title": s.title, "url": s.url} for s in res.web_sources])
    if res.reasoning_summary:
        data.setdefault("_reasoning_summary", res.reasoning_summary)
    return data


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_content_text(content: Any) -> Iterable[str]:
    if content is None:
        return []
    items = content if isinstance(content, list) else [content]
    out: List[str] = []
    for part in items:
        ptype = _safe_get(part, "type")
        if ptype in {"output_text", "text", "input_text"}:
            txt = _safe_get(part, "text")
            if isinstance(txt, str) and txt.strip():
                out.append(txt); continue
        for key in ("text", "value", "content"):
            val = _safe_get(part, key)
            if isinstance(val, str) and val.strip():
                out.append(val); break
    return out


def _extract_response_text(resp: Any) -> str:
    for key in ("output_text", "text", "content"):
        val = _safe_get(resp, key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    output = _safe_get(resp, "output") or []
    texts: List[str] = []
    for item in output:
        itype = _safe_get(item, "type")
        if itype in {"message", "output_message", "response"}:
            texts.extend([t for t in _iter_content_text(_safe_get(item, "content")) if t.strip()])
        elif itype in {"output_text", "text"}:
            txt = _safe_get(item, "text")
            if isinstance(txt, str) and txt.strip():
                texts.append(txt)
        else:
            for key in ("text", "content"):
                val = _safe_get(item, key)
                if isinstance(val, str) and val.strip():
                    texts.append(val); break
                texts.extend([t for t in _iter_content_text(val) if t.strip()])
    joined = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
    if joined:
        return joined
    for ch in (_safe_get(resp, "choices") or []):
        msg = _safe_get(ch, "message") or {}
        content = _safe_get(msg, "content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        joined = "\n".join(t.strip() for t in _iter_content_text(content) if t.strip()).strip()
        if joined:
            return joined
    try:
        recovered = _deep_find_text(_to_plain(resp))
        if recovered:
            return recovered
    except Exception:
        pass
    return ""


def _extract_reasoning_summary(summary_obj: Any) -> Optional[str]:
    if summary_obj is None:
        return None
    if isinstance(summary_obj, str) and summary_obj.strip():
        return summary_obj.strip()
    if isinstance(summary_obj, list):
        parts = [_safe_get(x, "text") for x in summary_obj]
        parts = [p for p in parts if isinstance(p, str) and p.strip()]
        if parts:
            return "\n".join(p.strip() for p in parts)
    txt = _safe_get(summary_obj, "text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return None


def _to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return _to_plain(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return str(obj)


def _deep_find_text(obj: Any) -> str:
    texts: List[str] = []
    def walk(x: Any) -> None:
        if isinstance(x, str):
            if x.strip(): texts.append(x.strip())
        elif isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
    walk(obj)
    return "\n".join(texts).strip()


def _preview_response(resp: Any, limit: int = 500) -> str:
    try:
        txt = _extract_response_text(resp)
        return txt[:limit] if txt else json.dumps(_to_plain(resp), ensure_ascii=False)[:limit]
    except Exception:
        return repr(resp)[:limit]


def _preview_text_or_response(text: str, resp: Any, limit: int = 500) -> str:
    return text.strip()[:limit] if (text and text.strip()) else _preview_response(resp, limit=limit)
