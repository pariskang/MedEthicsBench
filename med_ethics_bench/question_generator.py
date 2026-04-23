from __future__ import annotations

"""
question_generator.py — robust stage-3 generator.

Key fixes:
1) Generation uses reasoning_effort=None to prevent planning text leakage.
2) JSON parse failures are retried with explicit failure feedback.
3) Final failures are written to question_failures.jsonl for inspection.
4) Compatible with both new and old PoeClient.respond_json signatures.
"""

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .concurrency import run_parallel
from .config import (
    COMPARISON_PAIRS,
    COMPARISON_QTYPE,
    COUNTRY_AXES,
    QUESTION_TYPES,
    SINGLE_COUNTRY_QTYPES,
    TOPIC_AXES,
    UNIVERSAL_QTYPES,
    Config,
)
from .knowledge_base import KnowledgeBase
from .poe_client import PoeClient
from .prompts import (
    QGEN_COMPARISON_TEMPLATE,
    QGEN_SINGLE_COUNTRY_TEMPLATE,
    QGEN_SYSTEM,
    QGEN_UNIVERSAL_TEMPLATE,
)
from .schemas import KeyPoint, Option, Question

log = logging.getLogger(__name__)


def _qid(prefix: str, parts: Sequence[str], idx: int) -> str:
    raw = "|".join([prefix, *parts, str(idx)])
    return f"q_{prefix}_{'_'.join(parts)}_{idx}_{hashlib.md5(raw.encode()).hexdigest()[:6]}"


def _country_map() -> Dict[str, dict]:
    return {c["id"]: c for c in COUNTRY_AXES}


def _qtype_map() -> Dict[str, dict]:
    return {q["id"]: q for q in QUESTION_TYPES}


class QuestionGenerator:
    def __init__(self, client: PoeClient, config: Config, kb: KnowledgeBase):
        self.client = client
        self.config = config
        self.kb = kb
        self._c = _country_map()
        self._qt = _qtype_map()
        os.makedirs(config.output_dir, exist_ok=True)
        self.path = Path(config.output_dir) / "questions.jsonl"
        self.fail_path = Path(config.output_dir) / "question_failures.jsonl"

    def run(self, languages: Optional[List[str]] = None, on_question_saved=None) -> List[Question]:
        languages = languages or ["zh"]
        existing = self._load_existing()
        seen = {q.question_id for q in existing}

        jobs = []
        for country in COUNTRY_AXES:
            for topic in TOPIC_AXES:
                if not self.kb.has_single_country_material(topic["id"], country["id"]):
                    continue
                for qt_id in SINGLE_COUNTRY_QTYPES:
                    for lang in languages:
                        for i in range(self.config.single_country_per_cell):
                            jobs.append(("single", country["id"], topic["id"], qt_id, lang, i))

        for pair in COMPARISON_PAIRS:
            a, b = pair["countries"]
            for topic in TOPIC_AXES:
                if not self.kb.has_comparison_material(topic["id"], a, b):
                    continue
                for lang in languages:
                    for i in range(self.config.comparison_per_cell):
                        jobs.append(("compare", pair["id"], topic["id"], COMPARISON_QTYPE, lang, i))

        for topic in TOPIC_AXES:
            if not self.kb.has_universal_material(topic["id"]):
                continue
            for qt_id in UNIVERSAL_QTYPES:
                for lang in languages:
                    for i in range(self.config.universal_per_cell):
                        jobs.append(("univ", "-", topic["id"], qt_id, lang, i))

        log.info("题目生成任务 = %d（已存在 %d）", len(jobs), len(seen))
        log.info(
            "  ① 单国深度: %d | ② 跨国对比: %d | ③ 普适困境: %d",
            sum(1 for j in jobs if j[0] == "single"),
            sum(1 for j in jobs if j[0] == "compare"),
            sum(1 for j in jobs if j[0] == "univ"),
        )

        pending = []
        for track, key1, topic_id, qt_id, lang, i in jobs:
            qid = _qid(track, [key1, topic_id, qt_id, lang], i)
            if qid not in seen:
                pending.append((qid, track, key1, topic_id, qt_id, lang, i))

        def _worker(job):
            qid, track, key1, topic_id, qt_id, lang, _i = job
            if track == "single":
                return self._generate_single(qid, country_id=key1, topic_id=topic_id, qt_id=qt_id, lang=lang)
            if track == "compare":
                return self._generate_comparison(qid, pair_id=key1, topic_id=topic_id, lang=lang)
            return self._generate_universal(qid, topic_id=topic_id, qt_id=qt_id, lang=lang)

        new_qs: List[Question] = []
        for job, q, err in run_parallel(pending, _worker, max_workers=self.config.max_concurrency, desc="出题"):
            qid = job[0]
            if err is not None:
                log.warning("生成失败 qid=%s：%s", qid, err)
                self._append_failure(qid, str(err), {"job": job})
                continue
            if q is None:
                continue
            new_qs.append(q)
            self._append(q)
            if on_question_saved is not None:
                try:
                    on_question_saved(q)
                except Exception as cb_err:  # noqa: BLE001
                    log.warning("on_question_saved 回调异常（忽略）: %s", cb_err)

        log.info("本次新增 %d 题，累计 %d", len(new_qs), len(existing) + new_qs.__len__())
        return existing + new_qs

    def _generate_single(self, qid: str, *, country_id: str, topic_id: str, qt_id: str, lang: str) -> Optional[Question]:
        country = self._c[country_id]
        topic = next(t for t in TOPIC_AXES if t["id"] == topic_id)
        qt = self._qt[qt_id]
        ct_block, cc_block, gt_block, used_ids = self.kb.render_single_country_blocks(topic_id, country_id)
        prompt = QGEN_SINGLE_COUNTRY_TEMPLATE.format(
            country_zh=country["zh"], country_en=country["en"],
            cultural_notes=country["cultural_notes"],
            ethics_bodies=country["ethics_bodies"],
            topic_zh=topic["zh"],
            qtype_zh=qt["zh_name"], qtype_en=qt["id"], qtype_desc=qt["description"],
            language=lang,
            country_theory_blocks=ct_block,
            country_case_blocks=cc_block,
            global_theory_blocks=gt_block,
        )
        return self._call_and_build(
            qid=qid, prompt=prompt, qt=qt, topic_id=topic_id, lang=lang,
            track="single_country", country_ids=[country_id], comparison_pair_id=None, source_doc_ids=used_ids,
        )

    def _generate_comparison(self, qid: str, *, pair_id: str, topic_id: str, lang: str) -> Optional[Question]:
        pair = next(p for p in COMPARISON_PAIRS if p["id"] == pair_id)
        a_id, b_id = pair["countries"]
        a, b = self._c[a_id], self._c[b_id]
        topic = next(t for t in TOPIC_AXES if t["id"] == topic_id)
        qt = self._qt[COMPARISON_QTYPE]
        blocks, used_ids = self.kb.render_comparison_blocks(topic_id, a_id, b_id)
        prompt = QGEN_COMPARISON_TEMPLATE.format(
            topic_zh=topic["zh"],
            country_a_zh=a["zh"], country_a_en=a["en"],
            country_b_zh=b["zh"], country_b_en=b["en"],
            comparison_focus=pair["focus_zh"],
            language=lang, qtype_zh=qt["zh_name"],
            a_theory_blocks=blocks["a_theory"], a_case_blocks=blocks["a_case"], a_cultural_notes=a["cultural_notes"],
            b_theory_blocks=blocks["b_theory"], b_case_blocks=blocks["b_case"], b_cultural_notes=b["cultural_notes"],
        )
        return self._call_and_build(
            qid=qid, prompt=prompt, qt=qt, topic_id=topic_id, lang=lang,
            track="comparison", country_ids=[a_id, b_id], comparison_pair_id=pair_id, source_doc_ids=used_ids,
        )

    def _generate_universal(self, qid: str, *, topic_id: str, qt_id: str, lang: str) -> Optional[Question]:
        topic = next(t for t in TOPIC_AXES if t["id"] == topic_id)
        qt = self._qt[qt_id]
        gt_block, used_ids = self.kb.render_universal_blocks(topic_id)
        prompt = QGEN_UNIVERSAL_TEMPLATE.format(
            topic_zh=topic["zh"], qtype_zh=qt["zh_name"], qtype_en=qt["id"], qtype_desc=qt["description"],
            language=lang, global_theory_blocks=gt_block,
        )
        return self._call_and_build(
            qid=qid, prompt=prompt, qt=qt, topic_id=topic_id, lang=lang,
            track="universal", country_ids=[], comparison_pair_id=None, source_doc_ids=used_ids,
        )

    def _call_and_build(
        self,
        *,
        qid: str,
        prompt: str,
        qt: dict,
        topic_id: str,
        lang: str,
        track: str,
        country_ids: List[str],
        comparison_pair_id: Optional[str],
        source_doc_ids: List[str],
    ) -> Optional[Question]:
        # Only require the structurally critical fields for initial parse.
        # Short metadata fields (expected_*, common_pitfalls, requires_reasoning_trace)
        # are now generated FIRST in the prompt, so they should always be present.
        # But if they're still missing, we complete them in _complete_missing_fields().
        required_keys = [
            "stem", "reference_answer", "key_points",
        ]
        # Full set of expected fields — used to detect what needs completion
        all_expected_keys = [
            "stem", "context", "options", "reference_answer", "key_points",
            "expected_principles", "expected_stakeholders",
            "common_pitfalls", "requires_reasoning_trace",
        ]
        last_err: Optional[str] = None
        last_raw_data: Optional[dict] = None

        for attempt in range(3):
            attempt_prompt = prompt
            if attempt > 0 and last_err:
                attempt_prompt += (
                    "\n\n【系统纠错要求】\n"
                    f"上一次输出失败原因：{last_err}\n"
                    "请从头重新生成，直接输出一个合法 JSON 对象。"
                    "不要输出任何思考过程、前言、解释、标题、Markdown 围栏。"
                )
            try:
                data = self._respond_json_compat(
                    model=self.config.generator_model,
                    system=QGEN_SYSTEM,
                    user=attempt_prompt,
                    enable_web_search=False,
                    reasoning_effort=None,
                    reasoning_summary="none",
                    max_output_tokens=7000,
                    required_keys=required_keys,
                    repair_on_fail=True,
                )

                # If any short metadata fields are still missing, complete them
                # with a targeted API call instead of failing the entire attempt.
                data = self._complete_missing_fields(data, all_expected_keys, qt, attempt_prompt)
                last_raw_data = data
            except Exception as e:  # noqa: BLE001
                last_err = f"生成/JSON 解析失败：{e}"
                log.warning("qid=%s 第 %d 次生成失败：%s", qid, attempt + 1, e)
                continue

            validated, last_err = self._validate_with_error(data, qt=qt, track=track, country_ids=country_ids)
            if validated is None:
                log.warning("qid=%s 第 %d 次业务校验失败：%s", qid, attempt + 1, last_err)
                continue

            stem, context, options, reference, kps, principles, stakeholders, pitfalls, req_rt = validated
            return Question(
                question_id=qid,
                type=qt["id"],
                topic_id=topic_id,
                language=lang,
                difficulty="expert",
                track=track,
                country_ids=country_ids,
                comparison_pair_id=comparison_pair_id,
                stem=stem,
                context=context,
                options=options,
                reference_answer=reference,
                key_points=kps,
                expected_principles=principles,
                expected_stakeholders=stakeholders,
                common_pitfalls=pitfalls,
                requires_reasoning_trace=req_rt,
                source_doc_ids=source_doc_ids,
                generated_by=self.config.generator_model,
            )

        log.warning("题目校验失败 qid=%s：%s", qid, last_err)
        self._append_failure(qid, last_err or "unknown", {"prompt_preview": prompt[:1000], "raw": last_raw_data})
        return None

    def _complete_missing_fields(
        self, data: dict, all_expected_keys: list, qt: dict, original_prompt: str
    ) -> dict:
        """Fetch missing or too-short fields in one targeted API call instead of failing."""
        completable = {
            "expected_principles", "expected_stakeholders",
            "common_pitfalls", "requires_reasoning_trace",
        }
        missing = [k for k in all_expected_keys if k in completable and k not in data]

        # Also detect a reference_answer that is present but too short to pass validation
        ref = str(data.get("reference_answer") or "").strip()
        ref_too_short = bool(ref) and len(ref) < self.config.min_reference_answer_chars
        if ref_too_short and "reference_answer" not in missing:
            missing.append("reference_answer")

        if not missing:
            return data

        log.info("字段补全：缺少或过短 %s，发起补全请求。", missing)
        stem_preview = str(data.get("stem", ""))[:600]
        ref_preview = ref[:600] if ref else "（尚未生成）"

        complete_system = (
            "你是医学伦理基准题目的字段补全助手。"
            "根据已有题目内容，补全或续写指定的缺失/过短字段。"
            "直接输出一个仅含目标字段的 JSON 对象，不输出其他任何内容。"
        )
        from .poe_client import _missing_keys_example  # noqa: PLC0415
        ref_instruction = ""
        if "reference_answer" in missing:
            if ref_too_short:
                ref_instruction = (
                    f"\n注意：reference_answer 当前只有 {len(ref)} 字，需续写至至少 "
                    f"{self.config.min_reference_answer_chars} 字。"
                    f"已有内容：{ref_preview}\n续写时保持连贯，并补充：规范识别 → 冲突权衡 → 反方处理 → 结论。"
                )
            else:
                ref_instruction = (
                    "\n注意：reference_answer 需要完整体现：事实识别 → 规范识别 → 冲突权衡 → 反方处理 → 结论，"
                    f"至少 {self.config.min_reference_answer_chars} 字。"
                )

        complete_user = (
            f"题型：{qt['id']}，题目摘要：\n"
            f"[题干] {stem_preview}\n\n"
            f"请为以下字段生成/补全内容：{missing}{ref_instruction}\n\n"
            "直接输出只含这些字段的 JSON 对象，示例格式：\n"
            + _missing_keys_example([k for k in missing if k != "reference_answer"])
        )
        try:
            completion = self._respond_json_compat(
                model=self.config.generator_model,
                system=complete_system,
                user=complete_user,
                enable_web_search=False,
                reasoning_effort=None,
                reasoning_summary="none",
                max_output_tokens=2000,
                required_keys=[],
                repair_on_fail=True,
            )
            for k in missing:
                if k in completion and completion[k] is not None:
                    # For reference_answer, append to existing partial content
                    if k == "reference_answer" and ref_too_short and ref:
                        existing = data.get("reference_answer", "")
                        extension = str(completion[k]).strip()
                        # Avoid duplicating content
                        if not extension.startswith(existing[:50]):
                            data[k] = existing + "\n" + extension
                        else:
                            data[k] = extension
                    else:
                        data[k] = completion[k]
        except Exception as e:  # noqa: BLE001
            log.warning("字段补全失败（将使用默认值）：%s", e)

        # Safe defaults for anything still missing
        data.setdefault("expected_principles", [])
        data.setdefault("expected_stakeholders", [])
        data.setdefault("common_pitfalls", [])
        data.setdefault("requires_reasoning_trace", True)
        data.setdefault("context", None)
        data.setdefault("options", None)
        return data

    def _respond_json_compat(self, **kwargs):
        sig = inspect.signature(self.client.respond_json)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.client.respond_json(**supported)

    def _validate_with_error(self, data: dict, *, qt: dict, track: str, country_ids: List[str]) -> Tuple[Optional[Tuple], Optional[str]]:
        for k in ("stem", "reference_answer", "key_points"):
            if k not in data or data.get(k) in (None, "", []):
                return None, f"缺字段 {k}"

        stem = str(data["stem"]).strip()
        reference = str(data["reference_answer"]).strip()
        context = data.get("context")
        if context in ("", "null"):
            context = None

        if track == "single_country" and len(stem) < self.config.min_stem_chars_single:
            return None, f"单国题 stem 过短：{len(stem)}"
        if track == "comparison" and len(stem) < self.config.min_stem_chars_comparison:
            return None, f"对比题 stem 过短：{len(stem)}"
        if track == "universal" and len(stem) < self.config.min_stem_chars_universal:
            return None, f"普适题 stem 过短：{len(stem)}"
        if len(reference) < self.config.min_reference_answer_chars:
            return None, f"reference_answer 过短：{len(reference)}"
        if self.config.require_context_for_country_items and track in {"single_country", "comparison"} and not context:
            return None, "country/comparison 题缺 context"

        options = None
        if qt["id"] in ("single_choice", "multi_choice"):
            opt_list = data.get("options") or []
            if len(opt_list) != 4:
                return None, f"{qt['id']}：选项数 {len(opt_list)} ≠ 4"
            correct = sum(1 for o in opt_list if o.get("is_correct"))
            if qt["id"] == "single_choice" and correct != 1:
                return None, f"单选正确项 = {correct}"
            if qt["id"] == "multi_choice" and correct < 2:
                return None, f"多选正确项 = {correct} < 2"
            options = [
                Option(
                    label=str(o.get("label") or chr(65 + i)).strip(),
                    text=str(o.get("text", "")).strip(),
                    is_correct=bool(o.get("is_correct", False)),
                    rationale=o.get("rationale"),
                )
                for i, o in enumerate(opt_list)
            ]

        raw_kps = data.get("key_points") or []
        kps: List[KeyPoint] = []
        categories = set()
        for i, kp in enumerate(raw_kps):
            content = str(kp.get("content", "")).strip()
            if not content:
                continue
            cat = str(kp.get("category", "content"))
            categories.add(cat)
            kps.append(
                KeyPoint(
                    id=kp.get("id") or f"kp{i+1}",
                    content=content,
                    weight=float(kp.get("weight", 1.0)),
                    required=bool(kp.get("required", False)),
                    category=cat,
                )
            )

        if len(kps) < self.config.min_key_points:
            return None, f"key_points < {self.config.min_key_points}（实际 {len(kps)}）"
        if sum(1 for k in kps if k.required) < self.config.min_required_key_points:
            return None, "required key points 不足"

        required_categories = {"principle", "stakeholder", "citation"}
        if self.config.require_counterargument_category:
            required_categories.add("counterargument")
        if track in {"single_country", "comparison"}:
            required_categories.add("cultural_context")
        if not required_categories.issubset(categories):
            return None, f"缺少关键 categories: {sorted(required_categories - categories)}"

        principles = [str(x).strip() for x in (data.get("expected_principles") or []) if str(x).strip()]
        stakeholders = [str(x).strip() for x in (data.get("expected_stakeholders") or []) if str(x).strip()]
        pitfalls = [str(x).strip() for x in (data.get("common_pitfalls") or []) if str(x).strip()]
        req_rt = bool(data.get("requires_reasoning_trace", qt.get("need_reasoning", True)))

        if len(principles) < 2:
            return None, "expected_principles < 2"
        if len(stakeholders) < 2 and qt["id"] in {"case_analysis", "stakeholder_map", "cross_cultural", "ethical_dilemma"}:
            return None, "expected_stakeholders 过少"
        if len(pitfalls) < self.config.min_common_pitfalls:
            return None, f"common_pitfalls < {self.config.min_common_pitfalls}"

        if track in {"single_country", "comparison"}:
            if not self._has_jurisdictional_pitfall(pitfalls):
                return None, "缺少法域混淆类 pitfall"
            if self.config.require_country_anchor_mentions and not self._is_country_anchored(stem, reference, country_ids):
                return None, "country/comparison 题未通过法域锚点校验"

        return (stem, context, options, reference, kps, principles, stakeholders, pitfalls, req_rt), None

    def _has_jurisdictional_pitfall(self, pitfalls: List[str]) -> bool:
        text = " ".join(pitfalls).lower()
        keys = [
            "另一国", "他国", "法域", "jurisdiction", "country", "framework",
            "美国", "中国", "英国", "德国", "法国", "俄罗斯",
            "us", "uk", "de", "fr", "ru", "cn",
        ]
        return any(k in text for k in keys)

    def _is_country_anchored(self, stem: str, reference: str, country_ids: List[str]) -> bool:
        haystack = f"{stem}\n{reference}".lower()
        for cid in country_ids:
            c = self._c[cid]
            tokens = {c["zh"].lower(), c["en"].lower(), cid.lower()}
            bodies = c.get("ethics_bodies", "")
            for sep in ["；", ";", "，", ",", "、"]:
                bodies = bodies.replace(sep, "|")
            tokens.update({x.strip().lower() for x in bodies.split("|") if x.strip()})
            if not any(tok and tok in haystack for tok in tokens):
                return False
        return True

    def _append(self, q: Question) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(q.model_dump_json() + "\n")

    def _append_failure(self, qid: str, reason: str, payload: dict) -> None:
        rec = {"qid": qid, "reason": reason, **payload}
        with open(self.fail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_existing_for_seed(self) -> List[Question]:
        """Read questions.jsonl without modifying internal state (used by pipeline for seeding)."""
        if not self.path.exists():
            return []
        out: List[Question] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(Question.model_validate_json(line))
                    except Exception as e:  # noqa: BLE001
                        log.warning("questions.jsonl 预读跳过损坏行：%s", e)
        return out

    def _load_existing(self) -> List[Question]:
        if not (self.config.resume and self.path.exists()):
            if self.path.exists():
                self.path.unlink()
            return []
        out: List[Question] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(Question.model_validate_json(line))
                    except Exception as e:  # noqa: BLE001
                        log.warning("跳过损坏题目：%s", e)
        return out
