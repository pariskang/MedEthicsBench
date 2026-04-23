"""
evaluator.py — frontier-hard 评估

思路：
1) 选择题：答案集合客观判分 + judge 评价其推理与其余维度；
2) 开放题：judge 全量打分；
3) 对高难 benchmark 增加后处理硬门槛：
   - required key points 命中率不足时封顶；
   - 单国/对比题若 citation / cultural / jurisdictional 过低则封顶；
   - 论证类题若 counterargument handling 太差则封顶。
"""
from __future__ import annotations
import logging
from typing import List, Sequence, Set

from .config import Config
from .poe_client import PoeClient
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from .schemas import Evaluation, KeyPointHit, ModelAnswer, Question, RubricScore

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, client: PoeClient, config: Config):
        self.client = client
        self.config = config

    def evaluate(self, question: Question, answer: ModelAnswer) -> Evaluation:
        if question.type in ("single_choice", "multi_choice"):
            return self._evaluate_choice(question, answer)
        return self._evaluate_open(question, answer)

    # ------------------------------------------------------------------
    # 选择题评估
    # ------------------------------------------------------------------
    def _evaluate_choice(self, q: Question, ans: ModelAnswer) -> Evaluation:
        correct_set: Set[str] = {o.label for o in (q.options or []) if o.is_correct}
        picked_set: Set[str] = set(ans.selected_labels or [])

        tp = len(correct_set & picked_set)
        precision = tp / len(picked_set) if picked_set else 0.0
        recall = tp / len(correct_set) if correct_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        final_correct_score = f1

        judge_eval = self._call_judge(q, ans)

        rubrics: List[RubricScore] = []
        for rs in judge_eval.rubric_scores:
            if rs.dimension == "final_answer_correctness":
                rs = RubricScore(
                    dimension=rs.dimension,
                    score=final_correct_score,
                    reason=(
                        f"客观判分：选中 {sorted(picked_set)}，正确 {sorted(correct_set)}，"
                        f"F1={final_correct_score:.2f}"
                    ),
                )
            rubrics.append(rs)

        if not any(r.dimension == "final_answer_correctness" for r in rubrics):
            rubrics.append(RubricScore(
                dimension="final_answer_correctness",
                score=final_correct_score,
                reason=f"客观判分 F1={final_correct_score:.2f}",
            ))

        final_score = self._aggregate(rubrics)
        final_score = self._apply_hard_caps(q, ans, judge_eval.key_point_hits, rubrics, final_score)
        return Evaluation(
            question_id=q.question_id,
            model_name=ans.model_name,
            judge_model=self.config.judge_model,
            key_point_hits=judge_eval.key_point_hits,
            rubric_scores=rubrics,
            final_score=final_score,
            passed=final_score >= self.config.choice_pass_threshold,
            overall_feedback=judge_eval.overall_feedback,
        )

    # ------------------------------------------------------------------
    # 开放题评估
    # ------------------------------------------------------------------
    def _evaluate_open(self, q: Question, ans: ModelAnswer) -> Evaluation:
        ev = self._call_judge(q, ans)
        final_score = self._aggregate(ev.rubric_scores)
        final_score = self._apply_hard_caps(q, ans, ev.key_point_hits, ev.rubric_scores, final_score)
        return Evaluation(
            question_id=q.question_id,
            model_name=ans.model_name,
            judge_model=self.config.judge_model,
            key_point_hits=ev.key_point_hits,
            rubric_scores=ev.rubric_scores,
            final_score=final_score,
            passed=final_score >= self.config.open_pass_threshold,
            overall_feedback=ev.overall_feedback,
        )

    # ------------------------------------------------------------------
    # 调 LLM-judge
    # ------------------------------------------------------------------
    def _call_judge(self, q: Question, ans: ModelAnswer) -> Evaluation:
        kp_text = "\n".join(
            f"- [{kp.id}] (weight={kp.weight}, required={kp.required}, category={kp.category}) {kp.content}"
            for kp in q.key_points
        )
        rubric_text = "\n".join(f"- {k}: 权重 {v}" for k, v in self.config.rubric_weights.items())
        selected_labels_block = f"[选中的选项标签] {ans.selected_labels}" if ans.selected_labels is not None else ""
        context_block = f"[题目背景材料]\n{q.context}" if q.context else ""

        # ── Input truncation: cap long fields to keep input tokens manageable ──
        # Target: keep total input under ~6000 tokens (≈12000 chars for Chinese).
        # Priority: keep key_points + rubric full; trim content fields.
        MAX_STEM      = 1200
        MAX_CONTEXT   = 800
        MAX_REF       = 1500
        MAX_ANSWER    = 2000
        MAX_REASONING = 1000

        stem_t      = _truncate(q.stem,                MAX_STEM)
        context_t   = _truncate(q.context or "",       MAX_CONTEXT)
        ref_t       = _truncate(q.reference_answer,    MAX_REF)
        answer_t    = _truncate(ans.final_answer,      MAX_ANSWER)
        reasoning_t = _truncate(ans.reasoning_trace or "（未提供）", MAX_REASONING)
        context_block_t = f"[题目背景材料]\n{context_t}" if context_t else ""

        user_prompt = JUDGE_USER_TEMPLATE.format(
            qtype=q.type,
            track=q.track,
            country_ids=", ".join(q.country_ids) if q.country_ids else "（无 / 普适题）",
            stem=stem_t,
            context_block=context_block_t,
            reference_answer=ref_t,
            key_points_block=kp_text,
            rubric_dimensions_block=rubric_text,
            reasoning_trace=reasoning_t,
            final_answer=answer_t,
            selected_labels_block=selected_labels_block,
        )

        # Judge output: 8 rubric dimensions × ~200 chars + 5-7 key_points × ~150 chars
        # + overall_feedback ~400 chars ≈ 3000-4000 chars ≈ 1500-2000 tokens.
        # Reasoning models also consume internal reasoning tokens.
        # Use 5000 to be safe for gpt-5.x / o-series acting as judge.
        JUDGE_OUTPUT_TOKENS = 5000

        data = self.client.respond_json(
            model=self.config.judge_model,
            system=JUDGE_SYSTEM,
            user=user_prompt,
            enable_web_search=False,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_summary="concise",
            max_output_tokens=JUDGE_OUTPUT_TOKENS,
        )

        hits_raw = data.get("key_point_hits") or []
        hits = [
            KeyPointHit(
                key_point_id=h.get("key_point_id", ""),
                hit=bool(h.get("hit", False)),
                evidence=h.get("evidence"),
                partial_credit=float(h.get("partial_credit", 1.0 if h.get("hit") else 0.0)),
            )
            for h in hits_raw
            if h.get("key_point_id")
        ]

        rubrics_raw = data.get("rubric_scores") or []
        rubrics = [
            RubricScore(
                dimension=r.get("dimension", ""),
                score=max(0.0, min(1.0, float(r.get("score", 0.0)))),
                reason=r.get("reason", ""),
            )
            for r in rubrics_raw
            if r.get("dimension") in self.config.rubric_weights
        ]

        seen = {r.dimension for r in rubrics}
        for dim in self.config.rubric_weights:
            if dim not in seen:
                rubrics.append(RubricScore(
                    dimension=dim,
                    score=0.0,
                    reason="judge 未返回该维度，计 0 分。",
                ))

        return Evaluation(
            question_id=q.question_id,
            model_name=ans.model_name,
            judge_model=self.config.judge_model,
            key_point_hits=hits,
            rubric_scores=rubrics,
            final_score=0.0,
            passed=False,
            overall_feedback=str(data.get("overall_feedback", "")).strip(),
        )

    # ------------------------------------------------------------------
    # 聚合 + hard caps
    # ------------------------------------------------------------------
    def _aggregate(self, rubrics: Sequence[RubricScore]) -> float:
        weights = self.config.rubric_weights
        score = 0.0
        for r in rubrics:
            score += weights.get(r.dimension, 0.0) * r.score
        return round(score, 4)

    def _apply_hard_caps(
        self,
        q: Question,
        ans: ModelAnswer,
        hits: Sequence[KeyPointHit],
        rubrics: Sequence[RubricScore],
        score: float,
    ) -> float:
        by_dim = {r.dimension: r.score for r in rubrics}
        hit_map = {h.key_point_id: h for h in hits}

        # 过短答案直接重罚
        if len((ans.final_answer or "").strip()) < 40:
            return round(min(score, 0.30), 4)

        # required key points 命中率不足，封顶
        required = [kp for kp in q.key_points if kp.required]
        if required:
            req_hits = 0
            for kp in required:
                h = hit_map.get(kp.id)
                if h and (h.hit or h.partial_credit >= 0.5):
                    req_hits += 1
            req_rate = req_hits / len(required)
            if req_rate < self.config.required_keypoint_hit_floor:
                score = min(score, 0.64)

        # 单国/对比题必须过法域精度门槛
        if q.track in {"single_country", "comparison"}:
            if by_dim.get("citation_accuracy", 0.0) < self.config.citation_floor_for_country_items:
                score = min(score, 0.66)
            if by_dim.get("cultural_contextualization", 0.0) < self.config.cultural_floor_for_country_items:
                score = min(score, 0.66)
            if by_dim.get("jurisdictional_precision", 0.0) < self.config.jurisdictional_floor_for_country_items:
                score = min(score, 0.62)

        # 论证型题若完全不处理反方，也不应高分通过
        argument_types = {"ethical_dilemma", "open_argument", "cross_cultural", "counterfactual", "principle_conflict", "case_analysis"}
        if q.type in argument_types and by_dim.get("counterargument_handling", 0.0) < self.config.counterargument_floor_for_argument_items:
            score = min(score, 0.69)

        return round(score, 4)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending an ellipsis marker if cut."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[截断]"
