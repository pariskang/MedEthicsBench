"""
pipeline.py — 顶层 Pipeline

把四个模块串起来，提供 4 个能力：
1) build_benchmark()          —— Stage 1–4 全量跑；
2) generate_questions_only()  —— 只从 Stage 3 开始，读取已有 documents.jsonl；
3) run_model(model)           —— 用指定模型作答题集；
4) grade(model)               —— 对某模型的作答集进行评估。
"""
from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from tqdm import tqdm

from .config import Config, TOPIC_AXES, QUESTION_TYPES
from .evaluator import Evaluator
from .knowledge_base import KnowledgeBase
from .poe_client import PoeClient
from .question_generator import QuestionGenerator
from .retriever import Retriever
from .schemas import (
    BenchmarkItem, BenchmarkMeta, Document, Evaluation, ModelAnswer, Question,
)

log = logging.getLogger(__name__)


class BenchmarkPipeline:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.client = PoeClient(
            api_key=config._effective_api_key(),
            base_url=config._effective_base_url(),
            backend=config.backend,
            request_timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
        os.makedirs(config.output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        lvl = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    # ==================================================================
    # Stage 1–2：检索 + 出题
    # ==================================================================
    def build_benchmark(self, languages: Optional[List[str]] = None) -> Path:
        est = self.config.estimate_total_questions()
        log.info("预计题量：单国 %d + 对比 %d + 普适 %d = 总 %d",
                 est["single_country"], est["comparison"], est["universal"], est["total"])

        log.info(">>> Stage 1: 检索理论文献与真实案例（按国别）")
        retriever = Retriever(self.client, self.config)
        docs: List[Document] = retriever.run()

        log.info(">>> Stage 2: 构建知识库（三维索引）")
        kb = KnowledgeBase(docs)

        log.info(">>> Stage 3: 生成题目（三分支）")
        qgen = QuestionGenerator(self.client, self.config, kb)

        # ── 实时缓存 ─────────────────────────────────────────────────────────
        out_dir = Path(self.config.output_dir)
        benchmark_path = out_dir / "benchmark.jsonl"
        cache_lock = __import__("threading").Lock()

        # 用已有 questions.jsonl 内容预填 benchmark.jsonl（真实数据源始终是 questions.jsonl）
        existing_questions = qgen._load_existing_for_seed()
        with open(benchmark_path, "w", encoding="utf-8") as _f:
            for _q in existing_questions:
                _f.write(BenchmarkItem(question=_q).model_dump_json() + "\n")

        def _cache_question(q: Question) -> None:
            item = BenchmarkItem(question=q)
            with cache_lock:
                with open(benchmark_path, "a", encoding="utf-8") as f:
                    f.write(item.model_dump_json() + "\n")

        questions: List[Question] = qgen.run(languages=languages, on_question_saved=_cache_question)

        log.info(">>> Stage 4: 打包 benchmark")
        out = self._package(questions)
        log.info("✅ Benchmark 生成完成：%s", out)
        return out


    def generate_questions_only(self, languages: Optional[List[str]] = None) -> Path:
        """只从 Stage 3 开始：读取已存在的 documents.jsonl，重建内存知识库，继续出题并打包。

        断点续做逻辑（config.resume=True，默认）：
        - questions.jsonl 是唯一的真实数据源（source of truth）
        - 每次启动时 benchmark.jsonl 先用 questions.jsonl 中的已有题目重写，
          再实时追加新生成的题目，确保任何时刻崩溃都能从 questions.jsonl 恢复。
        """
        docs = self._load_documents()
        if not docs:
            raise FileNotFoundError(
                f"未找到可用文档：{Path(self.config.output_dir) / 'documents.jsonl'}。请先完成 Stage 1 检索。"
            )

        log.info(">>> Stage 3-only: 读取已有 documents.jsonl，共 %d 篇", len(docs))
        kb = KnowledgeBase(docs)

        log.info(">>> Stage 3: 生成题目（三分支）")
        qgen = QuestionGenerator(self.client, self.config, kb)

        # ── 实时缓存设置 ──────────────────────────────────────────────────────
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = out_dir / "benchmark.jsonl"
        cache_lock = __import__("threading").Lock()

        # 始终重写 benchmark.jsonl，使其与 questions.jsonl 保持一致。
        # questions.jsonl 是真实数据源；benchmark.jsonl 是随时可以从前者重建的派生文件。
        # 先把已有题目写入，确保中途崩溃后 benchmark.jsonl 也是完整可用的。
        existing_questions = qgen._load_existing_for_seed()
        if existing_questions:
            log.info("断点续做：从 questions.jsonl 恢复 %d 题到 benchmark.jsonl", len(existing_questions))
        with open(benchmark_path, "w", encoding="utf-8") as _f:
            for _q in existing_questions:
                _f.write(BenchmarkItem(question=_q).model_dump_json() + "\n")

        def _cache_question(q: Question) -> None:
            """每生成一题，立即追加到 benchmark.jsonl（线程安全）。"""
            item = BenchmarkItem(question=q)
            with cache_lock:
                with open(benchmark_path, "a", encoding="utf-8") as f:
                    f.write(item.model_dump_json() + "\n")

        questions: List[Question] = qgen.run(languages=languages, on_question_saved=_cache_question)

        log.info(">>> Stage 4: 打包 benchmark（共 %d 题）", len(questions))
        out = self._package(questions)
        log.info("✅ Stage 3-only 完成：%s（共 %d 题）", out, len(questions))
        return out

    def repack_from_questions(self) -> Path:
        """从已有的 questions.jsonl 直接重建 benchmark.jsonl，不调用任何 LLM。

        适用场景：
        - 生成中途崩溃，questions.jsonl 有备份，想先获得当前已完成题目的 benchmark；
        - 修改了打包/统计逻辑，需要用原始题目数据重新打包；
        - 验证 questions.jsonl 数据完整性。
        """
        q_path = Path(self.config.output_dir) / "questions.jsonl"
        if not q_path.exists():
            raise FileNotFoundError(
                f"未找到 questions.jsonl：{q_path}\n"
                "请确保已运行过 generate/build 命令并产生了题目数据。"
            )

        questions: List[Question] = []
        skipped = 0
        with open(q_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    questions.append(Question.model_validate_json(line))
                except Exception as e:  # noqa: BLE001
                    log.warning("questions.jsonl 第 %d 行解析失败（跳过）：%s", lineno, e)
                    skipped += 1

        if not questions:
            raise ValueError(f"questions.jsonl 中没有可解析的题目（跳过 {skipped} 行）。")

        log.info(
            "repack：从 questions.jsonl 读取 %d 题（跳过损坏行 %d 条），重建 benchmark.jsonl",
            len(questions), skipped,
        )
        out = self._package(questions)
        log.info("✅ repack 完成：%s（共 %d 题）", out, len(questions))
        return out

    def _package(self, questions: List[Question]) -> Path:
        meta = BenchmarkMeta(
            n_items=len(questions),
            topic_ids=sorted({q.topic_id for q in questions}),
            question_types=sorted({q.type for q in questions}),
            generator_model=self.config.generator_model,
            retrieval_model=self.config.retrieval_model,
        )
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 元信息
        (out_dir / "benchmark_meta.json").write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )

        # 主数据集
        benchmark_path = out_dir / "benchmark.jsonl"
        with open(benchmark_path, "w", encoding="utf-8") as f:
            for q in questions:
                item = BenchmarkItem(question=q)
                f.write(item.model_dump_json() + "\n")

        # 人类可读统计
        stats = _compute_stats(questions)
        (out_dir / "benchmark_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return benchmark_path

    # ==================================================================
    # Stage 3：用被测模型作答
    # ==================================================================
    def run_model(self, model_name: str, *, limit: Optional[int] = None) -> Path:
        questions = self._load_questions()
        if limit:
            questions = questions[:limit]

        out_path = Path(self.config.output_dir) / f"answers_{_safe(model_name)}.jsonl"
        done_ids = _done_question_ids(out_path) if self.config.resume else set()
        if not self.config.resume and out_path.exists():
            out_path.unlink()

        log.info("使用 %s 作答 %d 道题（跳过已完成 %d 道）",
                 model_name, len(questions), len(done_ids))

        for q in tqdm(questions, desc=f"作答[{model_name}]"):
            if q.question_id in done_ids:
                continue
            try:
                ans = self._answer_one(model_name, q)
            except Exception as e:  # noqa: BLE001
                log.warning("作答失败 qid=%s：%s", q.question_id, e)
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(ans.model_dump_json() + "\n")
        log.info("✅ 作答完成 → %s", out_path)
        return out_path

    def _answer_one(self, model_name: str, q: Question) -> ModelAnswer:
        sys_prompt = (
            "You are taking a challenging medical ethics & psychology exam. "
            "Think step by step about the competing ethical principles, identify "
            "stakeholders, reason carefully, then give a concise final answer. "
            "If the question is multiple choice, end your answer with a line "
            "'最终选择: X' (or X,Y for multi-choice)."
        )
        user_prompt = _render_question_for_subject(q)

        res = self.client.respond(
            model=model_name,
            system=sys_prompt,
            user=user_prompt,
            enable_web_search=False,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_summary="detailed",
            max_output_tokens=2000,
        )
        final_text = res.text
        reasoning = res.reasoning_summary
        picked: Optional[List[str]] = None
        if q.options:
            picked = _parse_selected_labels(final_text, valid_labels=[o.label for o in q.options])

        return ModelAnswer(
            question_id=q.question_id,
            model_name=model_name,
            reasoning_trace=reasoning,
            final_answer=final_text,
            selected_labels=picked,
        )

    # ==================================================================
    # Stage 4：评分 & 报表
    # ==================================================================
    def grade(self, model_name: str) -> Dict:
        questions = {q.question_id: q for q in self._load_questions()}
        ans_path = Path(self.config.output_dir) / f"answers_{_safe(model_name)}.jsonl"
        if not ans_path.exists():
            raise FileNotFoundError(
                f"未找到作答文件 {ans_path}，请先 run_model('{model_name}')。"
            )

        eval_path = Path(self.config.output_dir) / f"evaluations_{_safe(model_name)}.jsonl"
        done_ids = _done_question_ids(eval_path) if self.config.resume else set()
        if not self.config.resume and eval_path.exists():
            eval_path.unlink()

        evaluator = Evaluator(self.client, self.config)
        evals: List[Evaluation] = []

        # Load already-completed evaluations so the final report includes them
        if self.config.resume and eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            evals.append(Evaluation.model_validate_json(line))
                        except Exception:  # noqa: BLE001
                            pass

        with open(ans_path, "r", encoding="utf-8") as f:
            answers = [ModelAnswer.model_validate_json(l) for l in f if l.strip()]

        # Report coverage before starting
        missing = [a.question_id for a in answers if a.question_id not in questions]
        if missing:
            log.warning(
                "答案文件中有 %d 道题在题库中找不到（question_id 不匹配），将跳过。\n"
                "  首个缺失 ID：%s\n"
                "  提示：运行 'python main.py repack' 可从 questions.jsonl 重建题库。",
                len(missing), missing[0],
            )
        matched = len(answers) - len(missing)
        log.info("答案匹配：%d/%d 道题成功匹配（跳过 %d 道）",
                 matched, len(answers), len(missing))

        for ans in tqdm(answers, desc=f"评分[{model_name}]"):
            if ans.question_id in done_ids:
                continue
            q = questions.get(ans.question_id)
            if q is None:
                continue   # already warned above
            try:
                ev = evaluator.evaluate(q, ans)
            except Exception as e:  # noqa: BLE001
                log.warning("评分失败 qid=%s：%s", ans.question_id, e)
                continue
            evals.append(ev)
            with open(eval_path, "a", encoding="utf-8") as f:
                f.write(ev.model_dump_json() + "\n")

        # 汇总
        report = _compute_report(evals, questions)
        report_path = Path(self.config.output_dir) / f"report_{_safe(model_name)}.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("✅ 评分完成 → %s（共评 %d 题）", report_path, len(evals))
        return report

    # ==================================================================
    # 辅助
    # ==================================================================
    def _load_documents(self) -> List[Document]:
        p = Path(self.config.output_dir) / "documents.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"未找到文档集 {p}，请先完成检索阶段。")
        docs: List[Document] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(Document.model_validate_json(line))
                except Exception as e:  # noqa: BLE001
                    log.warning("跳过损坏文档：%s", e)
        return docs

    def _load_questions(self) -> List[Question]:
        """Load questions, merging from all available sources.

        Priority: questions.jsonl (source of truth) > benchmark.jsonl (derived).
        This ensures grade() can always find questions even when benchmark.jsonl
        was partially rebuilt or regenerated from a different run.
        """
        out_dir = Path(self.config.output_dir)
        questions: Dict[str, Question] = {}

        # 1. Load from benchmark.jsonl (wrapped BenchmarkItem format)
        bench_path = out_dir / "benchmark.jsonl"
        if bench_path.exists():
            with open(bench_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = BenchmarkItem.model_validate_json(line)
                        questions[item.question.question_id] = item.question
                    except Exception as e:  # noqa: BLE001
                        log.debug("benchmark.jsonl 第 %d 行跳过：%s", lineno, e)

        # 2. Load from questions.jsonl (raw Question format) — overrides benchmark.jsonl
        q_path = out_dir / "questions.jsonl"
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = Question.model_validate_json(line)
                        questions[q.question_id] = q
                    except Exception as e:  # noqa: BLE001
                        log.debug("questions.jsonl 第 %d 行跳过：%s", lineno, e)

        if not questions:
            raise FileNotFoundError(
                f"在 {out_dir} 中未找到任何题目。\n"
                "请确保存在 questions.jsonl 或 benchmark.jsonl。\n"
                "可运行 'python main.py repack' 重建 benchmark.jsonl。"
            )

        total = len(questions)
        from_bench = bench_path.exists()
        from_q = q_path.exists()
        log.info(
            "加载题库：共 %d 题（来源：%s）",
            total,
            " + ".join(filter(None, [
                f"questions.jsonl({sum(1 for _ in open(q_path) if _.strip()) if from_q else 0})" if from_q else None,
                f"benchmark.jsonl" if from_bench else None,
            ])),
        )
        return list(questions.values())


# ======================================================================
# 工具函数
# ======================================================================
def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _done_question_ids(path: Path):
    ids = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line).get("question_id"))
            except Exception:  # noqa: BLE001
                continue
    return ids


def _render_question_for_subject(q: Question) -> str:
    """把题目渲染成给被测模型看的形式（不泄露 rubric、参考答案、key_points）。"""
    parts = []
    if q.context:
        parts.append(f"【背景材料】\n{q.context}\n")
    parts.append(f"【题干】\n{q.stem}\n")
    if q.options:
        parts.append("【选项】")
        for o in q.options:
            parts.append(f"{o.label}. {o.text}")
        parts.append("")
    if q.requires_reasoning_trace:
        parts.append("请先分步说明你的推理过程，再给出最终答案。")
    return "\n".join(parts)


def _parse_selected_labels(text: str, valid_labels: List[str]):
    """从 '最终选择: A,C' 这样的文本里抽出标签。找不到就回 None。"""
    import re
    if not text:
        return None
    # 中英文冒号都支持
    m = re.search(r"最终选择[:：\s]*([A-Ea-e,，\s]+)", text)
    if not m:
        m = re.search(r"final\s*answer[:\s]*([A-Ea-e,\s]+)", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).upper().replace("，", ",")
    labels = [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()]
    labels = [x for x in labels if x in valid_labels]
    return labels or None


def _compute_stats(questions: List[Question]) -> Dict:
    by_topic: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    by_track: Dict[str, int] = defaultdict(int)
    by_country: Dict[str, int] = defaultdict(int)
    for q in questions:
        by_topic[q.topic_id] += 1
        by_type[q.type] += 1
        by_track[q.track] += 1
        for cid in q.country_ids:
            by_country[cid] += 1
    return {
        "total_questions": len(questions),
        "by_track":   dict(by_track),
        "by_country": dict(by_country),
        "by_topic":   dict(by_topic),
        "by_type":    dict(by_type),
        "topics_configured": len(TOPIC_AXES),
        "types_configured":  len(QUESTION_TYPES),
    }


def _compute_report(evals: List[Evaluation], questions: Dict[str, Question]) -> Dict:
    if not evals:
        return {"n": 0, "overall_score": 0.0}
    by_type: Dict[str, List[float]]    = defaultdict(list)
    by_topic: Dict[str, List[float]]   = defaultdict(list)
    by_track: Dict[str, List[float]]   = defaultdict(list)
    by_country: Dict[str, List[float]] = defaultdict(list)
    by_dim: Dict[str, List[float]]     = defaultdict(list)
    pass_rate = 0
    for ev in evals:
        q = questions.get(ev.question_id)
        if q:
            by_type[q.type].append(ev.final_score)
            by_topic[q.topic_id].append(ev.final_score)
            by_track[q.track].append(ev.final_score)
            for cid in q.country_ids:
                by_country[cid].append(ev.final_score)
        for rs in ev.rubric_scores:
            by_dim[rs.dimension].append(rs.score)
        if ev.passed:
            pass_rate += 1
    return {
        "n": len(evals),
        "overall_score": round(mean(ev.final_score for ev in evals), 4),
        "pass_rate": round(pass_rate / len(evals), 4),
        "by_track":   {k: round(mean(v), 4) for k, v in by_track.items()},
        "by_country": {k: round(mean(v), 4) for k, v in by_country.items()},
        "by_type":    {k: round(mean(v), 4) for k, v in by_type.items()},
        "by_topic":   {k: round(mean(v), 4) for k, v in by_topic.items()},
        "by_dimension": {k: round(mean(v), 4) for k, v in by_dim.items()},
    }
