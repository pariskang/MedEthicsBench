"""
knowledge_base.py — v2：三维索引（topic × country × doc_type）

核心查询：
- global_theories(topic_id)          —— 国际性理论/顶刊文献
- country_theories(topic_id, country_id) —— 某国本土指南/法律
- country_cases(topic_id, country_id)    —— 某国本土真实案例
- has_single_country_material(topic_id, country_id)
- has_comparison_material(topic_id, country_a, country_b)
- has_universal_material(topic_id)

为三类出题场景分别提供渲染方法：
- render_single_country_blocks()   → 本国理论 + 本国案例 + 国际理论参照
- render_comparison_blocks()       → A/B 两国各自的材料
- render_universal_blocks()        → 仅国际理论
"""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .schemas import Document, DocType

log = logging.getLogger(__name__)

_MAX_SUMMARY_CHARS = 500
_MAX_DOCS_PER_SLOT = 3


class KnowledgeBase:
    def __init__(self, docs: List[Document]):
        self.docs: List[Document] = docs

        # 索引 1：全球理论（country_id == None） topic → [Document]
        self._global_theory: Dict[str, List[Document]] = defaultdict(list)
        # 索引 2：国别理论/指南   (topic, country) → [Document]
        self._country_theory: Dict[Tuple[str, str], List[Document]] = defaultdict(list)
        # 索引 3：国别案例       (topic, country) → [Document]
        self._country_case: Dict[Tuple[str, str], List[Document]] = defaultdict(list)

        for d in docs:
            for t in d.topic_ids:
                if d.doc_type == DocType.CASE and d.country_id:
                    self._country_case[(t, d.country_id)].append(d)
                elif d.doc_type in (DocType.GUIDELINE, DocType.STATUTE) and d.country_id:
                    self._country_theory[(t, d.country_id)].append(d)
                elif d.doc_type == DocType.THEORY and d.country_id:
                    # 带国别标签的理论也归入国别理论
                    self._country_theory[(t, d.country_id)].append(d)
                else:
                    # 全球理论：无 country_id
                    self._global_theory[t].append(d)

        log.info("知识库：全球理论 %d 主题，国别理论 %d (主题,国家)，国别案例 %d (主题,国家)",
                 len(self._global_theory), len(self._country_theory), len(self._country_case))

    # ==================================================================
    # 查询
    # ==================================================================
    def global_theories(self, topic_id: str) -> List[Document]:
        return self._global_theory.get(topic_id, [])[:_MAX_DOCS_PER_SLOT]

    def country_theories(self, topic_id: str, country_id: str) -> List[Document]:
        return self._country_theory.get((topic_id, country_id), [])[:_MAX_DOCS_PER_SLOT]

    def country_cases(self, topic_id: str, country_id: str) -> List[Document]:
        return self._country_case.get((topic_id, country_id), [])[:_MAX_DOCS_PER_SLOT]

    # ---- 是否有足够材料出题 ----
    def has_single_country_material(self, topic_id: str, country_id: str) -> bool:
        # 至少要有 1 条本国案例（核心）
        return bool(self.country_cases(topic_id, country_id))

    def has_comparison_material(self, topic_id: str, a: str, b: str) -> bool:
        # 两国各自至少要有 1 条案例
        return (bool(self.country_cases(topic_id, a))
                and bool(self.country_cases(topic_id, b)))

    def has_universal_material(self, topic_id: str) -> bool:
        return bool(self.global_theories(topic_id))

    # ==================================================================
    # 渲染（给 prompt 的 context block）
    # ==================================================================
    def render_single_country_blocks(
        self, topic_id: str, country_id: str
    ) -> Tuple[str, str, str, List[str]]:
        """返回 (country_theory_block, country_case_block, global_theory_block, used_ids)。"""
        used: List[str] = []
        country_theory = self._render(
            self.country_theories(topic_id, country_id), tag="CT", used_ids=used,
            fallback="（未检索到该国该主题的本土理论/指南；请谨慎出题并注明。）"
        )
        country_case = self._render(
            self.country_cases(topic_id, country_id), tag="CC", used_ids=used,
            fallback="（未检索到该国该主题的真实案例。）"
        )
        global_theory = self._render(
            self.global_theories(topic_id), tag="GT", used_ids=used,
            fallback="（未检索到国际理论文献。）"
        )
        return country_theory, country_case, global_theory, used

    def render_comparison_blocks(
        self, topic_id: str, a: str, b: str
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        返回 (blocks_dict, used_ids)。
        blocks_dict 含：
          a_theory / a_case / b_theory / b_case / global_theory
        """
        used: List[str] = []
        blocks = {
            "a_theory": self._render(
                self.country_theories(topic_id, a), tag=f"{a}T", used_ids=used,
                fallback=f"（未检索到 {a} 的本土理论/指南。）"
            ),
            "a_case": self._render(
                self.country_cases(topic_id, a), tag=f"{a}C", used_ids=used,
                fallback=f"（未检索到 {a} 的真实案例。）"
            ),
            "b_theory": self._render(
                self.country_theories(topic_id, b), tag=f"{b}T", used_ids=used,
                fallback=f"（未检索到 {b} 的本土理论/指南。）"
            ),
            "b_case": self._render(
                self.country_cases(topic_id, b), tag=f"{b}C", used_ids=used,
                fallback=f"（未检索到 {b} 的真实案例。）"
            ),
            "global_theory": self._render(
                self.global_theories(topic_id), tag="GT", used_ids=used,
                fallback="（未检索到国际理论文献。）"
            ),
        }
        return blocks, used

    def render_universal_blocks(self, topic_id: str) -> Tuple[str, List[str]]:
        used: List[str] = []
        text = self._render(
            self.global_theories(topic_id), tag="GT", used_ids=used,
            fallback="（未检索到国际理论文献。）"
        )
        return text, used

    # ==================================================================
    # 渲染工具
    # ==================================================================
    @staticmethod
    def _render(docs: List[Document], *, tag: str, used_ids: List[str],
                fallback: str) -> str:
        if not docs:
            return fallback
        lines: List[str] = []
        for i, d in enumerate(docs, 1):
            used_ids.append(d.doc_id)
            summary = (d.summary or "")[:_MAX_SUMMARY_CHARS].replace("\n", " ").strip()
            kps = "；".join((d.key_points or [])[:5]) or "（无）"
            src = " ｜ ".join(filter(None, [d.source, str(d.year) if d.year else None]))
            url = f" ({d.url})" if d.url else ""
            lines.append(
                f"[{tag}{i}] {d.title} ｜ {src}{url}\n"
                f"摘要：{summary}\n"
                f"要点：{kps}"
            )
        return "\n\n".join(lines)
