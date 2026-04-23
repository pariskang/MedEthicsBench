"""
retriever.py — v2：多国分层检索

三类任务：
1) 全球理论（by topic）：不分国家，顶刊理论 + 国际指南；
2) 国别理论/指南（by topic × country）：本国官方文件、本国学者代表作；
3) 国别案例（by topic × country）：本国真实案例。

所有产物统一落到 documents.jsonl，通过 country_id 区分。
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Set

from .concurrency import run_parallel
from .config import Config, TOPIC_AXES, COUNTRY_AXES
from .poe_client import PoeClient
from .prompts import (
    RETRIEVAL_SYSTEM,
    RETRIEVAL_GLOBAL_THEORY_USER,
    RETRIEVAL_COUNTRY_THEORY_USER,
    RETRIEVAL_CASE_COUNTRY_USER,
)
from .schemas import Document, DocType

log = logging.getLogger(__name__)


def _doc_hash(title: str, url: Optional[str], country_id: Optional[str]) -> str:
    key = (url or "") + "|" + (title or "") + "|" + (country_id or "")
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


class Retriever:
    def __init__(self, client: PoeClient, config: Config):
        self.client = client
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        self.docs_path = Path(config.output_dir) / "documents.jsonl"

    # ------------------------------------------------------------------
    def run(self) -> List[Document]:
        existing = self._load_existing()
        seen: Set[str] = {d.doc_id for d in existing}

        # ---- 任务清单 ----
        tasks = []
        # ① 全球理论（每主题一次）
        for topic in TOPIC_AXES:
            tasks.append({
                "kind": "global_theory",
                "topic": topic,
                "country": None,
                "k": self.config.global_theory_per_topic,
            })
        # ② 国别理论
        for topic in TOPIC_AXES:
            for country in COUNTRY_AXES:
                tasks.append({
                    "kind": "country_theory",
                    "topic": topic,
                    "country": country,
                    "k": self.config.country_theory_per_topic,
                })
        # ③ 国别案例
        for topic in TOPIC_AXES:
            for country in COUNTRY_AXES:
                tasks.append({
                    "kind": "country_case",
                    "topic": topic,
                    "country": country,
                    "k": self.config.cases_per_country_topic,
                })

        log.info("检索任务总数 = %d（已存在 %d 篇）", len(tasks), len(seen))

        new_docs: List[Document] = []

        def _worker(task: dict) -> List[Document]:
            return self._retrieve_one(**task)

        for task, docs, err in run_parallel(
            tasks, _worker,
            max_workers=self.config.max_concurrency,
            desc="检索",
        ):
            if err is not None:
                log.warning("检索失败 kind=%s topic=%s country=%s: %s",
                            task["kind"], task["topic"]["id"],
                            (task["country"] or {}).get("id"), err)
                continue
            for d in (docs or []):
                if d.doc_id in seen:
                    continue
                seen.add(d.doc_id)
                new_docs.append(d)
                self._append_doc(d)    # 主线程串行写盘，无需锁

        log.info("本次新增 %d 篇，累计 %d 篇", len(new_docs), len(seen))
        return existing + new_docs

    # ------------------------------------------------------------------
    # 单次检索分派
    # ------------------------------------------------------------------
    def _retrieve_one(self, *, kind: str, topic: dict, country: Optional[dict], k: int) -> List[Document]:
        if kind == "global_theory":
            return self._retrieve_global_theory(topic=topic, k=k)
        if kind == "country_theory":
            return self._retrieve_country_theory(topic=topic, country=country, k=k)
        if kind == "country_case":
            return self._retrieve_country_case(topic=topic, country=country, k=k)
        raise ValueError(f"未知 kind: {kind}")

    # ---- ① 全球理论 ----
    def _retrieve_global_theory(self, *, topic: dict, k: int) -> List[Document]:
        data = self.client.respond_json(
            model=self.config.retrieval_model,
            system=RETRIEVAL_SYSTEM,
            user=RETRIEVAL_GLOBAL_THEORY_USER.format(
                topic_zh=topic["zh"], topic_en=topic["en"], k=k
            ),
            enable_web_search=True,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_summary=self.config.reasoning_summary,
            max_output_tokens=self.config.retrieval_max_tokens,
        )
        return self._parse_docs(
            data.get("documents") or [],
            topic_id=topic["id"],
            country_id=None,
            doc_type=DocType.THEORY,
        )

    # ---- ② 国别理论 ----
    def _retrieve_country_theory(self, *, topic: dict, country: dict, k: int) -> List[Document]:
        data = self.client.respond_json(
            model=self.config.retrieval_model,
            system=RETRIEVAL_SYSTEM,
            user=RETRIEVAL_COUNTRY_THEORY_USER.format(
                topic_zh=topic["zh"], topic_en=topic["en"],
                country_zh=country["zh"], country_en=country["en"],
                ethics_bodies=country["ethics_bodies"],
                cultural_notes=country["cultural_notes"],
                search_hints=country["search_hints"],
                k=k,
            ),
            enable_web_search=True,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_summary=self.config.reasoning_summary,
            max_output_tokens=self.config.retrieval_max_tokens,
        )
        return self._parse_docs(
            data.get("documents") or [],
            topic_id=topic["id"],
            country_id=country["id"],
            doc_type=DocType.GUIDELINE,
        )

    # ---- ③ 国别案例 ----
    def _retrieve_country_case(self, *, topic: dict, country: dict, k: int) -> List[Document]:
        data = self.client.respond_json(
            model=self.config.retrieval_model,
            system=RETRIEVAL_SYSTEM,
            user=RETRIEVAL_CASE_COUNTRY_USER.format(
                topic_zh=topic["zh"], topic_en=topic["en"],
                country_zh=country["zh"], country_en=country["en"],
                cultural_notes=country["cultural_notes"],
                search_hints=country["search_hints"],
                k=k,
            ),
            enable_web_search=True,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_summary=self.config.reasoning_summary,
            max_output_tokens=self.config.retrieval_max_tokens,
        )
        return self._parse_docs(
            data.get("cases") or [],
            topic_id=topic["id"],
            country_id=country["id"],
            doc_type=DocType.CASE,
        )

    # ------------------------------------------------------------------
    # 通用解析
    # ------------------------------------------------------------------
    def _parse_docs(self, items: List[dict], *, topic_id: str,
                    country_id: Optional[str], doc_type: DocType) -> List[Document]:
        out: List[Document] = []
        for it in items:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            url = it.get("url")
            out.append(Document(
                doc_id=_doc_hash(title, url, country_id),
                doc_type=doc_type,
                title=title,
                url=url,
                source=it.get("source"),
                authors=it.get("authors") or None,
                year=it.get("year"),
                region=it.get("region"),
                country_id=country_id,
                language=it.get("language"),
                summary=(it.get("summary") or "").strip(),
                key_points=[s for s in (it.get("key_points") or []) if s],
                topic_ids=[topic_id],
                raw_snippet=json.dumps(it, ensure_ascii=False),
            ))
        return out

    # ------------------------------------------------------------------
    def _append_doc(self, doc: Document) -> None:
        with open(self.docs_path, "a", encoding="utf-8") as f:
            f.write(doc.model_dump_json() + "\n")

    def _load_existing(self) -> List[Document]:
        if not (self.config.resume and self.docs_path.exists()):
            if self.docs_path.exists():
                self.docs_path.unlink()
            return []
        docs: List[Document] = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(Document.model_validate_json(line))
                except Exception as e:  # noqa: BLE001
                    log.warning("跳过损坏行：%s", e)
        return docs
