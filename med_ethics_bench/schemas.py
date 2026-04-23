"""
schemas.py — 全流程数据模型

所有跨模块传递的结构都在此定义，基于 pydantic v2：
- 自动校验字段类型与取值；
- 支持 .model_dump_json() 直接落盘；
- 上下游模块只依赖这里的 schema，不再散落 dict。
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


# ==================================================================
# 1. 文献 / 案例
# ==================================================================
class DocType(str, Enum):
    THEORY = "theory"          # 理论/综述/社论/指南
    CASE = "case"              # 具体事件/判例/新闻
    GUIDELINE = "guideline"    # 官方指南、监管文件
    STATUTE = "statute"        # 法律法规条文


class Document(BaseModel):
    """通用文档对象，覆盖论文和新闻案例。"""
    doc_id: str = Field(..., description="内部稳定 id (hash)")
    doc_type: DocType
    title: str
    url: Optional[str] = None
    source: Optional[str] = None        # 期刊名 / 媒体名 / 机构名
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    region: Optional[str] = None        # 国际 / 中国 / 欧盟 / 美国 ...
    country_id: Optional[str] = None    # CN / US / UK / FR / RU / DE / None(=全球)
    language: Optional[str] = None      # zh / en / ...
    summary: str                        # 由 LLM 生成的 200–400 字摘要
    key_points: List[str] = Field(default_factory=list)   # 3–7 条关键论点
    topic_ids: List[str] = Field(default_factory=list)    # 关联的 TOPIC_AXES.id
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    # 原始检索产物，保留以便追溯
    raw_snippet: Optional[str] = None


class Case(BaseModel):
    """高价值案例的结构化表示（从 Document 中抽取，用于题目生成）。"""
    case_id: str
    title: str
    region: Optional[str] = None
    year: Optional[int] = None
    scenario: str                                       # 事实陈述
    stakeholders: List[str] = Field(default_factory=list)
    conflicting_principles: List[str] = Field(default_factory=list)
    actual_outcome: Optional[str] = None                # 现实中的实际处理/判决
    ethical_tensions: List[str] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)


# ==================================================================
# 2. 题目
# ==================================================================
class Option(BaseModel):
    label: str           # A/B/C/D/E
    text: str
    is_correct: bool = False
    rationale: Optional[str] = None   # 为何正确/错误


class KeyPoint(BaseModel):
    """评分用的关键信息点（rubric 粒度最小单位）。"""
    id: str                          # kp1, kp2 ...
    content: str                     # 关键点内容
    weight: float = 1.0              # 相对权重（同一题下会归一化）
    required: bool = False           # True 表示缺失即判大幅扣分
    category: str = "content"        # content / principle / stakeholder / citation / counterargument


class Question(BaseModel):
    question_id: str
    type: str                        # QUESTION_TYPES.id
    topic_id: str                    # TOPIC_AXES.id
    language: str = "zh"             # zh / en
    difficulty: str = "hard"         # easy / medium / hard / expert

    # —— 国别维度 ——
    # "single_country" | "comparison" | "universal"
    track: str = "universal"
    country_ids: List[str] = Field(default_factory=list)   # 所涉及的国家 id 列表
    comparison_pair_id: Optional[str] = None                # 对比题引用的 pair id

    stem: str                        # 题干
    context: Optional[str] = None    # 背景材料（案例原文、文献节选等）
    options: Optional[List[Option]] = None   # 仅选择题有

    # —— 评分关键结构 ——
    reference_answer: str            # 参考答案（完整、可执行）
    key_points: List[KeyPoint]       # rubric：评分用关键点
    expected_principles: List[str] = Field(default_factory=list)  # 应被引用的伦理原则/规范
    expected_stakeholders: List[str] = Field(default_factory=list)
    common_pitfalls: List[str] = Field(default_factory=list)      # 常见错误答案
    requires_reasoning_trace: bool = True

    # —— 溯源 ——
    source_doc_ids: List[str] = Field(default_factory=list)
    generated_by: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ==================================================================
# 3. 被测模型的回答与评估
# ==================================================================
class ModelAnswer(BaseModel):
    question_id: str
    model_name: str
    # 被测模型的"思考过程"，来自 reasoning summary 或 <think>...</think>
    reasoning_trace: Optional[str] = None
    final_answer: str
    # 若为选择题，额外给出选中的标签，便于机器判分
    selected_labels: Optional[List[str]] = None
    raw_response: Optional[Dict[str, Any]] = None


class RubricScore(BaseModel):
    dimension: str                   # 来自 Config.rubric_weights 的 key
    score: float                     # 0 – 1
    reason: str                      # judge 给出的评分理由


class KeyPointHit(BaseModel):
    key_point_id: str
    hit: bool
    evidence: Optional[str] = None   # 从答案中截取的命中证据
    partial_credit: float = 0.0      # 命中百分比，0–1


class Evaluation(BaseModel):
    question_id: str
    model_name: str
    judge_model: str

    # 细粒度
    key_point_hits: List[KeyPointHit]
    rubric_scores: List[RubricScore]

    # 聚合
    final_score: float               # 0 – 1，按 rubric_weights 加权
    passed: bool                     # 是否达标（默认 >= 0.7）

    # judge 的整体评语
    overall_feedback: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ==================================================================
# 4. 打包到 Benchmark 最终产物
# ==================================================================
class BenchmarkItem(BaseModel):
    """最终对外发布的 benchmark 单条记录。"""
    question: Question
    # 可选：预生成的参考模型答案 + 评估结果（作为 "gold" 参考）
    reference_model_answer: Optional[ModelAnswer] = None
    reference_evaluation: Optional[Evaluation] = None


class BenchmarkMeta(BaseModel):
    name: str = "MedEthicsPsychBench"
    version: str = "0.1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    n_items: int = 0
    topic_ids: List[str] = Field(default_factory=list)
    question_types: List[str] = Field(default_factory=list)
    generator_model: Optional[str] = None
    retrieval_model: Optional[str] = None
    notes: Optional[str] = None
