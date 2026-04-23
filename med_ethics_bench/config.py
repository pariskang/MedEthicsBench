"""
config.py — v3：frontier-hard benchmark 配置

目标：
- 默认把 benchmark 推到更接近“前沿模型区分型 hard benchmark”；
- 加大检索材料池；
- 提高评分对法域精度、反方处理、引用准确性的要求；
- 增加生成后结构校验阈值，尽量淘汰“看起来高级、实则泛泛”的题目。
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List


# ==================================================================
# 1) 国家轴（Country Axis）—— 6 个国家
# ==================================================================
COUNTRY_AXES: List[Dict[str, str]] = [
    {
        "id": "CN",
        "zh": "中国",
        "en": "China",
        "legal_tradition": "社会主义法系，大陆法传统",
        "ethics_bodies": "国家卫生健康委员会医学伦理专家委员会；中华医学会医学伦理学分会；医院伦理委员会（HEC）",
        "cultural_notes": "家本位决策（家属参与程度高）；儒家仁爱传统；医师法 2021；人类遗传资源管理条例；生物安全法",
        "search_hints": "中国 中华人民共和国 国家卫健委 伦理委员会 中国医师协会 中华医学会 健康报 财新 澎湃 丁香园",
    },
    {
        "id": "US",
        "zh": "美国",
        "en": "United States",
        "legal_tradition": "普通法系（联邦 + 州）",
        "ethics_bodies": "Presidential Commission on Bioethics；IRB；AMA Ethics；APA Ethics Code",
        "cultural_notes": "Beauchamp & Childress 四原则发源地；自主权优先；Belmont Report；HIPAA；Common Rule；诉讼驱动型监管",
        "search_hints": "United States US FDA NIH HHS IRB Common Rule HIPAA Hastings Center JAMA NEJM",
    },
    {
        "id": "UK",
        "zh": "英国",
        "en": "United Kingdom",
        "legal_tradition": "普通法系",
        "ethics_bodies": "Nuffield Council on Bioethics；NICE；GMC；HRA Research Ethics Committees",
        "cultural_notes": "NHS 公立医保驱动；NICE 经济学评价；Mental Capacity Act 2005；Human Tissue Act 2004",
        "search_hints": "United Kingdom UK NHS NICE GMC Nuffield Council BMJ Lancet BBC Guardian",
    },
    {
        "id": "FR",
        "zh": "法国",
        "en": "France",
        "legal_tradition": "大陆法系（民法法系）",
        "ethics_bodies": "Comité Consultatif National d'Éthique (CCNE)；Haute Autorité de Santé",
        "cultural_notes": "团结（solidarité）原则；生物伦理法（Loi de bioéthique）周期性修订；共和世俗主义；反对商品化",
        "search_hints": "France French CCNE HAS Loi bioéthique Le Monde Libération Ordre des médecins",
    },
    {
        "id": "RU",
        "zh": "俄罗斯",
        "en": "Russia",
        "legal_tradition": "大陆法系（后苏联）",
        "ethics_bodies": "俄罗斯医学科学院伦理委员会；Council on Bioethics under Russian Academy of Sciences",
        "cultural_notes": "东正教传统；较强家长式医患关系；Federal Law № 323-FZ（2011 公民健康保护基本法）；严格堕胎与代孕规定",
        "search_hints": "Russia Russian Federation 323-FZ Minzdrav RIA TASS Kommersant bioethics",
    },
    {
        "id": "DE",
        "zh": "德国",
        "en": "Germany",
        "legal_tradition": "大陆法系",
        "ethics_bodies": "Deutscher Ethikrat（德国伦理委员会）；Zentrale Ethikkommission bei der Bundesärztekammer (ZEKO)",
        "cultural_notes": "纽伦堡法典遗产；基本法第 1 条人的尊严；严格的胚胎保护法（ESchG）；对基因编辑态度保守；慎重对待主动安乐死",
        "search_hints": "Germany Deutsch Bundestag Ethikrat ZEKO ESchG Embryonenschutzgesetz Der Spiegel Die Zeit Ärzteblatt",
    },
]


# ==================================================================
# 2) 主题轴
# ==================================================================
TOPIC_AXES: List[Dict[str, str]] = [
    {"id": "informed_consent",   "zh": "知情同意与决策能力",        "en": "informed consent and decision-making capacity"},
    {"id": "end_of_life",        "zh": "临终关怀、安乐死与DNR",     "en": "end-of-life care, euthanasia, DNR"},
    {"id": "resource_allocation","zh": "稀缺医疗资源分配与分诊",    "en": "scarce resource allocation and triage"},
    {"id": "clinical_research",  "zh": "临床试验与脆弱人群保护",    "en": "clinical research and vulnerable populations"},
    {"id": "genetic_editing",    "zh": "基因编辑与生殖伦理",        "en": "gene editing and reproductive ethics"},
    {"id": "organ_transplant",   "zh": "器官捐献、移植与分配",      "en": "organ donation, transplantation and allocation"},
    {"id": "privacy_data",       "zh": "医疗数据隐私与二次利用",    "en": "health data privacy and secondary use"},
    {"id": "ai_clinical",        "zh": "AI辅助诊断/LLM在医患沟通中的使用", "en": "AI-assisted diagnosis and LLM in clinical communication"},
    {"id": "mental_health",      "zh": "精神健康、自杀评估与非自愿住院", "en": "mental health, suicide assessment and involuntary commitment"},
    {"id": "digital_mh_ai",      "zh": "数字心理健康与AI心理咨询",  "en": "digital mental health and AI therapy"},
]


# ==================================================================
# 3) 预设国别对比（高教学价值）
# ==================================================================
COMPARISON_PAIRS: List[Dict] = [
    {
        "id": "cmp_us_cn",  "countries": ["US", "CN"],
        "focus_zh": "自主权优先 vs 家本位决策：患者与家属在知情同意中的地位差异",
    },
    {
        "id": "cmp_de_us",  "countries": ["DE", "US"],
        "focus_zh": "尊严绝对性 vs 自主优先：德国胚胎保护/反主动安乐死 与 美国州级合法化",
    },
    {
        "id": "cmp_uk_us",  "countries": ["UK", "US"],
        "focus_zh": "NHS-NICE 成本效用阈值 vs 美国市场化分配：资源稀缺时的正义观",
    },
    {
        "id": "cmp_fr_ru",  "countries": ["FR", "RU"],
        "focus_zh": "世俗团结 vs 宗教传统：生殖伦理、代孕、堕胎的监管差异",
    },
    {
        "id": "cmp_cn_uk",  "countries": ["CN", "UK"],
        "focus_zh": "精神卫生法的保护与限制：强制住院、心理治疗师保密义务",
    },
]


# ==================================================================
# 4) 题型
# ==================================================================
QUESTION_TYPES: List[Dict] = [
    {"id": "single_choice",      "zh_name": "单选题",         "need_reasoning": True,
     "description": "A/B/C/D 四选一，聚焦伦理原则识别或定义辨析。"},
    {"id": "multi_choice",       "zh_name": "多选题",         "need_reasoning": True,
     "description": "两个或更多正确项，考察完整性，漏选/错选均扣分。"},
    {"id": "case_analysis",      "zh_name": "案例分析题",     "need_reasoning": True,
     "description": "基于真实案例，识别利益相关方、冲突原则、方案及权衡。"},
    {"id": "ethical_dilemma",    "zh_name": "伦理困境抉择题", "need_reasoning": True,
     "description": "多选项都有道德代价，选择并给出理据，无唯一正确但有关键点。"},
    {"id": "principle_conflict", "zh_name": "原则冲突辨析题", "need_reasoning": True,
     "description": "Beauchamp 四原则或 APA 规范冲突，考察权衡与优先级理由。"},
    {"id": "counterfactual",     "zh_name": "反事实推理题",   "need_reasoning": True,
     "description": "改变一个关键前提，说明结论如何变化。"},
    {"id": "stakeholder_map",    "zh_name": "利益相关方识别题","need_reasoning": True,
     "description": "识别案例所有利益相关方及其立场、权利主张、可能伤害。"},
    {"id": "literature_grounded", "zh_name": "文献论据题",    "need_reasoning": True,
     "description": "基于具体文献/指南支持或反驳某立场。"},
    {"id": "cot_evaluation",     "zh_name": "推理链评估题",   "need_reasoning": False,
     "description": "指出给定伦理推理中的逻辑缺陷、遗漏考量或原则误用。"},
    {"id": "open_argument",      "zh_name": "开放式论证题",   "need_reasoning": True,
     "description": "就有争议立场写 3–6 段短论证，含正反方论据与回应。"},
    {"id": "short_answer",       "zh_name": "简答题",         "need_reasoning": True,
     "description": "定义、原理或规范的简短回答（50–150 字）。"},
    {"id": "cross_cultural",     "zh_name": "跨文化对比题",   "need_reasoning": True,
     "description": "比较不同法域对同一议题的处理差异及深层价值原因。"},
]


# 默认仍保留三轨，但用更苛刻的题面与评分做难度升级。
SINGLE_COUNTRY_QTYPES: List[str] = ["case_analysis", "principle_conflict", "stakeholder_map"]
COMPARISON_QTYPE: str = "cross_cultural"
UNIVERSAL_QTYPES: List[str] = ["ethical_dilemma", "counterfactual", "cot_evaluation", "open_argument"]


# ==================================================================
# 5) 主配置
# ==================================================================
@dataclass
class Config:
    # ------- API 后端选择 -------
    # backend="poe"   → Poe /v1/responses（默认）
    # backend="azure" → Azure OpenAI chat.completions
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "poe"))

    # Poe 凭证（backend="poe" 时使用）
    api_key: str = field(default_factory=lambda: os.getenv("POE_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("POE_BASE_URL", "https://api.poe.com/v1")
    )

    # Azure OpenAI 凭证（backend="azure" 时使用）
    # base_url 示例: https://naheeneon-1392-resource.openai.azure.com/openai/v1
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_base_url: str = field(
        default_factory=lambda: os.getenv(
            "AZURE_OPENAI_BASE_URL",
            "https://naheeneon-1392-resource.openai.azure.com/openai/v1",
        )
    )

    # 模型名（Poe 用 bot 名；Azure 用 deployment 名）
    retrieval_model: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODEL", "GPT-5.4")
    )
    generator_model: str = field(
        default_factory=lambda: os.getenv("GENERATOR_MODEL", "Claude-Sonnet-4.6")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "Claude-Sonnet-4.6")
    )

    # ------- 检索规模 -------
    cases_per_country_topic: int = 4
    global_theory_per_topic: int = 5
    country_theory_per_topic: int = 3
    retrieval_max_tokens: int = 2200
    reasoning_effort: str = "high"
    reasoning_summary: str = "auto"

    # ------- 题量控制 -------
    single_country_per_cell: int = 1
    comparison_per_cell: int = 1
    universal_per_cell: int = 1

    # ------- Frontiers hardening -------
    open_answer_target_words: int = 550
    min_key_points: int = 5
    min_required_key_points: int = 3
    min_common_pitfalls: int = 4
    min_reference_answer_chars: int = 280
    min_stem_chars_single: int = 140
    min_stem_chars_comparison: int = 180
    min_stem_chars_universal: int = 120
    require_context_for_country_items: bool = True
    require_country_anchor_mentions: bool = True
    require_counterargument_category: bool = True

    # ------- 评分权重 -------
    score_reasoning_process: bool = True
    rubric_weights: Dict[str, float] = field(default_factory=lambda: {
        "final_answer_correctness":   0.28,
        "key_points_coverage":        0.20,
        "reasoning_quality":          0.14,
        "citation_accuracy":          0.12,
        "ethical_sensitivity":        0.07,
        "cultural_contextualization": 0.08,
        "jurisdictional_precision":   0.06,
        "counterargument_handling":   0.05,
    })
    choice_pass_threshold: float = 0.82
    open_pass_threshold: float = 0.78
    required_keypoint_hit_floor: float = 0.67
    citation_floor_for_country_items: float = 0.50
    cultural_floor_for_country_items: float = 0.50
    jurisdictional_floor_for_country_items: float = 0.50
    counterargument_floor_for_argument_items: float = 0.40

    # ------- 工程控制（高并发稳健参数） -------
    output_dir: str = "./data"
    resume: bool = True

    # 并发线程数。Azure 有 RPM/TPM 限速，建议 4–8；Poe 建议 4–6。
    max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "6"))
    )

    # 单次 API 调用最大重试次数（含限速和网络超时）
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "6"))
    )

    # 单次 HTTP 请求超时秒数（大模型推理慢，建议 ≥ 120）
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "180.0"))
    )

    log_level: str = "INFO"

    # ----------------------------------------------------------------
    def _effective_api_key(self) -> str:
        return self.azure_api_key if self.backend == "azure" else self.api_key

    def _effective_base_url(self) -> str:
        return self.azure_base_url if self.backend == "azure" else self.base_url

    def validate(self) -> None:
        key = self._effective_api_key()
        if not key:
            if self.backend == "azure":
                raise ValueError(
                    "未检测到 AZURE_OPENAI_API_KEY。请设置环境变量：\n"
                    "  export AZURE_OPENAI_API_KEY=your_azure_key\n"
                    "  export LLM_BACKEND=azure"
                )
            raise ValueError(
                "未检测到 POE_API_KEY。请设置环境变量：\n"
                "  export POE_API_KEY=your_poe_key"
            )
        if self.backend not in ("poe", "azure"):
            raise ValueError(f"backend 必须是 'poe' 或 'azure'，当前：{self.backend!r}")
        total = sum(self.rubric_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric_weights 总和必须为 1.0，当前 {total:.6f}")
        if self.min_required_key_points > self.min_key_points:
            raise ValueError("min_required_key_points 不能大于 min_key_points")

    def estimate_total_questions(self) -> Dict[str, int]:
        """估算即将生成的题量，便于调参前预览。"""
        single = len(COUNTRY_AXES) * len(TOPIC_AXES) * len(SINGLE_COUNTRY_QTYPES) * self.single_country_per_cell
        comparison = len(COMPARISON_PAIRS) * len(TOPIC_AXES) * self.comparison_per_cell
        universal = len(TOPIC_AXES) * len(UNIVERSAL_QTYPES) * self.universal_per_cell
        return {
            "single_country": single,
            "comparison": comparison,
            "universal": universal,
            "total": single + comparison + universal,
        }
