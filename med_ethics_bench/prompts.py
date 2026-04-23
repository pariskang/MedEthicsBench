from __future__ import annotations

"""prompts.py — hardened prompts for stage-3 JSON generation and LLM-judge evaluation."""


# ==================================================================
# Judge prompts (used by evaluator.py)
# ==================================================================

JUDGE_SYSTEM = """你是一位医学伦理基准考试的专业评审 LLM（judge）。
你的任务是对"被测模型"的作答进行多维度打分，并逐条判断关键知识点的命中情况。

【输出格式】严格输出一个 JSON 对象，不含任何 Markdown 围栏、前言或后记，结构如下：
{{
  "key_point_hits": [
    {{
      "key_point_id": "<与题目中 key_points[].id 完全一致>",
      "hit": true/false,
      "evidence": "<从被测回答中截取的命中证据（≤30字，1句），未命中则写 null>",
      "partial_credit": 0.0
    }}
  ],
  "rubric_scores": [
    {{
      "dimension": "<维度名，必须与提供的 rubric 维度完全一致>",
      "score": 0.0,
      "reason": "<评分理由（≤60字）>"
    }}
  ],
  "overall_feedback": "<整体评语（≤120字），含主要亮点和主要缺陷>"
}}

【评分原则】
1. 以"前沿模型区分"为标准：泛泛之谈不应得高分；精准法域引用、反方论证处理、多层难点覆盖才算优秀。
2. key_point_hits 须逐条判断；partial_credit：完整命中=1.0，部分命中=0.5，未命中=0.0。
3. rubric_scores 必须覆盖所有提供的评分维度，缺少任何维度视为无效输出。
4. citation_accuracy：引用的法律/机构名若存在明显错误（张冠李戴、混淆国家）即大幅扣分。
5. jurisdictional_precision：把他国法律套用到题目所在法域，该维度不超过 0.4。
6. counterargument_handling：论证型题若完全未提反方意见，不超过 0.3。
7. 选择题的 final_answer_correctness 由系统客观判分后覆盖，judge 仍需在 reason 说明推理质量。
8. 被测回答少于 40 字，所有维度一律不超过 0.3。
9. 【紧凑性要求】evidence ≤30字，reason ≤60字，overall_feedback ≤120字。严格遵守，勿超长。
"""

JUDGE_USER_TEMPLATE = """======= 题目信息 =======
[题型] {qtype}
[题轨] {track}
[所涉国家/法域] {country_ids}

[题干]
{stem}

{context_block}

[参考答案]
{reference_answer}

[关键知识点（Key Points）列表]
{key_points_block}

======= 评分维度权重 =======
{rubric_dimensions_block}

======= 被测模型的作答 =======
[推理过程 / Reasoning Trace]
{reasoning_trace}

[最终答案]
{final_answer}
{selected_labels_block}

======= 评审任务 =======
请严格按照 JUDGE_SYSTEM 中规定的 JSON 格式输出评审结果：
1. 对每个 Key Point 逐条判断命中情况（key_point_hits）；
2. 对每个评分维度打分并给出理由（rubric_scores），维度必须与上方权重列表完全匹配；
3. 给出整体评语（overall_feedback）。

记住：第一个非空白字符必须是 '{{'，最后一个非空白字符必须是 '}}'，不要输出任何 JSON 之外的内容。
"""


# ==================================================================
# Retrieval prompts (used by retriever.py)
# ==================================================================

RETRIEVAL_SYSTEM = """你是一位医学伦理与心理伦理学领域的专业文献研究员，具备网络检索能力。
你的任务是查找并结构化返回真实存在的文献、指南、法规或案例信息。

【输出格式】严格输出一个 JSON 对象，不含任何 Markdown 围栏、前言或后记。
- 对于理论/指南检索任务，顶层键为 "documents"，值为文献对象数组；
- 对于案例检索任务，顶层键为 "cases"，值为案例对象数组。

文献对象字段（documents 数组中每项）：
  title     - 文献完整标题
  url       - 可访问的原文链接，无则 null
  source    - 期刊名/机构名/媒体名
  authors   - 作者列表，如 ["作者1", "作者2"]
  year      - 出版年份（整数），不确定则 null
  region    - 地区标识，如 国际/中国/美国/英国/法国/俄罗斯/德国
  language  - zh 或 en
  summary   - 200-400 字的客观摘要，含核心论点、方法、结论
  key_points - 3-7 条关键论点，每条 20-80 字

案例对象字段（cases 数组中每项）：
  title     - 案例标题或简称
  url       - 原始报道/判决链接，无则 null
  source    - 来源媒体/机构/法院
  year      - 发生年份（整数），不确定则 null
  region    - 国家/地区
  language  - zh 或 en
  summary   - 150-350 字，含事实经过、核心伦理冲突、实际处理结果
  key_points - 3-5 条伦理张力点

【严格规则】
- 只返回真实存在的文献/案例，禁止捏造标题、作者、URL 或数据；
- 不确定某字段时写 null，不要猜测；
- 摘要必须客观，不得加入你自己的伦理立场；
- 禁止输出 JSON 之外的任何文字；
- 第一个非空白字符必须是 '{'，最后一个非空白字符必须是 '}'。
"""

RETRIEVAL_GLOBAL_THEORY_USER = """请检索 {k} 篇关于【{topic_zh}（{topic_en}）】的国际权威理论文献。

检索要求：
1. 优先选取：顶刊（NEJM/JAMA/Lancet/BMJ/AJOB/JME/Kennedy Institute of Ethics Journal）、WHO/联合国/UNESCO 指南、Beauchamp & Childress 经典著作及重要综述；
2. 时间范围：优先 2010 年后，经典著作可更早；
3. 必须覆盖不同立场/观点（支持 vs. 反对，不同伦理框架）；
4. 不绑定特定国家，面向全球/普适视角；
5. 每篇 summary 须体现：核心论点 + 使用的伦理框架 + 主要结论。

请以 JSON 格式返回，顶层键为 "documents"，包含 {k} 个文献对象。
"""

RETRIEVAL_COUNTRY_THEORY_USER = """请检索 {k} 篇【{country_zh}（{country_en}）】关于【{topic_zh}（{topic_en}）】的本国权威理论文献、官方指南或法规文件。

【该国伦理机构参考】{ethics_bodies}
【该国文化/法律背景】{cultural_notes}
【建议检索关键词/来源】{search_hints}

检索要求：
1. 优先选取：该国官方指南、该国权威期刊/学者、国家级法律法规、伦理委员会意见书；
2. 必须是该国本土文献，不要只返回国际文献；
3. summary 须说明：该文献在本国语境下的地位、具体规定或立场、与国际准则的异同；
4. 至少 1 篇应涉及具体法律条文或官方机构文件。

请以 JSON 格式返回，顶层键为 "documents"，包含 {k} 个文献对象。
"""

RETRIEVAL_CASE_COUNTRY_USER = """请检索 {k} 个发生在【{country_zh}（{country_en}）】的真实医学伦理案例，主题为【{topic_zh}（{topic_en}）】。

【该国文化/法律背景】{cultural_notes}
【建议检索关键词/来源】{search_hints}

检索要求：
1. 必须是真实发生的案例（有新闻报道、法院判决、机构公开文件等可查来源）；
2. 案例必须发生在 {country_zh} 境内，体现该国特有的制度/文化/法律约束；
3. 优先选取：有明显伦理争议、有多方利益冲突、有实际处理结果的案例；
4. summary 须包含：事实经过、各方立场、伦理冲突核心、实际结果；
5. key_points 须点出该案例在医学伦理教学中的核心价值（如揭示哪些原则冲突、有哪些争议至今未解）。

请以 JSON 格式返回，顶层键为 "cases"，包含 {k} 个案例对象。
"""


# ==================================================================
# Question generation prompts (used by question_generator.py)
# ==================================================================

QGEN_SYSTEM = """You are a benchmark item writer for an advanced medical ethics benchmark.
Your job is to output exactly ONE final JSON object and nothing else.

Critical constraints:
- Do NOT reveal planning, scratch work, chain-of-thought, or drafting text.
- Do NOT start with phrases like 'Let me think', 'I will create', or similar.
- The first non-whitespace character of your output must be '{'.
- The last non-whitespace character of your output must be '}'.
- Do NOT use Markdown fences.
- Keep all prose inside JSON string values.
- Inside prose strings, avoid raw ASCII double quotes. Prefer Chinese quotes “ ” / ‘ ’ or single quotes.
- If you need quotations in prose, do not write naked \" characters unless they are properly escaped for JSON.
"""

_JSON_FORMAT = """
只输出严格 JSON，结构如下（严格按此键名，字段顺序必须与此一致）：
{{
  "requires_reasoning_trace": true,
  "expected_principles": ["伦理原则1", "伦理原则2"],
  "expected_stakeholders": ["利益方1", "利益方2"],
  "common_pitfalls": ["陷阱1","陷阱2","陷阱3","陷阱4"],
  "key_points": [
    {{"id":"kp1","content":"...","weight":1.5,"required":true,"category":"principle"}}
  ],
  "options": [ ... 或 null（非选择题时） ],
  "stem": "...",
  "context": "可选背景；country/comparison 题通常不应为 null；无则写 null",
  "reference_answer": "..."
}}

严格规则：
- 字段顺序必须严格按上方模板，从 requires_reasoning_trace 开始，reference_answer 最后；
- key_points 至少 5 条，至少 3 条 required=true；
- key_points.category 从 ["content","principle","stakeholder","citation","counterargument","cultural_context"] 选；
- 至少覆盖 principle、stakeholder、citation、counterargument 四类；若为 country/comparison 题，还必须覆盖 cultural_context；
- 若为选择题（single/multi_choice），options 必须 4 个，格式 {{"label":"A","text":"...","is_correct":true/false,"rationale":"..."}}；
- 若非选择题，options 置 null；
- common_pitfalls 至少 4 条，其中至少 1 条应是法域/国家误套，至少 1 条应是把合法当成伦理最优，至少 1 条应是忽略程序义务或机构职责；
- stem 字数控制在 400–700 字（含关键细节即可，不要超长铺垫）；context 控制在 300–600 字；reference_answer 需完整但控制在 600–1200 字；
- reference_answer 必须体现：事实识别 → 规范识别 → 冲突权衡 → 反方处理 → 结论；
- 禁止输出 JSON 之外的任何文字；不要输出前言、解释、思考过程、标题或 Markdown；
- 输出必须直接以 {{ 开头，reference_answer 写完后立即关闭 }}，不得在结尾追加任何文字。
"""

QGEN_SINGLE_COUNTRY_TEMPLATE = """请基于以下材料，生成 1 道高难度的【{country_zh}】{qtype_zh}（{qtype_en}）考题。

【题型说明】{qtype_desc}
【目标主题】{topic_zh}
【目标国家】{country_zh}（{country_en}）
【该国法律/文化背景（必须用上）】{cultural_notes}
【该国主要伦理机构（题干或答案须出现至少 1 个）】{ethics_bodies}
【目标语言】{language}

========== 本国理论/指南/法律 ==========
{country_theory_blocks}

========== 本国真实案例 ==========
{country_case_blocks}

========== 国际理论文献（作为参照） ==========
{global_theory_blocks}
==============================

特别要求（单国深度题）：
1. 题干必须发生在 {country_zh} 的真实语境中，且必须出现该国法律条文名、机构名、或本土文化/制度约束；
2. 题目不能只是泛泛分析，必须设置至少两个层级的难点，如：新增事实、程序义务、患者意愿与家属诉求冲突、法律允许但伦理仍有争议；
3. reference_answer 必须引用至少 2 个该国具体法律/指南/机构依据，且说明为什么仅套用抽象四原则不够；
4. 必须设置一个近似合理但实际错误的陷阱，最好是把美国/国际模板直接迁移到 {country_zh}；
5. 至少 1 个 key_point 要求处理反方论证或指出 moral remainder；
""" + _JSON_FORMAT

QGEN_COMPARISON_TEMPLATE = """请基于以下材料，生成 1 道跨国对比题（{qtype_zh}）。

【目标主题】{topic_zh}
【对比国家】{country_a_zh}（{country_a_en}） vs {country_b_zh}（{country_b_en}）
【对比焦点（必须覆盖）】{comparison_focus}
【目标语言】{language}

【A 国（{country_a_zh}）材料】
--- A 国理论/指南 ---
{a_theory_blocks}
--- A 国案例 ---
{a_case_blocks}
--- A 国背景 ---
{a_cultural_notes}

【B 国（{country_b_zh}）材料】
--- B 国理论/指南 ---
{b_theory_blocks}
--- B 国案例 ---
{b_case_blocks}
--- B 国背景 ---
{b_cultural_notes}
==============================

特别要求（跨国对比题）：
1. 题干应使用同一个基础事实场景，但增加至少一个新增事实，使两国结论进一步分化或收敛；
2. 参考答案必须对称覆盖两国：处理路径、法律/机构/原则依据，以及差异不能被空泛价值表述抹平的原因；
3. 必须明确区分：法律允许性、机构程序、最终伦理可辩护性；
4. 至少 1 个 key_point 要求比较两国在程序正当性上的差异；
5. common_pitfalls 至少包含：只分析一国、把两国写得过于相似、错用一国术语描述另一国制度、只谈价值不落地到法源/机构。
""" + _JSON_FORMAT

QGEN_UNIVERSAL_TEMPLATE = """请基于以下国际材料，生成 1 道不绑定国别的 {qtype_zh}（{qtype_en}）。

【题型说明】{qtype_desc}
【目标主题】{topic_zh}
【目标语言】{language}

【国际理论文献】
{global_theory_blocks}
==============================

特别要求（普适困境题）：
1. 必须是即便熟悉一般四原则也不能一眼作答的难题；
2. 至少包含以下元素中的 2 个：信息不完全、预测不确定性、双重效应、角色义务冲突、短期利益 vs 长期制度后果、个案公平 vs 规则公平；
3. 不得假定具体国家法律背景；
4. reference_answer 必须呈现一个最强反方意见及其回应；
5. 至少 1 个 key_point 说明为什么不能仅靠某一个原则直接推出结论。
""" + _JSON_FORMAT
