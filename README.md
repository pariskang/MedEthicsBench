# MedEthicsBench / 医学伦理基准

> A frontier-hard, dynamic research-grade benchmark toolkit for **medical ethics reasoning**, designed for paper-level evaluation and reproducible experiments.
>
> 面向医学伦理推理的动态 benchmark 工具链，支持题库构建、模型作答与自动评分。

---

## 1) Project Positioning / 项目定位

**EN**
- MedEthicsBench focuses on hard medical-ethics reasoning tasks, with legal-jurisdiction anchoring, cultural context constraints, and strict rubric-based grading.
- The toolkit covers the full pipeline: retrieval corpus construction → question generation → model answering → automatic grading/reporting.
- This repository has been re-organized as a complete research-style Git project based on the provided archive.

**中文**
- MedEthicsBench 面向高难度医学伦理推理评测，强调法域锚点、跨文化情境与可判分 rubric。
- 工具链覆盖完整流程：语料检索与组织 → 题目生成 → 模型作答 → 自动评分与报告。
- 本仓库已基于你提供的压缩包重构为完整研究型 Git 项目结构。

---

## 2) Benchmark Snapshot / 基准概览

### 2.1 Track × Topic distribution / 赛道与主题分布

[View Figure 3 PDF / 查看图3 PDF](fig/Fig3_topic_by_track_heatmap.pdf)

### 2.2 Evidence composition / 证据来源构成

[View Figure 10 PDF / 查看图10 PDF](fig/Fig10_source_corpus_overview.pdf)

### 2.3 Annotation profile / 标注结构

[View Figure 4 PDF / 查看图4 PDF](fig/Fig4_annotation_density_by_format.pdf)


### 2.4 Prompt token distribution / 提示词长度分布

[View Figure 5 PDF / 查看图5 PDF](fig/Fig5_prompt_token_distribution.pdf)

### 2.5 Baseline performance tables / 基线性能表

**Table IV. Overall and track-level performance**

| Model | Overall | Pass rate | Single-country | Comparison | Universal |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 0.3637 | 0.0000 | 0.3692 | 0.3444 | 0.3655 |
| Gemini 3.1 Pro | 0.1660 | 0.0000 | 0.1568 | 0.1580 | 0.2386 |
| GPT-5.4 | 0.5963 | 0.0506 | 0.6025 | 0.5544 | 0.6318 |

**Table V. Mean score by jurisdiction**

| Model | CN | US | UK | FR | RU | DE |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 0.3224 | 0.3670 | 0.3613 | 0.3671 | 0.3954 | 0.3476 |
| Gemini 3.1 Pro | 0.1780 | 0.1704 | 0.1566 | 0.1356 | 0.1459 | 0.1457 |
| GPT-5.4 | 0.5589 | 0.5705 | 0.5803 | 0.6131 | 0.6223 | 0.5824 |

---

---

## Citation / 引用信息

If you use **MedEthicsBench**, please cite:

```bibtex
@misc{kang2026medethicsbench,
  title={MedEthicsBench: Evaluating Medical Ethics Reasoning Across Jurisdictions},
  author={Yanlan Kang and Lee shou-yu and Liying Chu and Sunsi Wu and Wenqing Qu and Weichen Liu and Longlong Cao and Chengbin Hou and William Cheng-Chung Chu},
  year={2026}
}
```

## 3) Repository Structure / 仓库结构

```text
.
├── main.py                          # CLI entry / 命令行入口
├── requirements.txt
├── fig/
│   ├── Fig3_topic_by_track_heatmap.pdf
│   ├── Fig4_annotation_density_by_format.pdf
│   ├── Fig5_prompt_token_distribution.pdf
│   └── Fig10_source_corpus_overview.pdf
├── med_ethics_bench/
│   ├── config.py                    # config and defaults / 配置
│   ├── pipeline.py                  # main orchestration / 主流程
│   ├── retriever.py                 # source retrieval / 检索
│   ├── question_generator.py        # benchmark generation / 出题
│   ├── evaluator.py                 # grading logic / 评分
│   ├── knowledge_base.py            # KB utilities / 知识库工具
│   ├── poe_client.py                # API client
│   ├── prompts.py                   # prompting templates
│   ├── schemas.py                   # data schemas
│   └── concurrency.py               # parallel execution
├── data/
│   ├── documents.jsonl              # raw+processed source documents
│   ├── benchmark.jsonl              # built benchmark items
│   ├── benchmark_meta.json
│   ├── benchmark_stats.json
│   └── question_failures.jsonl
├── examples/
│   └── run_example.py
└── med_ethics_benchmark_fixedV5.zip # original archive / 原始压缩包
```

---

## 4) Installation / 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API key / 设置 API Key：

```bash
export POE_API_KEY="your_key_here"
```

> Note / 注意：`main.py` currently checks `POE_API_KEY` before dispatching subcommands.

---

## 5) Quick Start / 快速开始

### 5.1 Estimate benchmark size / 预估题量

```bash
python main.py estimate
```

### 5.2 Build benchmark (recommended zh+en) / 构建题库（推荐中英双语）

```bash
python main.py build --zh-en
```

### 5.3 Stage-3 only generation / 仅从第3阶段生成题目

```bash
python main.py generate --output-dir ./data --zh-en
```

### 5.4 Run model answers / 模型作答

```bash
python main.py answer --model GPT-5.4
```

### 5.5 Grade model outputs / 自动评分

```bash
python main.py grade --model GPT-5.4
```

---

## 6) Research-Oriented Recommendations / 论文级实验建议

**EN**
1. Report both overall score and per-track score (single-country / comparison / universal).
2. Add per-jurisdiction slices to avoid overclaiming global generalization.
3. Include error taxonomy (citation failure, legal anchor miss, stakeholder omission, weak counterargument handling).
4. Perform human audit on 30–50 sampled items to verify rubric strictness and trap validity.

**中文**
1. 同时报告总体分数与各赛道分数（single-country / comparison / universal）。
2. 加入分法域统计，避免“全球泛化”过度结论。
3. 提供错误类型分析（引文错误、法域锚点缺失、利益相关者遗漏、反方论证不足）。
4. 抽样 30–50 题人工复核，验证 rubric 严格性与陷阱有效性。
