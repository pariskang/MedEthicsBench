"""
examples/run_example.py — 最小冒烟跑通

建议先把 TOPIC_AXES 精简成 1 个主题、QUESTION_TYPES 精简成 2 个题型，
验证 API key 和链路通了，再放开全量配置。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from med_ethics_bench import Config, BenchmarkPipeline
from med_ethics_bench import config as cfg_mod


def main():
    # —— 临时缩小配置，加快冒烟 ——
    cfg_mod.TOPIC_AXES = [cfg_mod.TOPIC_AXES[0]]                        # 只保留第 1 个主题
    cfg_mod.QUESTION_TYPES = [cfg_mod.QUESTION_TYPES[0], cfg_mod.QUESTION_TYPES[2]]
    # 只保留单选题和案例分析题

    c = Config()
    c.theory_docs_per_topic = 2
    c.case_docs_per_topic = 2
    c.questions_per_cell = 1
    c.output_dir = "./data_smoke"

    pipe = BenchmarkPipeline(c)

    print("▶ Step 1/3: 构建 benchmark（含联网检索）")
    pipe.build_benchmark(languages=["zh"])

    print("\n▶ Step 2/3: 让 Claude-Sonnet-4.6 作答（被测模型示例）")
    pipe.run_model("Claude-Sonnet-4.6")

    print("\n▶ Step 3/3: 评分")
    report = pipe.grade("Claude-Sonnet-4.6")
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
