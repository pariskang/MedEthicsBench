from __future__ import annotations

"""main.py — CLI entry with safer local-import precedence and flexible arg parsing."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from med_ethics_bench import BenchmarkPipeline, Config  # noqa: E402


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir")
    parser.add_argument("--global-theory-per-topic", type=int)
    parser.add_argument("--country-theory-per-topic", type=int)
    parser.add_argument("--cases-per-country-topic", type=int)
    parser.add_argument("--single-country-per-cell", type=int)
    parser.add_argument("--comparison-per-cell", type=int)
    parser.add_argument("--universal-per-cell", type=int)
    parser.add_argument("--retrieval-model")
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续做")
    parser.add_argument("--log-level", default="INFO")


def _make_config(args: argparse.Namespace) -> Config:
    c = Config()
    for name in [
        "output_dir", "global_theory_per_topic", "country_theory_per_topic",
        "cases_per_country_topic", "single_country_per_cell", "comparison_per_cell",
        "universal_per_cell", "retrieval_model", "generator_model", "judge_model",
    ]:
        val = getattr(args, name, None)
        if val is not None:
            setattr(c, name, val)
    if getattr(args, "no_resume", False):
        c.resume = False
    if getattr(args, "log_level", None):
        c.log_level = args.log_level
    return c


def cmd_estimate(args):
    print(json.dumps(_make_config(args).estimate_total_questions(), indent=2, ensure_ascii=False))


def cmd_build(args):
    pipe = BenchmarkPipeline(_make_config(args))
    langs = ["zh", "en"] if getattr(args, "zh_en", False) else ["zh"]
    path = pipe.build_benchmark(languages=langs)
    print(f"\n✅ Benchmark 构建完成: {path}")


def cmd_repack(args):
    """从已有 questions.jsonl 重建 benchmark.jsonl，不调用任何 LLM。"""
    pipe = BenchmarkPipeline(_make_config(args))
    path = pipe.repack_from_questions()
    print(f"\n✅ repack 完成: {path}")


def cmd_generate(args):
    pipe = BenchmarkPipeline(_make_config(args))
    langs = ["zh", "en"] if getattr(args, "zh_en", False) else ["zh"]
    path = pipe.generate_questions_only(languages=langs)
    print(f"\n✅ Stage 3-only 生成完成: {path}")


def cmd_answer(args):
    pipe = BenchmarkPipeline(_make_config(args))
    path = pipe.run_model(args.model, limit=args.limit)
    print(f"\n✅ 作答完成: {path}")


def cmd_grade(args):
    pipe = BenchmarkPipeline(_make_config(args))
    report = pipe.grade(args.model)
    print("\n===== 评分报告 =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(prog="med-ethics-bench", description="医学伦理心理学 frontier-hard benchmark 工具")
    _add_common_flags(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("estimate", help="预估题量")
    _add_common_flags(pe)
    pe.set_defaults(func=cmd_estimate)

    pb = sub.add_parser("build", help="检索 + 生成题目，产出 benchmark")
    _add_common_flags(pb)
    pb.add_argument("--zh-en", action="store_true", help="同时生成中英文题目")
    pb.set_defaults(func=cmd_build)

    pr = sub.add_parser("repack", help="从已有 questions.jsonl 重建 benchmark.jsonl（断点恢复，不调用 LLM）")
    _add_common_flags(pr)
    pr.set_defaults(func=cmd_repack)

    pg3 = sub.add_parser("generate", help="只从 Stage 3 开始，读取已有 documents.jsonl 生成题目")
    _add_common_flags(pg3)
    pg3.add_argument("--zh-en", action="store_true", help="同时生成中英文题目")
    pg3.set_defaults(func=cmd_generate)

    pa = sub.add_parser("answer", help="用指定模型作答 benchmark")
    _add_common_flags(pa)
    pa.add_argument("--model", required=True)
    pa.add_argument("--limit", type=int, default=None)
    pa.set_defaults(func=cmd_answer)

    pg = sub.add_parser("grade", help="对已作答的答案进行评分")
    _add_common_flags(pg)
    pg.add_argument("--model", required=True)
    pg.set_defaults(func=cmd_grade)

    args = p.parse_args()
    if not os.getenv("POE_API_KEY"):
        print("⚠️  环境变量 POE_API_KEY 未设置：\n  export POE_API_KEY=你的_poe_key", file=sys.stderr)
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
