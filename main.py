"""
main.py — 命令行入口（frontier-hard 版）

支持两种写法：
  python main.py --output-dir ./data generate
  python main.py generate --output-dir ./data

示例：
  python main.py build --zh-en
  python main.py generate --output-dir ./data --log-level INFO
  python main.py answer --model GPT-5.4 --limit 20
  python main.py grade --model GPT-5.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from med_ethics_bench import BenchmarkPipeline, Config


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir")
    # --- 后端选择 ---
    parser.add_argument("--backend", choices=["poe", "azure"],
                        help="LLM 后端: poe（默认）或 azure")
    # --- Poe ---
    parser.add_argument("--api-key", help="Poe API Key（也可用 POE_API_KEY 环境变量）")
    # --- Azure ---
    parser.add_argument("--azure-api-key",
                        help="Azure OpenAI API Key（也可用 AZURE_OPENAI_API_KEY 环境变量）")
    parser.add_argument("--azure-base-url",
                        help="Azure OpenAI base_url，如 https://<resource>.openai.azure.com/openai/v1")
    # --- 模型 ---
    parser.add_argument("--retrieval-model")
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model")
    # --- 规模 ---
    parser.add_argument("--global-theory-per-topic", type=int)
    parser.add_argument("--country-theory-per-topic", type=int)
    parser.add_argument("--cases-per-country-topic", type=int)
    parser.add_argument("--single-country-per-cell", type=int)
    parser.add_argument("--comparison-per-cell", type=int)
    parser.add_argument("--universal-per-cell", type=int)
    # --- 并发 & 稳健性 ---
    parser.add_argument("--max-concurrency", type=int,
                        help="并发线程数（默认 6，Azure 建议 4–8）")
    parser.add_argument("--max-retries", type=int,
                        help="单次 API 调用最大重试次数（默认 6）")
    parser.add_argument("--request-timeout", type=float,
                        help="单次 HTTP 请求超时秒数（默认 180）")
    # --- 其他 ---
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续做")
    parser.add_argument("--log-level", default="INFO")


def _make_config(args: argparse.Namespace) -> Config:
    c = Config()
    # Backend & credentials
    if getattr(args, "backend", None):
        c.backend = args.backend
    if getattr(args, "api_key", None):
        c.api_key = args.api_key
    if getattr(args, "azure_api_key", None):
        c.azure_api_key = args.azure_api_key
    if getattr(args, "azure_base_url", None):
        c.azure_base_url = args.azure_base_url
    # Models
    for name in ("retrieval_model", "generator_model", "judge_model"):
        val = getattr(args, name, None)
        if val:
            setattr(c, name, val)
    # Scale
    for name in ("global_theory_per_topic", "country_theory_per_topic",
                 "cases_per_country_topic", "single_country_per_cell",
                 "comparison_per_cell", "universal_per_cell"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(c, name, val)
    # Concurrency & robustness
    if getattr(args, "max_concurrency", None) is not None:
        c.max_concurrency = args.max_concurrency
    if getattr(args, "max_retries", None) is not None:
        c.max_retries = args.max_retries
    if getattr(args, "request_timeout", None) is not None:
        c.request_timeout = args.request_timeout
    # Misc
    if getattr(args, "output_dir", None):
        c.output_dir = args.output_dir
    if getattr(args, "no_resume", False):
        c.resume = False
    if getattr(args, "log_level", None):
        c.log_level = args.log_level
    return c


def cmd_estimate(args: argparse.Namespace) -> None:
    cfg = _make_config(args)
    print(json.dumps(cfg.estimate_total_questions(), indent=2, ensure_ascii=False))


def cmd_build(args: argparse.Namespace) -> None:
    pipe = BenchmarkPipeline(_make_config(args))
    langs = ["zh", "en"] if getattr(args, "zh_en", False) else ["zh"]
    path = pipe.build_benchmark(languages=langs)
    print(f"\n✅ Benchmark 构建完成: {path}")


def cmd_repack(args: argparse.Namespace) -> None:
    """从已有 questions.jsonl 重建 benchmark.jsonl，不调用任何 LLM。"""
    pipe = BenchmarkPipeline(_make_config(args))
    path = pipe.repack_from_questions()
    print(f"\n✅ repack 完成: {path}")


def cmd_generate(args: argparse.Namespace) -> None:
    pipe = BenchmarkPipeline(_make_config(args))
    langs = ["zh", "en"] if getattr(args, "zh_en", False) else ["zh"]
    path = pipe.generate_questions_only(languages=langs)
    print(f"\n✅ Stage 3-only 生成完成: {path}")


def cmd_answer(args: argparse.Namespace) -> None:
    pipe = BenchmarkPipeline(_make_config(args))
    path = pipe.run_model(args.model, limit=args.limit)
    print(f"\n✅ 作答完成: {path}")


def cmd_grade(args: argparse.Namespace) -> None:
    pipe = BenchmarkPipeline(_make_config(args))
    report = pipe.grade(args.model)
    print("\n===== 评分报告 =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common)

    p = argparse.ArgumentParser(
        prog="med-ethics-bench",
        description="医学伦理心理学 frontier-hard benchmark 工具",
        parents=[common],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("estimate", help="预估题量", parents=[common])
    pe.set_defaults(func=cmd_estimate)

    pb = sub.add_parser("build", help="检索 + 生成题目，产出 benchmark", parents=[common])
    pb.add_argument("--zh-en", action="store_true", help="同时生成中英文题目")
    pb.set_defaults(func=cmd_build)

    pr = sub.add_parser(
        "repack",
        help="从已有 questions.jsonl 重建 benchmark.jsonl（断点恢复后快速打包，不调用 LLM）",
        parents=[common],
    )
    pr.set_defaults(func=cmd_repack)

    pg3 = sub.add_parser(
        "generate",
        help="只从 Stage 3 开始，读取已有 documents.jsonl 生成题目",
        parents=[common],
    )
    pg3.add_argument("--zh-en", action="store_true", help="同时生成中英文题目")
    pg3.set_defaults(func=cmd_generate)

    pa = sub.add_parser("answer", help="用指定模型作答 benchmark", parents=[common])
    pa.add_argument("--model", required=True, help="被测模型名，如 GPT-5.4 / Claude-Sonnet-4.6")
    pa.add_argument("--limit", type=int, default=None)
    pa.set_defaults(func=cmd_answer)

    pg = sub.add_parser("grade", help="对已作答的答案进行评分", parents=[common])
    pg.add_argument("--model", required=True, help="要评分的被测模型名")
    pg.set_defaults(func=cmd_grade)

    args = p.parse_args()
    if not os.getenv("POE_API_KEY"):
        print("⚠️  环境变量 POE_API_KEY 未设置：\n  export POE_API_KEY=你的_poe_key", file=sys.stderr)
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
