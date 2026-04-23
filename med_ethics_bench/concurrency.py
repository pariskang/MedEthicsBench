"""
concurrency.py — 通用并发执行器

设计取舍：
- 用线程池（IO 密集，GIL 对 HTTP 请求不是瓶颈）；
- 每个 job 的异常被隔离，不会拖垮整批；
- 结果通过 yield 以 **完成顺序**（as_completed）返回，上层可以边收边写盘；
- 所有对文件/计数器的写入都交给 **主线程**，避免锁。
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Tuple, TypeVar

from tqdm import tqdm

log = logging.getLogger(__name__)

T = TypeVar("T")   # 任务输入类型
R = TypeVar("R")   # 任务输出类型


def run_parallel(
    jobs: Iterable[T],
    worker: Callable[[T], R],
    *,
    max_workers: int = 4,
    desc: str = "running",
    total: int | None = None,
) -> Iterator[Tuple[T, R | None, BaseException | None]]:
    """
    以最多 max_workers 个线程并发执行 worker(job)。

    对每个任务产出 (job, result, err) 三元组：
      - 成功：err is None，result 为返回值
      - 失败：result is None，err 为异常实例
    上层负责 if err is None: write(result)；失败直接 log。
    """
    jobs_list = list(jobs)
    total = total or len(jobs_list)
    if max_workers <= 1 or total <= 1:
        # 单线程退化，方便调试与降级
        for job in tqdm(jobs_list, total=total, desc=desc):
            try:
                yield job, worker(job), None
            except BaseException as e:  # noqa: BLE001
                yield job, None, e
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_job = {ex.submit(worker, j): j for j in jobs_list}
        for fut in tqdm(as_completed(future_to_job),
                        total=total, desc=desc):
            job = future_to_job[fut]
            try:
                yield job, fut.result(), None
            except BaseException as e:  # noqa: BLE001
                yield job, None, e
