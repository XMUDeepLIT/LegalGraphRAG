"""Multi-GPU worker for the direct-model protocols."""
import json
import multiprocessing
import os
from typing import Any, Dict, List

from tqdm import tqdm


def split_chunks(cases: List[Dict[str, Any]], workers: int) -> List[List[Dict[str, Any]]]:
    workers = max(1, min(workers, len(cases)))
    size, remainder = divmod(len(cases), workers)
    chunks = []
    start = 0
    for index in range(workers):
        width = size + (1 if index < remainder else 0)
        chunks.append(cases[start : start + width])
        start += width
    return [chunk for chunk in chunks if chunk]


def direct_worker(payload: Dict[str, Any]):
    from ..catalog import get_model
    from .session import JudgmentSession

    spec = get_model(payload["model"])
    session = JudgmentSession(spec, payload["device"], model_name=payload.get("model_name"))
    results = []
    try:
        for case in tqdm(payload["cases"], desc=payload["device"]):
            results.append(session.judge_case(case))
    finally:
        session.close()
    with open(payload["output_file"], "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def run_direct(args, spec, cases, output_path):
    devices = ["cpu"] if spec.backend != "local" else (args.devices or ["cuda:0"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks = split_chunks(cases, len(devices))
    part_files = [f"{output_path}.part{index}" for index in range(len(chunks))]
    if len(chunks) == 1:
        direct_worker(
            {
                "cases": chunks[0],
                "model": spec.key,
                "device": devices[0],
                "model_name": args.model_name,
                "output_file": part_files[0],
            }
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        processes = []
        for index, chunk in enumerate(chunks):
            process = ctx.Process(
                target=direct_worker,
                args=(
                    {
                        "cases": chunk,
                        "model": spec.key,
                        "device": devices[index],
                        "model_name": args.model_name,
                        "output_file": part_files[index],
                    },
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Baseline worker exited with status {process.exitcode}")
    merged = []
    for part in part_files:
        with open(part, "r", encoding="utf-8") as handle:
            merged.extend(json.load(handle))
        os.remove(part)
    return merged
