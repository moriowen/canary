"""
Download a scaffold's dataset and turn every run into model-ready tables.

    canary build swe        # nebius/SWE-agent-trajectories      (80k runs)
    canary build oh         # nebius/SWE-rebench-openhands-...   (67k runs)

Writes data/processed/<scaffold>/:
    runs.parquet      one row per run: ids, label, length, issue, outcome detail
    turns/*.parquet   one row per turn (clipped text + signals)
    prefix.parquet    one row per (run, checkpoint k) with prefix features
    full.parquet      end-of-run features (taxonomy input)
    steps/*.parquet   per-turn vectors for the sequence model
    splits.parquet    instance_id -> train/val/test, task type
"""

import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from canary.data import processed
from canary.features import prefix_rows, step_matrix
from canary.splits import make_splits
from canary.turns import parse_openhands, parse_sweagent

REPOS = {
    "swe": "nebius/SWE-agent-trajectories",
    "oh": "nebius/SWE-rebench-openhands-trajectories",
}
SWE_COLS = ["instance_id", "model_name", "target", "trajectory", "exit_status",
            "generated_patch", "eval_logs"]
OH_COLS = ["trajectory_id", "instance_id", "repo", "trajectory", "model_patch",
           "exit_status", "resolved"]


def _s(x):
    return x if isinstance(x, str) else ""


def _patch_stats(p):
    p = _s(p)
    files = re.findall(r"^diff --git a/(\S+)", p, re.M)
    added = len(re.findall(r"^\+(?!\+\+)", p, re.M))
    removed = len(re.findall(r"^-(?!--)", p, re.M))
    return len(files), added + removed


def _swe_outcome(patch, log, target):
    if target:
        return "resolved"
    if not patch.strip():
        return "no_patch"
    if not log:
        return "no_eval"
    if "error: patch failed" in log or "Patch Apply Failed" in log:
        return "apply_fail"
    return "tests_fail"


def _work(job):
    scaffold, run_id, row = job
    if scaffold == "swe":
        issue, turns = parse_sweagent(row["trajectory"])
        model = row["model_name"]
        target = bool(row["target"])
        patch = _s(row["generated_patch"])
        log = _s(row["eval_logs"])
        outcome = _swe_outcome(patch, log, target)
        failed_names = re.findall(r"^FAILED (\S+)", log, re.M)
        fails = len(failed_names)
        passes = len(re.findall(r"^PASSED ", log, re.M))
        repo = row["instance_id"].rsplit("-", 1)[0]
    else:
        issue, turns = parse_openhands(row["trajectory"])
        model = "openhands"
        target = bool(row["resolved"])
        patch = _s(row.get("model_patch"))
        outcome = "resolved" if target else ("no_patch" if not patch.strip() else "tests_fail")
        fails = passes = -1
        failed_names = []
        repo = row["repo"]
    pf, pl = _patch_stats(patch)
    rows, full = prefix_rows(turns, issue, full=True)
    meta = {
        "run_id": run_id, "scaffold": scaffold, "instance_id": row["instance_id"],
        "repo": repo, "model": model, "target": target, "exit_status": str(row["exit_status"]),
        "outcome": outcome, "n_turns": len(turns), "issue": issue[:8000],
        "patch_files": pf, "patch_lines": pl, "eval_failed": fails, "eval_passed": passes,
        "patch": patch[:4000], "failed_tests": "\n".join(failed_names[:15]),
        "ctx_chars": sum(t["out_chars"] + t["thought_chars"] + len(t["cmd"]) for t in turns),
    }
    for t in turns:
        t["run_id"] = run_id
    for r in rows:
        r["run_id"] = run_id
    full["run_id"] = run_id
    steps = step_matrix(turns)
    return meta, turns, rows, full, steps


def _chunks(scaffold, files):
    """Yield (part_name, list of (scaffold, run_id, row)) one row group at a time."""
    if scaffold == "swe":
        for si, f in enumerate(files):
            df = pd.read_parquet(f, columns=SWE_COLS)
            yield f"{si:02d}", [(scaffold, f"swe-{si:02d}-{i:05d}", r) for i, r in
                                enumerate(df.to_dict("records"))]
    else:
        for f in files:
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                cols = [c for c in OH_COLS if c in pf.schema_arrow.names]
                df = pf.read_row_group(g, columns=cols).to_pandas()
                yield f"{g:02d}", [(scaffold, r["trajectory_id"], r) for r in df.to_dict("records")]


def add_args(p):
    p.add_argument("scaffold", choices=sorted(REPOS))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0, help="rows per part (smoke test)")
    p.add_argument("--source", nargs="*", help="local parquet files instead of downloading")


def run(args):
    out = processed(args.scaffold)
    for sub in ("turns", "steps"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if args.source:
        files = args.source
    else:
        local = snapshot_download(REPOS[args.scaffold], repo_type="dataset",
                                  allow_patterns=["*.parquet"])
        files = sorted(str(p) for p in Path(local).rglob("*.parquet"))
    print(f"{len(files)} parquet files from {REPOS[args.scaffold]}")

    metas, prefix, fulls = [], [], []
    with Pool(args.workers) as pool:
        for part, jobs in _chunks(args.scaffold, files):
            if args.limit:
                jobs = jobs[: args.limit]
            res = pool.map(_work, jobs, chunksize=16)
            turns = [t for r in res for t in r[1]]
            pd.DataFrame(turns).to_parquet(out / "turns" / f"part-{part}.parquet", index=False)
            pd.DataFrame({
                "run_id": [r[0]["run_id"] for r in res],
                "n": [len(r[4]) for r in res],
                "steps": [r[4].ravel() for r in res],
            }).to_parquet(out / "steps" / f"part-{part}.parquet", index=False)
            metas += [r[0] for r in res]
            prefix += [x for r in res for x in r[2]]
            fulls += [r[3] for r in res]
            print(f"  part {part}: {len(res):,} runs, {len(turns):,} turns", flush=True)

    runs = pd.DataFrame(metas)
    runs.to_parquet(out / "runs.parquet", index=False)
    pd.DataFrame(prefix).to_parquet(out / "prefix.parquet", index=False)
    pd.DataFrame(fulls).to_parquet(out / "full.parquet", index=False)
    splits = make_splits(runs)
    splits.to_parquet(out / "splits.parquet", index=False)

    print(f"\n{len(runs):,} runs, solved {runs.target.mean():.1%}, "
          f"median turns {runs.n_turns.median():.0f}, prefix rows {len(prefix):,}")
    print(splits.groupby(["split", "task_type"]).size().unstack(fill_value=0))
    print(runs.outcome.value_counts())
