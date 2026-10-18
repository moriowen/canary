"""Univariate within-task AUROC of each feature at fixed k, per scaffold.

Which run-level signals mean the same thing on SWE-agent and OpenHands, and which
flip sign? Writes results/signal_flip.csv.
"""
import sys

import numpy as np
import pandas as pd

from canary.data import RESULTS, processed
from canary.metrics import within_task_auc

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rows = []
for s in ("swe", "oh"):
    d = processed(s)
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "instance_id", "target"])
    sp = pd.read_parquet(d / "splits.parquet")
    X = pd.read_parquet(d / "prefix.parquet").query("k == @K").merge(runs, on="run_id")
    X = X.merge(sp[["instance_id", "task_type"]], on="instance_id").query("task_type == 'mixed'")
    for c in X.columns:
        if c in ("run_id", "instance_id", "target", "task_type", "k") or c.startswith("issue_"):
            continue
        v = X[c].astype(float)
        if v.nunique() < 2:
            continue
        auc, _ = within_task_auc(X.assign(p=v.fillna(v.median()))[["instance_id", "target", "p"]])
        rows.append({"scaffold": s, "feature": c, "within_auc": auc})
df = pd.DataFrame(rows).pivot(index="feature", columns="scaffold", values="within_auc")
df["flip"] = (df.swe - 0.5) * (df.oh - 0.5) < 0
df["strength"] = (df.swe - 0.5).abs() + (df.oh - 0.5).abs()
df = df.sort_values("strength", ascending=False)
df.to_csv(RESULTS / f"signal_flip_k{K}.csv")
print(df.head(30).round(3).to_string())
