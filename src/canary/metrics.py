"""
Scoring predictions. Every prediction table has columns run_id, k, p, where
p = P(run succeeds | first k turns).

Two views of the same numbers:
  pooled       AUROC over all alive runs at k. Mixes "this task is hard" with
               "this run is going badly", so it flatters any model.
  within-task  AUROC over (solved, failed) pairs of runs on the SAME issue.
               Task difficulty cancels; only run-level signal scores.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _auc_counts(y, p):
    """(AUC * n_pos * n_neg, n_pos * n_neg) for one group, ties count half."""
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return 0.0, 0
    r = pd.Series(p).rank(method="average").to_numpy()
    u = r[y].sum() - npos * (npos + 1) / 2
    return float(u), int(npos * nneg)


def within_task_auc(df):
    """Pair-weighted AUC over runs that share an instance_id."""
    num = den = 0
    for _, g in df.groupby("instance_id"):
        a, b = _auc_counts(g.target.to_numpy(), g.p.to_numpy())
        num += a
        den += b
    return num / den if den else np.nan, den


def score(preds, runs, splits, split="test"):
    """Per-checkpoint metrics on one split."""
    meta = runs[["run_id", "instance_id", "target"]].merge(
        splits[["instance_id", "split", "task_type"]], on="instance_id")
    df = preds.drop(columns=["split"], errors="ignore").merge(meta, on="run_id")
    df = df[df.split == split]
    rows = []
    for k, g in df.groupby("k"):
        y = g.target.to_numpy()
        if y.all() or not y.any():
            continue
        wt, pairs = within_task_auc(g[g.task_type == "mixed"])
        rows.append({
            "k": k, "n": len(g), "solve_rate": y.mean(),
            "auroc": roc_auc_score(y, g.p), "auprc": average_precision_score(y, g.p),
            "brier": brier_score_loss(y, g.p.clip(0, 1)),
            "within_auc": wt, "pairs": pairs,
        })
    return pd.DataFrame(rows)
