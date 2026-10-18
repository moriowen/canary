"""
Score every prediction file under results/<tag>/preds and the rule baselines.

    canary evaluate swe
    canary evaluate swe_to_oh --data oh

Writes results/<tag>/{metrics.csv, policy.csv, summary.json}.
"""

import json

import pandas as pd

from canary.data import RESULTS, processed
from canary.metrics import score
from canary.policy import Sim, at_budget, run_costs, sweep_budget, sweep_model, sweep_streak

POLICY_SCHEDULE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]
HEADLINE_K = [0, 1, 3, 5, 10, 20]


def add_args(p):
    p.add_argument("tag", help="results/<tag> to score, e.g. swe or swe_to_oh")
    p.add_argument("--data", help="scaffold whose runs the preds are about (default: tag prefix)")


def run(args):
    scaffold = args.data or args.tag.split("_")[0]  # e.g. swe_to_oh needs --data oh
    d = processed(scaffold)
    out = RESULTS / args.tag
    runs = pd.read_parquet(d / "runs.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    test_ids = set(runs.merge(splits, on="instance_id").query("split == 'test'").run_id)
    test_runs = runs[runs.run_id.isin(test_ids)]
    turns = pd.read_parquet(d / "turns", columns=None, filters=[("run_id", "in", list(test_ids))])
    sim = Sim(test_runs, run_costs(turns, test_runs))

    metrics, curves, summary = [], [], {}
    for f in sorted((out / "preds").glob("*.parquet")):
        name = f.stem
        preds = pd.read_parquet(f)
        m = score(preds, runs, splits, within=not name.startswith("oracle")).assign(model=name)
        metrics.append(m)
        tp = preds[preds.run_id.isin(test_ids) & preds.k.isin(POLICY_SCHEDULE)]
        c = sweep_model(sim, tp).assign(model=name)
        curves.append(c)
        summary[name] = {
            **{f"auroc@{k}": _get(m, k, "auroc") for k in HEADLINE_K},
            **{f"within@{k}": _get(m, k, "within_auc") for k in HEADLINE_K},
            **at_budget(c), **at_budget(c, metric="saved_turns"),
            **_thresholds(c),
        }
        print(f"{name:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in summary[name].items()
                                          if v == v and ("@5" in k or "@1%" in k or "auroc@5" in k)))

    for name, c in [("rule_streak", sweep_streak(sim, turns)), ("rule_budget", sweep_budget(sim))]:
        c = c.assign(model=name)
        curves.append(c)
        summary[name] = {**at_budget(c), **at_budget(c, metric="saved_turns")}
        print(f"{name:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in summary[name].items()))

    pd.concat(metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(curves).to_csv(out / "policy.csv", index=False)
    summary["_data"] = {
        "test_runs": len(test_runs), "test_solve_rate": float(test_runs.target.mean()),
        "test_tasks": int(test_runs.instance_id.nunique()),
        "median_turns": float(test_runs.n_turns.median()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/metrics.csv, policy.csv, summary.json")


def _thresholds(c):
    """Kill threshold on P(success) that saves the most context within each loss budget."""
    out = {}
    for b in (0.02, 0.05, 0.10):
        ok = c[c.lost <= b]
        out[f"kill_threshold@{int(b * 100)}%"] = float(ok.loc[ok.saved_ctx.idxmax(), "param"]) if len(ok) else 0.0
    return out


def _get(m, k, col):
    r = m[m.k == k]
    return float(r[col].iloc[0]) if len(r) else float("nan")
