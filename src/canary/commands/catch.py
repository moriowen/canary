"""
Which failure modes does the monitor catch, and how early?

    canary catch swe [--model gbm] [--budget 0.05]

Joins the taxonomy (test-split quiet failures) with the monitor's kill decisions
at the threshold that loses at most `budget` of successes, and profiles each
cluster's behaviour from end-of-run features. Writes results/<tag>/catch.csv.
"""

import json

import pandas as pd

from canary.commands.evaluate import POLICY_SCHEDULE
from canary.data import RESULTS, processed
from canary.policy import threshold_kills

PROFILE = {"n_turns": "turns", "frac_err": "error rate", "n_test": "test runs",
           "n_src_edit": "source edits", "frac_lex_hedge": "hedging", "frac_repeat_cmd": "repeats",
           "repro_err": "reproduced bug"}


def add_args(p):
    p.add_argument("tag")
    p.add_argument("--model", default="gbm")
    p.add_argument("--budget", type=float, default=0.05)


def run(args):
    out = RESULTS / args.tag
    tx = out / "taxonomy"
    s = json.loads((out / "summary.json").read_text())
    tau = s[args.model][f"kill_threshold@{int(args.budget * 100)}%"]
    clusters = pd.read_parquet(tx / "clusters.parquet").query("split == 'test'")
    info = pd.DataFrame(json.loads((tx / "clusters.json").read_text()))[["cluster", "name"]]
    preds = pd.read_parquet(out / "preds" / f"{args.model}.parquet")
    preds = preds[preds.run_id.isin(clusters.run_id) & preds.k.isin(POLICY_SCHEDULE)]
    kills = threshold_kills(preds, tau)

    d = processed(args.tag)
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "n_turns"])
    full = pd.read_parquet(d / "full.parquet")
    df = clusters.merge(runs, on="run_id").merge(full.drop(columns=["k"]), on="run_id").merge(info, on="cluster")
    df["kill_k"] = df.run_id.map(kills)
    df["caught"] = df.kill_k.notna()
    df["saved"] = (df.n_turns - df.kill_k).where(df.caught, 0) / df.n_turns

    g = df.groupby("name")
    tab = pd.DataFrame({
        "runs": g.size(), "share": g.size() / len(df), "caught": g.caught.mean(),
        "median_kill_turn": g.kill_k.median(), "turns_saved": g.saved.mean(),
        **{v: g[k].mean() for k, v in PROFILE.items() if k in df},
    }).sort_values("share", ascending=False)
    tab.loc["all quiet failures"] = [len(df), 1.0, df.caught.mean(), df.kill_k.median(), df.saved.mean(),
                                     *[df[k].mean() for k in PROFILE if k in df]]
    tab.to_csv(out / "catch.csv")
    stage = df.groupby("stage").agg(runs=("run_id", "size"), caught=("caught", "mean"),
                                    turns_saved=("saved", "mean")).sort_values("runs", ascending=False)
    stage.to_csv(out / "catch_stage.csv")
    print(f"threshold {tau:.4f} ({args.model}, <= {args.budget:.0%} successes lost)\n")
    print(tab.round(3).to_string())
    print()
    print(stage.round(3).to_string())
