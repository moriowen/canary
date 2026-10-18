"""
Which failure modes does the monitor catch, and how early?

    canary catch swe [--model gbm] [--budget 0.05]

Joins the taxonomy (test-split quiet failures) with the monitor's kill decisions
at the threshold that loses at most `budget` of successes, and profiles each
cluster's behaviour from end-of-run features. Writes results/<tag>/catch.csv.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from canary.commands.evaluate import POLICY_SCHEDULE
from canary.data import REPORTS, RESULTS, processed
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

    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    t = tab.drop("all quiet failures").sort_values("share")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    ax.barh(t.index, t.share * 100, color="#cbd5e1", label="share of quiet failures (%)")
    ax.barh(t.index, t.share * t.caught * 100, color="#2563eb", label="caught by the monitor (%)")
    for i, (sh, c) in enumerate(zip(t.share, t.caught)):
        ax.text(sh * 100 + 0.3, i, f"{c:.0%} caught", va="center", fontsize=8)
    ax.set_xlabel("% of test-split quiet failures")
    ax.set_title(f"Failure modes and how many the monitor kills early (≤{args.budget:.0%} successes lost)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / f"{args.tag}_taxonomy_catch.png", dpi=160)
    for f in ("catch.csv", "catch_stage.csv"):
        (REPORTS / args.tag).mkdir(parents=True, exist_ok=True)
        (REPORTS / args.tag / f).write_text((out / f).read_text())
    (REPORTS / args.tag / "clusters.json").write_text((tx / "clusters.json").read_text())
