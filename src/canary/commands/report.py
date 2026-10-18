"""
Figures and markdown tables from results/. Run after evaluate.

    canary report

Writes reports/figures/*.png, reports/tables.md, and copies each tag's small
summaries (summary.json, metrics.csv, importance) into reports/<tag>/.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import shutil

from canary.data import REPORTS, RESULTS, processed

NAMES = {
    "gbm": "LightGBM (run features)", "gru": "GRU monitor", "text": "ModernBERT (transcript)",
    "logreg": "Logistic regression", "gbm_issue": "Issue text only", "oracle_task": "Oracle task difficulty",
    "rule_streak": "Rule: N consecutive errors", "rule_budget": "Rule: hard turn cap",
    "ensemble": "Ensemble (GBM + GRU + text)", "gbm_rank": "LightGBM within-task ranker",
    "gbm_all": "LightGBM + issue features",
}
COLORS = {
    "gbm": "#2563eb", "gru": "#16a34a", "text": "#9333ea", "logreg": "#64748b", "ensemble": "#dc2626",
    "gbm_issue": "#f59e0b", "gbm_rank": "#db2777", "gbm_all": "#93c5fd", "oracle_task": "#a8a29e", "rule_streak": "#0891b2", "rule_budget": "#111827",
}
ORDER = ["ensemble", "gbm", "gru", "text", "gbm_rank", "gbm_all", "logreg", "gbm_issue", "oracle_task"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def fig_auroc(tag, out):
    m = pd.read_csv(RESULTS / tag / "metrics.csv")
    m = m[m.k <= 30]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, title in [(axes[0], "auroc", "Pooled AUROC (all test runs)"),
                           (axes[1], "within_auc", "Within-task AUROC (same issue, mixed outcomes)")]:
        for name in ORDER:
            g = m[m.model == name]
            if len(g):
                ax.plot(g.k, g[col], marker="o", ms=3, lw=1.8 if name in ("gbm", "ensemble") else 1.2,
                        color=COLORS[name], label=NAMES[name])
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("turns observed (k)")
        _style(ax)
    axes[0].set_ylabel("AUROC, P(success)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / f"{tag}_auroc.png", dpi=160)
    plt.close(fig)


def fig_policy(tag, out, metric="saved_ctx"):
    c = pd.read_csv(RESULTS / tag / "policy.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name in ORDER + ["rule_streak", "rule_budget"]:
        g = c[c.model == name].sort_values("lost")
        if not len(g) or name == "oracle_task":
            continue
        # Pareto frontier: best saving at each loss level.
        g = g.assign(best=g[metric].cummax())
        ax.plot(g.lost * 100, g.best * 100, lw=1.8 if name in ("gbm", "ensemble") else 1.2,
                ls="--" if name.startswith("rule") else "-", color=COLORS[name], label=NAMES[name])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, None)
    ax.set_xlabel("successful runs killed (%)")
    ax.set_ylabel("compute saved (% of context tokens)" if metric == "saved_ctx" else "turns saved (%)")
    ax.set_title("Abort policy: compute saved vs. successes lost", fontsize=10)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_policy_{metric}.png", dpi=160)
    plt.close(fig)


def fig_importance(tag, out, n=15):
    f = RESULTS / tag / "importance_gbm.csv"
    if not f.exists():
        return
    imp = pd.read_csv(f).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(imp.feature, imp.gain / imp.gain.sum() * 100, color=COLORS["gbm"])
    ax.set_xlabel("share of gain among top features (%)")
    ax.set_title("What the LightGBM monitor relies on", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_importance.png", dpi=160)
    plt.close(fig)


def table_summary(tag, models=None):
    s = json.loads((RESULTS / tag / "summary.json").read_text())
    models = models or [m for m in ORDER + ["rule_streak", "rule_budget"] if m in s]
    hdr = ("| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 "
           "| Compute saved @ 2% lost | @ 5% lost | @ 10% lost |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = [hdr]
    fmt = lambda v: "n/a" if v is None or v != v else f"{v:.3f}"
    pct = lambda v: "n/a" if v is None or v != v else f"{v * 100:.1f}%"
    for m in models:
        r = s[m]
        rows.append(f"| {NAMES.get(m, m)} | {fmt(r.get('auroc@0'))} | {fmt(r.get('auroc@5'))} | "
                    f"{fmt(r.get('auroc@10'))} | {fmt(r.get('within@5'))} | {fmt(r.get('within@10'))} | "
                    f"{pct(r.get('saved_ctx@2%'))} | {pct(r.get('saved_ctx@5%'))} | {pct(r.get('saved_ctx@10%'))} |")
    return "\n".join(rows)


def add_args(p):
    p.add_argument("--tags", nargs="*", default=None)


def run(args):
    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tags = args.tags or [p.name for p in RESULTS.iterdir() if (p / "summary.json").exists()]
    md = []
    for tag in sorted(tags):
        fig_auroc(tag, out)
        fig_policy(tag, out)
        fig_policy(tag, out, "saved_turns")
        fig_importance(tag, out)
        md.append(f"## {tag}\n\n{table_summary(tag)}\n")
        (REPORTS / tag).mkdir(parents=True, exist_ok=True)
        for f in ("summary.json", "metrics.csv", "importance_gbm.csv", "coef_logreg.csv", "tabular.json"):
            if (RESULTS / tag / f).exists():
                shutil.copy(RESULTS / tag / f, REPORTS / tag / f)
        print("figures for", tag)
    (REPORTS / "tables.md").write_text("\n".join(md))
    print((REPORTS / "tables.md").read_text())
