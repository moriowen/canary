"""
Pull ONLY the small label columns from all parquet shards.

Parquet is columnar, so selecting 4 narrow columns avoids downloading the
~5.6 GB `trajectory` column. This verifies the headline numbers cheaply
before committing to a full download.

Output: data/interim/metadata.parquet  (one row per trajectory, no text)
"""

import pandas as pd

from canary.data import METADATA, shards

COLS = ["instance_id", "model_name", "target", "exit_status"]


def add_args(p):
    p.add_argument("--out", default=str(METADATA))


def run(args):
    urls = shards()
    print(f"Found {len(urls)} shards. Reading columns {COLS} ...")
    df = pd.concat([pd.read_parquet(u, columns=COLS) for u in urls], ignore_index=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(df):,}\n")

    print("=" * 62)
    print(f"TOTAL TRAJECTORIES: {len(df):,}")
    print(f"UNIQUE TASKS (instance_id): {df.instance_id.nunique():,}")

    print("\n--- Outcome (target = issue actually resolved) ---")
    vc = df.target.value_counts()
    for k, v in vc.items():
        print(f"  {str(k):<6} {v:>7,}  ({v / len(df):6.2%})")

    print("\n--- Backbone model ---")
    for k, v in df.model_name.value_counts().items():
        print(f"  {k:<24} {v:>7,}")

    print("\n--- exit_status x outcome ---")
    ct = pd.crosstab(df.exit_status, df.target).rename(
        columns={False: "failed", True: "solved"}
    )
    for c in ("failed", "solved"):
        if c not in ct:
            ct[c] = 0
    ct["total"] = ct.failed + ct.solved
    ct["solve_rate"] = (ct.solved / ct.total).map("{:.2%}".format)
    print(ct.sort_values("total", ascending=False).to_string())

    clean = df[df.exit_status.str.startswith("submitted")]
    clean_fail = clean[~clean.target]
    print("\n" + "=" * 62)
    print("QUIET FAILURES")
    print(f"  Exited cleanly (status starts 'submitted'): {len(clean):,}")
    print(f"  ...of which FAILED anyway:                  {len(clean_fail):,}")
    print(f"  = {len(clean_fail) / len(df):.1%} of all trajectories:")
    print("    finished normally, submitted a patch, still did not fix the bug.")
    print("    exit_status explains none of these; the failure taxonomy does.")
    print("=" * 62)

    # Within-task confound: most tasks are never solved, so a model can score
    # well by recognising hopeless tasks rather than runs going badly.
    per_task = df.groupby("instance_id").target.agg(["any", "all"])
    mixed = per_task["any"] & ~per_task["all"]
    print("\n--- Task difficulty ---")
    print(f"  never solved : {(~per_task['any']).sum():,} tasks")
    print(f"  always solved: {per_task['all'].sum():,} tasks")
    print(f"  mixed outcome: {mixed.sum():,} tasks, "
          f"{df.instance_id.isin(per_task.index[mixed]).sum():,} runs")

    print("\n--- Task overlap across model scales ---")
    sets = {m: set(g.instance_id) for m, g in df.groupby("model_name")}
    names = sorted(sets, key=lambda m: -len(sets[m]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a} ∩ {b}: {len(sets[a] & sets[b]):,} shared tasks")
