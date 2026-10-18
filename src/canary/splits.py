"""
Task-grouped splits.

Every attempt at the same GitHub issue lands in the same split, otherwise a
model can memorise which issues are hopeless. Tasks are stratified by type so
the mixed-outcome tasks (the within-task test) are spread evenly.
"""

import numpy as np
import pandas as pd

FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}
SEED = 7641


def task_types(runs):
    g = runs.groupby("instance_id").target.agg(["any", "all", "size", "mean"])
    t = np.where(~g["any"], "never", np.where(g["all"], "always", "mixed"))
    return pd.DataFrame({"instance_id": g.index, "task_type": t,
                         "attempts": g["size"].values, "solve_rate": g["mean"].values})


def make_splits(runs, seed=SEED):
    tasks = task_types(runs)
    rng = np.random.default_rng(seed)
    out = []
    for _, grp in tasks.groupby("task_type"):
        idx = rng.permutation(len(grp))
        cuts = np.cumsum([int(round(f * len(grp))) for f in FRACS.values()])[:-1]
        lab = np.empty(len(grp), dtype=object)
        for name, part in zip(FRACS, np.split(idx, cuts)):
            lab[part] = name
        out.append(grp.assign(split=lab))
    return pd.concat(out, ignore_index=True)
