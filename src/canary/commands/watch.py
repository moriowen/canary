"""
Replay a held-out run turn by turn and show what the monitor thinks.

    canary watch swe                       # a random test run
    canary watch swe --run-id swe-03-01234
    canary watch swe --failed --threshold 0.08

Uses the LightGBM monitor from `canary tabular`. The threshold is the kill line;
results/<tag>/summary.json lists the threshold that loses at most 5% of
successes on the test split (`kill_threshold@5%`).
"""

import json
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from canary.data import RESULTS, processed
from canary.features import prefix_rows
from canary.models.tabular import MODEL_B

BAR = 30


def add_args(p):
    p.add_argument("scaffold")
    p.add_argument("--run-id")
    p.add_argument("--failed", action="store_true", help="pick a run that failed")
    p.add_argument("--solved", action="store_true", help="pick a run that succeeded")
    p.add_argument("--threshold", type=float, help="kill when P(success) drops below this")
    p.add_argument("--seed", type=int)


def run(args):
    d = processed(args.scaffold)
    out = RESULTS / args.scaffold
    model = lgb.Booster(model_file=str(out / "gbm.txt"))
    cols = model.feature_name()
    runs = pd.read_parquet(d / "runs.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    test = runs.merge(splits, on="instance_id").query("split == 'test'")
    if args.run_id:
        r = runs[runs.run_id == args.run_id].iloc[0]
    else:
        pool = test[test.n_turns >= 6]
        if args.failed:
            pool = pool[~pool.target]
        if args.solved:
            pool = pool[pool.target]
        r = pool.sample(1, random_state=args.seed if args.seed is not None else random.randrange(10**6)).iloc[0]
    tau = args.threshold
    if tau is None:
        s = json.loads((out / "summary.json").read_text())
        tau = s.get("gbm", {}).get("kill_threshold@5%", 0.05)

    turns = pd.read_parquet(d / "turns", filters=[("run_id", "==", r.run_id)]).sort_values("t")
    T = turns.to_dict("records")
    rows = prefix_rows(T, r.issue, checkpoints=list(range(len(T))))
    X = pd.DataFrame(rows)
    X["model_b"] = next((np.log(v) for s, v in MODEL_B.items() if r.model.endswith(s)), np.nan)
    p = model.predict(X[cols])

    print(f"\n{r.run_id}  {r.instance_id}  [{r.model}]  {r.n_turns} turns")
    print(f"issue: {' '.join(r.issue.split())[:150]}...")
    print(f"kill line: P(success) < {tau:.3f}\n")
    killed = None
    for k, pk in enumerate(p):
        if k == 0:
            act = "(before any turn)"
        else:
            x = T[k - 1]
            err = " ERR" if any(x[f"err_{e}"] for e in ("traceback", "edit", "notfound", "exception", "exit")) else ""
            act = f"{x['action']:<7}{err:<4} {' '.join(x['cmd'].split())[:52]}"
        bar = "#" * int(round(pk * BAR * 2.5)) if pk < 0.4 else "#" * BAR
        mark = ""
        if killed is None and k > 0 and pk < tau:
            killed = k
            mark = "  <-- KILL"
        print(f"  k={k:>3}  p={pk:5.3f}  {bar[:BAR]:<{BAR}}  {act}{mark}")
    verdict = "SOLVED" if r.target else "FAILED"
    print(f"\noutcome: {verdict}.", end=" ")
    if killed is None:
        print("The monitor let it run to the end.")
    else:
        saved = r.n_turns - killed
        print(f"The monitor would have killed it at turn {killed}, saving {saved} of {r.n_turns} turns"
              + (" -- a false kill." if r.target else "."))
