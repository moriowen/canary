"""
Average the monitors' log-odds per (run, k).

    canary ensemble swe [--members gbm gru text]

Rows where a member has no prediction (the text model only scores some k)
average whatever members are present.
"""

import numpy as np
import pandas as pd

from canary.data import RESULTS


def add_args(p):
    p.add_argument("tag")
    p.add_argument("--members", nargs="*", default=["gbm", "gru", "text"])
    p.add_argument("--name", default="ensemble", help="output prediction file name")


def run(args):
    d = RESULTS / args.tag / "preds"
    parts = []
    for m in args.members:
        f = d / f"{m}.parquet"
        if f.exists():
            x = pd.read_parquet(f)
            p = x.p.clip(1e-5, 1 - 1e-5)
            parts.append(x[["run_id", "k", "split"]].assign(z=np.log(p / (1 - p)), member=m))
            print("member", m, len(x))
    z = pd.concat(parts).groupby(["run_id", "k", "split"], as_index=False).z.mean()
    z["p"] = 1 / (1 + np.exp(-z.z))
    z.drop(columns="z").to_parquet(d / f"{args.name}.parquet", index=False)
    print(f"wrote {d / f'{args.name}.parquet'}: {len(z):,} rows")
