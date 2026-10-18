"""
Build a small, stratified working sample WITH the trajectory text.

The full `trajectory` column is ~5.6 GB. For development you do not need it.
This walks the shards one at a time, keeps a balanced slice from each, and
throws the rest away -- so peak memory stays around one shard.

Output: data/raw/sample.parquet
"""

import pandas as pd

from canary.data import SAMPLE, shards

SEED = 7641  # the existing data/raw/sample.parquet was drawn with this


def take(df, n, frac, seed):
    """Stratified draw: `frac` solved, remainder failed. frac=0 draws at random."""
    if not frac:
        return df.sample(min(n, len(df)), random_state=seed)
    want_ok = int(n * frac)
    ok = df[df.target]
    no = df[~df.target]
    ok = ok.sample(min(want_ok, len(ok)), random_state=seed)
    no = no.sample(min(n - len(ok), len(no)), random_state=seed)
    return pd.concat([ok, no])


def add_args(p):
    p.add_argument("--per-shard", type=int, default=500, help="rows kept per shard")
    p.add_argument("--solved-frac", type=float, default=0.35,
                   help="fraction of solved runs (real rate is 16.7%%)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=str(SAMPLE))


def run(args):
    urls = shards()
    print(f"{len(urls)} shards; taking ~{args.per_shard} rows from each.\n")

    parts = []
    for i, u in enumerate(urls, 1):
        print(f"  [{i:>2}/{len(urls)}] {u.split('/')[-1]} ...", end=" ", flush=True)
        shard = pd.read_parquet(u)
        parts.append(take(shard, args.per_shard, args.solved_frac, args.seed))
        del shard
        print("ok")

    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\nWrote {args.out}")
    print(f"  rows          : {len(df):,}")
    print(f"  unique tasks  : {df.instance_id.nunique():,}")
    print(f"  solved        : {df.target.sum():,} ({df.target.mean():.1%})")
    print(f"  steps/traj    : median {df.trajectory.map(len).median():.0f}")
    print(f"  columns       : {list(df.columns)}")
    if args.solved_frac:
        print("\nNOTE: successes are oversampled, so this sample is NOT")
        print("representative. Re-weight, or resample with --solved-frac 0,")
        print("before quoting any headline rate.")
