"""
Tabular models over prefix features.

    canary tabular swe                    # in-domain: train/val/test on SWE-agent
    canary tabular swe --eval-on oh       # transfer: train on SWE-agent, test on OpenHands

Writes results/<train>[_to_<eval>]/preds/<model>.parquet (run_id, k, p, split)
and feature importances.
"""

import json
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from canary.data import RESULTS, processed

MODEL_B = {"8b": 8, "70b": 70, "405b": 405}
# Features whose meaning differs by scaffold (SWE-agent has no exit codes or think
# tool, OpenHands no glued commands) or that encode the backbone; dropped for transfer.
NON_PORTABLE = ["model_b", "n_malformed", "n_err_exit", "n_think", "frac_think"]

# Large leaves + task-balanced weights: ~22 attempts share each issue, so small leaves
# memorise per-task solve rates instead of learning how runs behave.
GBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=1000,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  metric="auc", verbose=-1, seed=7641)


def load(scaffold):
    d = processed(scaffold)
    runs = pd.read_parquet(d / "runs.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    prefix = pd.read_parquet(d / "prefix.parquet")
    runs["model_b"] = runs.model.map(
        lambda m: next((math.log(v) for s, v in MODEL_B.items() if m.endswith(s)), np.nan))
    meta = runs[["run_id", "instance_id", "target", "model_b"]].merge(
        splits[["instance_id", "split"]], on="instance_id")
    X = prefix.merge(meta, on="run_id")
    return runs, splits, X


def align(X, cols):
    """Replace each feature by its percentile among the same scaffold's train-split
    runs at the same checkpoint k.

    Unsupervised domain alignment: "more errors than most runs at this point" means
    the same thing on both scaffolds even when raw counts and run lengths don't.
    Uses only unlabeled features from the scaffold's own train split.
    """
    X = X.copy()
    cols = [c for c in cols if c != "k"]
    V = X[cols].to_numpy(np.float64)
    out = np.full_like(V, np.nan)
    is_tr = (X.split == "train").to_numpy()
    kk = X.k.to_numpy()
    for k in np.unique(kk):
        rows = np.where(kk == k)[0]
        ref = np.where(is_tr & (kk == k))[0]
        if len(ref) < 50:   # sparse late checkpoints: borrow the nearest earlier ones
            ref = np.where(is_tr & (kk <= k) & (kk >= k // 2))[0]
        for j in range(len(cols)):
            srt = np.sort(V[ref, j][~np.isnan(V[ref, j])])
            if len(srt) == 0:
                continue
            v = V[rows, j]
            pct = (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / (2 * len(srt))
            out[rows, j] = np.where(np.isnan(v), np.nan, pct)
    X[cols] = out
    return X


def feature_cols(X, kind, portable=False):
    drop = {"run_id", "instance_id", "target", "split"}
    cols = [c for c in X.columns if c not in drop]
    if kind == "issue":
        cols = [c for c in cols if c.startswith("issue_") or c == "model_b"]
    elif kind == "run":
        # Issue statistics are identical across a task's attempts: a task fingerprint.
        cols = [c for c in cols if not c.startswith("issue_")]
    if portable:
        cols = [c for c in cols if c not in NON_PORTABLE]
    return cols


def task_weights(X):
    """Each task weighs the same, however many attempts or checkpoints it has."""
    return 1.0 / X.groupby("instance_id").run_id.transform("size").to_numpy()


def fit_gbm(tr, va, cols, rounds=4000):
    dtr = lgb.Dataset(tr[cols], tr.target.astype(int), weight=task_weights(tr) * len(tr) / tr.instance_id.nunique())
    dva = lgb.Dataset(va[cols], va.target.astype(int))
    return lgb.train(GBM_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)])


def fit_ranker(tr, va, cols, rounds=3000):
    """Rank attempts against each other within (task, k).

    Trained only on mixed-outcome tasks, grouped by issue and checkpoint, so the
    only thing it can learn is which attempt at the same issue is going better.
    Early-stops on within-task AUROC (pair-weighted), the number it exists for.
    """
    def prep(X):
        X = X[X.groupby("instance_id").target.transform("nunique") == 2]
        X = X.sort_values(["instance_id", "k"])
        grp = X.groupby(["instance_id", "k"], sort=False).size().to_numpy()
        return X, grp

    trX, trg = prep(tr)
    vaX, vag = prep(va)
    bounds = np.concatenate([[0], np.cumsum(vag)])
    vy = vaX.target.to_numpy(bool)

    def within_auc(preds, data):
        num = den = 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            y, p = vy[a:b], preds[a:b]
            npos = y.sum()
            nneg = len(y) - npos
            if npos and nneg:
                r = pd.Series(p).rank().to_numpy()
                num += r[y].sum() - npos * (npos + 1) / 2
                den += npos * nneg
        return "within_auc", num / den, True

    params = {**GBM_PARAMS, "objective": "rank_xendcg", "metric": "None", "min_data_in_leaf": 300}
    dtr = lgb.Dataset(trX[cols], trX.target.astype(int), group=trg)
    dva = lgb.Dataset(vaX[cols], vaX.target.astype(int), group=vag)
    return lgb.train(params, dtr, rounds, valid_sets=[dva], feval=within_auc,
                     callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(100)])


def fit_lr(tr, cols):
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      LogisticRegression(C=0.3, max_iter=3000))
    m.fit(tr[cols], tr.target)
    return m


def oracle_task(X):
    """Leave-one-out solve rate of the other attempts at the same issue.

    Not deployable (it peeks at sibling outcomes). It measures how far knowing
    task difficulty alone gets you; within-task it is 0.5 by construction.
    """
    runs = X.drop_duplicates("run_id")[["run_id", "instance_id", "target"]]
    g = runs.groupby("instance_id").target
    s, n = g.transform("sum"), g.transform("size")
    loo = ((s - runs.target) / (n - 1).clip(lower=1)).where(n > 1, s.mean() / n.mean())
    return X[["run_id"]].merge(runs.assign(p=loo.values)[["run_id", "p"]], on="run_id").p.to_numpy()


def _save(out, name, X, p):
    (out / "preds").mkdir(parents=True, exist_ok=True)
    X[["run_id", "k", "split"]].assign(p=p).to_parquet(out / "preds" / f"{name}.parquet", index=False)


def add_args(p):
    p.add_argument("scaffold")
    p.add_argument("--eval-on", help="score on another scaffold's test split (transfer)")
    p.add_argument("--portable", action="store_true", help="drop scaffold-specific features")
    p.add_argument("--align", action="store_true", help="per-scaffold percentile features (transfer)")


def run(args):
    runs, splits, X = load(args.scaffold)
    portable = args.portable or bool(args.eval_on)
    tag = (args.scaffold + (f"_to_{args.eval_on}" if args.eval_on else "")
           + ("_portable" if args.portable and not args.eval_on else "") + ("_aligned" if args.align else ""))
    out = RESULTS / tag
    if args.align:
        X = align(X, feature_cols(X, "all", portable))
    tr, va = X[X.split == "train"], X[X.split == "val"]
    if args.eval_on:
        _, _, E = load(args.eval_on)
        if args.align:
            E = align(E, feature_cols(E, "all", portable))
        ev = E[E.split.isin(["val", "test"])].reset_index(drop=True)
    else:
        ev = X[X.split.isin(["val", "test"])].reset_index(drop=True)
    print(f"train rows {len(tr):,}  val rows {len(va):,}  eval rows {len(ev):,}")

    info = {}
    for name, kind in [("gbm_issue", "issue"), ("gbm", "run"), ("gbm_all", "all")]:
        cols = feature_cols(X, kind, portable)
        m = fit_gbm(tr, va, cols)
        _save(out, name, ev, m.predict(ev[cols], num_iteration=m.best_iteration))
        info[name] = {"iters": m.best_iteration, "val_auc": m.best_score["valid_0"]["auc"],
                      "n_features": len(cols)}
        if name == "gbm":
            imp = pd.DataFrame({"feature": cols, "gain": m.feature_importance("gain")})
            imp.sort_values("gain", ascending=False).to_csv(out / "importance_gbm.csv", index=False)
            m.save_model(str(out / "gbm.txt"))
        print(name, info[name], flush=True)

    cols = feature_cols(X, "run", portable)
    rk = fit_ranker(tr, va, cols)
    _save(out, "gbm_rank", ev, 1 / (1 + np.exp(-rk.predict(ev[cols], num_iteration=rk.best_iteration))))
    info["gbm_rank"] = {"iters": rk.best_iteration, "val_within_auc": rk.best_score["valid_0"]["within_auc"]}
    print("gbm_rank", info["gbm_rank"], flush=True)

    lr = fit_lr(tr.sample(min(len(tr), 400_000), random_state=0), cols)
    _save(out, "logreg", ev, lr.predict_proba(ev[cols])[:, 1])
    coef = lr[-1].coef_[0][: len(cols)]
    pd.DataFrame({"feature": cols, "coef": coef}).sort_values("coef").to_csv(
        out / "coef_logreg.csv", index=False)

    _save(out, "oracle_task", ev, oracle_task(ev))
    (out / "tabular.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out}")
