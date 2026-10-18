"""
Unsupervised behavioural archetypes of quiet failures.

    canary archetypes swe [--k K]

The LLM taxonomy says *why* a run failed, from hindsight. This asks *how* failing
runs behave, with no labels and no text: each quiet failure is described only by
its end-of-run behaviour (rates of each action, errors, repeats, hedging, edits,
test runs, whether it verified its fix). Those features carry no repository or
issue information, so clusters can't form around a codebase the way embedded
diagnoses did.

Pipeline: log/standardise -> PCA (90% variance) -> Gaussian mixture, k by BIC.
Checks: stability across seeds (ARI), agreement with HDBSCAN, agreement with the
LLM categories (NMI/ARI), and the GRU monitor's catch rate per archetype.
Writes results/<scaffold>/archetypes/.
"""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from canary.commands.evaluate import POLICY_SCHEDULE
from canary.data import RESULTS, processed
from canary.policy import threshold_kills

RATES = ["frac_view", "frac_search", "frac_edit", "frac_create", "frac_run", "frac_test",
         "frac_install", "frac_cleanup", "frac_other", "frac_err", "frac_repeat_cmd",
         "frac_lex_still", "frac_lex_apology", "frac_lex_hedge", "frac_lex_success", "frac_lex_plan"]
COUNTS = ["k", "max_err_streak", "n_consec_repeat", "n_src_files", "n_viewed_files", "edit_lines",
          "n_test", "n_err_edit", "n_malformed", "run_after_edit"]
# Readable names for profiles.
LABEL = {"k": "turns", "frac_err": "error rate", "frac_repeat_cmd": "repeated cmds",
         "frac_lex_hedge": "hedging", "frac_lex_still": "'still/again'", "frac_lex_apology": "apologies",
         "frac_lex_success": "claims success", "frac_edit": "edit share", "frac_search": "search share",
         "frac_view": "view share", "frac_test": "test share", "frac_run": "run share",
         "n_src_files": "src files edited", "edit_lines": "lines edited", "n_test": "test runs",
         "max_err_streak": "longest error streak", "n_consec_repeat": "back-to-back repeats",
         "run_after_edit": "reran after edit", "first_src_rel": "first src edit (rel.)",
         "repro_ok": "reproduced bug", "n_err_edit": "rejected edits", "n_malformed": "malformed cmds",
         "n_viewed_files": "files viewed", "frac_create": "create share", "frac_install": "install share",
         "frac_cleanup": "cleanup share", "frac_other": "other share", "frac_lex_plan": "planning words"}


def features(full):
    X = full[RATES].copy()
    for c in COUNTS:
        X[c] = np.log1p(full[c].clip(lower=0))
    X["first_src_rel"] = (full.first_src_edit / full.k.clip(lower=1)).fillna(1.0)   # never = 1
    X["repro_ok"] = full.repro_err.fillna(0.0)                                      # never ran = 0
    return X.fillna(0.0)


def add_args(p):
    p.add_argument("scaffold")
    p.add_argument("--k", type=int, default=0, help="force the number of archetypes")
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--monitor", default="gru")


def run(args):
    d = processed(args.scaffold)
    out = RESULTS / args.scaffold / "archetypes"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "instance_id", "target", "exit_status", "outcome", "n_turns"])
    splits = pd.read_parquet(d / "splits.parquet")[["instance_id", "split"]]
    q = runs.merge(splits, on="instance_id")
    q = q[(~q.target) & q.exit_status.str.startswith("submitted") & (q.outcome == "tests_fail")]
    full = pd.read_parquet(d / "full.parquet").merge(q[["run_id", "split"]], on="run_id")
    X = features(full)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(0.9, random_state=0).fit(Z)
    P = pca.transform(Z)
    print(f"{len(full):,} quiet failures, {X.shape[1]} features -> {P.shape[1]} PCA dims (90% var)")

    rng = np.random.default_rng(0)
    sub = rng.choice(len(P), min(20000, len(P)), replace=False)
    sel = []
    for k in range(2, args.kmax + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(P[sub])
        lab = g.predict(P[sub])
        sil = silhouette_score(P[sub][:8000], lab[:8000])
        sel.append({"k": k, "bic": g.bic(P[sub]), "silhouette": sil})
        print(f"k={k:2d}  BIC={sel[-1]['bic']:.0f}  silhouette={sil:.3f}", flush=True)
    sel = pd.DataFrame(sel)
    # BIC keeps falling on 50k points; take the knee (largest drop in BIC improvement).
    gain = -sel.bic.diff()
    knee = int(sel.k.iloc[(gain - gain.shift(-1)).idxmax()]) if len(sel) > 3 else int(sel.k.iloc[sel.bic.idxmin()])
    k = args.k or knee
    print(f"chosen k={k} (BIC knee {knee})")

    gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=0).fit(P)
    full["archetype"] = gm.predict(P)
    full["confidence"] = gm.predict_proba(P).max(1)
    seeds = [GaussianMixture(k, covariance_type="full", n_init=1, random_state=s).fit(P[sub]).predict(P[sub])
             for s in range(1, 6)]
    stability = float(np.mean([adjusted_rand_score(full.archetype.to_numpy()[sub], s) for s in seeds]))
    hd = HDBSCAN(min_cluster_size=300, min_samples=50).fit(P[sub[:15000]])
    hmask = hd.labels_ >= 0
    hd_ari = float(adjusted_rand_score(full.archetype.to_numpy()[sub[:15000]][hmask], hd.labels_[hmask]))
    print(f"seed stability ARI={stability:.3f}; HDBSCAN: {hd.labels_.max() + 1} clusters, "
          f"{1 - hmask.mean():.0%} noise, ARI vs GMM on clustered points={hd_ari:.3f}")

    # Profiles: mean z-score per feature (what makes each archetype distinctive) and raw means.
    zdf = pd.DataFrame(Z, columns=X.columns).assign(archetype=full.archetype.values)
    zprof = zdf.groupby("archetype").mean()
    raw = full.assign(**{c: full[c] for c in ["k", "frac_err", "frac_repeat_cmd", "frac_lex_hedge",
                                               "n_test", "n_src_files", "run_after_edit"]})
    rprof = raw.groupby("archetype")[["k", "frac_err", "frac_repeat_cmd", "frac_lex_hedge", "n_test",
                                     "n_src_files", "run_after_edit", "edit_lines", "max_err_streak"]].mean()
    rprof["share"] = full.archetype.value_counts(normalize=True).sort_index()
    top = {a: ", ".join(f"{'+' if v > 0 else '-'}{LABEL.get(f, f)}" for f, v in
                        zprof.loc[a].reindex(zprof.loc[a].abs().sort_values(ascending=False).index).head(4).items())
           for a in zprof.index}
    rprof["distinctive"] = pd.Series(top)

    # Agreement with the LLM's hindsight categories.
    agree = {}
    tx = RESULTS / args.scaffold / "taxonomy" / "assigned.parquet"
    if tx.exists():
        llm = pd.read_parquet(tx, columns=["run_id", "name"]).merge(full[["run_id", "archetype"]], on="run_id")
        agree = {"runs": len(llm), "nmi": float(normalized_mutual_info_score(llm.name, llm.archetype)),
                 "ari": float(adjusted_rand_score(llm.name, llm.archetype))}
        ct = pd.crosstab(llm.archetype, llm.name, normalize="index")
        ct.to_csv(out / "crosstab_llm.csv")
        print(f"\nvs LLM categories ({len(llm):,} runs): NMI={agree['nmi']:.3f} ARI={agree['ari']:.3f}")
        print((ct * 100).round(0).to_string())

    # Monitor catch rate per archetype, test split, threshold at <= 5% successes lost.
    s = json.loads((RESULTS / args.scaffold / "summary.json").read_text())
    tau = s[args.monitor]["kill_threshold@5%"]
    preds = pd.read_parquet(RESULTS / args.scaffold / "preds" / f"{args.monitor}.parquet")
    test = full[full.split == "test"]
    preds = preds[preds.run_id.isin(test.run_id) & preds.k.isin(POLICY_SCHEDULE)]
    kills = threshold_kills(preds, tau)
    test = test.assign(caught=test.run_id.map(kills).notna())
    rprof["caught"] = test.groupby("archetype").caught.mean()
    rprof = rprof.sort_values("share", ascending=False)
    print("\n" + rprof.round(3).to_string())

    full[["run_id", "split", "archetype", "confidence"]].to_parquet(out / "assignments.parquet", index=False)
    rprof.to_csv(out / "profiles.csv")
    zprof.to_csv(out / "profiles_z.csv")
    sel.to_csv(out / "model_selection.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "runs": len(full), "features": list(X.columns), "pca_dims": int(P.shape[1]), "k": k, "bic_knee": knee,
        "seed_stability_ari": stability, "hdbscan_clusters": int(hd.labels_.max() + 1),
        "hdbscan_noise": float(1 - hmask.mean()), "hdbscan_ari": hd_ari, "llm_agreement": agree,
        "monitor": args.monitor, "threshold": tau, "caught_overall": float(test.caught.mean()),
    }, indent=2))
    print(f"\nwrote {out}")
