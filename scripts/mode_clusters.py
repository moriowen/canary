"""Cluster the LLM's short failure-mode labels (not the repo-specific summaries).

Embeds each distinct mode string once, k-means over runs, and prints the top
labels per cluster for k in a range so the taxonomy granularity can be chosen.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from canary.data import RESULTS
from canary.taxonomy import _embed

lab = pd.read_parquet(RESULTS / "swe" / "taxonomy" / "labels.parquet")
lab = lab[lab["mode"].str.len() > 0].copy()
lab["m"] = lab["mode"].str.lower().str.strip().str.replace("-", " ")
uniq = lab.m.unique().tolist()
E = dict(zip(uniq, _embed(uniq)))
X = np.stack([E[m] for m in lab.m])
for k in [int(a) for a in sys.argv[1:]] or [6, 8, 10, 12]:
    km = KMeans(k, n_init=20, random_state=0).fit(X)
    sub = np.random.default_rng(0).choice(len(X), 5000, replace=False)
    sil = silhouette_score(X[sub], km.labels_[sub], metric="cosine")
    print(f"\n===== k={k} silhouette={sil:.3f}")
    lab["c"] = km.labels_
    for c, g in sorted(lab.groupby("c"), key=lambda t: -len(t[1])):
        print(f"{len(g) / len(lab):6.1%}  " + ", ".join(f"{m} ({n})" for m, n in g.m.value_counts().head(7).items()))
