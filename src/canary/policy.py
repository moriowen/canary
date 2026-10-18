"""
Abort-policy simulation: what a monitor actually buys you.

A policy picks, for each run, the turn at which to kill it (or never). Killing
at turn k saves every turn the run would have spent after k, and loses the run
if it was going to succeed. Two cost units:

  turns   one unit per agent step.
  ctx     each step re-reads the whole transcript so far, so a step's cost is
          the context length at that step. Late turns are the expensive ones,
          which is why early kills save more than their turn count suggests.
"""

import numpy as np
import pandas as pd

LOSS_BUDGETS = [0.01, 0.02, 0.05, 0.10]


def run_costs(turns, runs):
    """Per-run cumulative cost arrays: turn_cost[i] = context chars when turn i runs."""
    base = runs.set_index("run_id").issue.str.len()
    t = turns[["run_id", "t", "out_chars", "thought_chars", "cmd"]].copy()
    t["c"] = t.out_chars + t.thought_chars + t.cmd.str.len()
    t = t.sort_values(["run_id", "t"])
    t["ctx"] = t.groupby("run_id").c.cumsum() - t.c + t.run_id.map(base).fillna(0)
    return {rid: g.ctx.to_numpy(np.float64) for rid, g in t.groupby("run_id", sort=False)}


class Sim:
    def __init__(self, runs, costs):
        self.runs = runs.set_index("run_id")
        self.ids = list(self.runs.index)
        self.n = self.runs.n_turns.reindex(self.ids).to_numpy()
        self.y = self.runs.target.reindex(self.ids).to_numpy(bool)
        self.suffix = {}
        tot_ctx = 0.0
        for rid in self.ids:
            c = costs.get(rid, np.zeros(0))
            s = np.concatenate([np.cumsum(c[::-1])[::-1], [0.0]])  # s[k] = cost of turns k..end
            self.suffix[rid] = s
            tot_ctx += s[0]
        self.tot_turns = self.n.sum()
        self.tot_ctx = tot_ctx

    def evaluate(self, kill):
        """kill: dict run_id -> k (kill after seeing k turns). Missing = never."""
        saved_t = saved_c = 0.0
        lost = killed = 0
        for i, rid in enumerate(self.ids):
            k = kill.get(rid)
            if k is None or k >= self.n[i]:
                continue
            killed += 1
            saved_t += self.n[i] - k
            saved_c += self.suffix[rid][min(k, len(self.suffix[rid]) - 1)]
            lost += self.y[i]
        S = self.y.sum()
        return {
            "killed": killed / len(self.ids),
            "lost": lost / max(S, 1),
            "saved_turns": saved_t / self.tot_turns,
            "saved_ctx": saved_c / self.tot_ctx,
            # successes per unit compute, relative to running everything to completion
            "yield_gain": ((S - lost) / (self.tot_ctx - saved_c)) / (S / self.tot_ctx) - 1,
        }


def threshold_kills(preds, tau, schedule=None):
    """Kill at the first checkpoint where P(success) < tau."""
    d = preds if schedule is None else preds[preds.k.isin(schedule)]
    hit = d[d.p < tau]
    first = hit.groupby("run_id").k.min()
    return first.to_dict()


def sweep_model(sim, preds, n=120, schedule=None):
    qs = np.unique(np.quantile(preds.p, np.linspace(0, 0.995, n)))
    rows = []
    for tau in qs:
        r = sim.evaluate(threshold_kills(preds, tau, schedule))
        rows.append({"param": float(tau), **r})
    return pd.DataFrame(rows)


def sweep_streak(sim, turns, max_m=12):
    """Rule: kill after m consecutive turns with an error."""
    err_cols = [c for c in turns.columns if c.startswith("err_")]
    t = turns[["run_id", "t"] + err_cols].sort_values(["run_id", "t"])
    e = t[err_cols].to_numpy().any(1)
    first = {}
    for m in range(1, max_m + 1):
        first[m] = {}
    for rid, idx in t.groupby("run_id", sort=False).indices.items():
        streak = 0
        ee = e[idx]
        best = {}
        for j, v in enumerate(ee):
            streak = streak + 1 if v else 0
            if streak and streak not in best:
                best[streak] = j + 1
        for m in range(1, max_m + 1):
            if m in best:
                first[m][rid] = best[m]
    return pd.DataFrame([{"param": m, **sim.evaluate(first[m])} for m in range(1, max_m + 1)])


def sweep_budget(sim, budgets=None):
    """Rule: hard cap of N turns."""
    budgets = budgets or [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150]
    rows = []
    for N in budgets:
        kill = {rid: N for rid, n in zip(sim.ids, sim.n) if n > N}
        rows.append({"param": N, **sim.evaluate(kill)})
    return pd.DataFrame(rows)


def at_budget(curve, budgets=LOSS_BUDGETS, metric="saved_ctx"):
    """Best saving achievable while losing at most b of the successes."""
    out = {}
    for b in budgets:
        ok = curve[curve.lost <= b]
        out[f"{metric}@{int(b * 100)}%"] = float(ok[metric].max()) if len(ok) else 0.0
    return out
