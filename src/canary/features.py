"""
Prefix features: what a monitor could know after watching the first k turns.

One pass over a run's turns, snapshotting at each checkpoint k. A run only
gets a row at k if it is still going (n_turns > k): a run that already
submitted can't be aborted, and its final `submit` would leak the ending.

Nothing here reads the label, exit_status, the patch, or the eval log.
"""

import math
import re

import numpy as np

from canary.turns import ACTIONS, ERR_KINDS

CHECKPOINTS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]

LEXICON = {
    "still": re.compile(r"\b(still|again|persists?|same (error|issue|problem)|keeps?)\b", re.I),
    "apology": re.compile(r"\b(apologi[sz]e|sorry|mistake|oops|my bad|incorrect(ly)?)\b", re.I),
    "hedge": re.compile(r"\b(seems|appears|might|maybe|perhaps|not sure|unclear|possibly)\b", re.I),
    "success": re.compile(r"\b(fixed|resolved|works?|working|success(fully)?|correct(ly)?)\b", re.I),
    "plan": re.compile(r"\b(first|then|next|step|let's|plan)\b", re.I),
}

_CODEBLOCK = re.compile(r"```")
_PYPATH = re.compile(r"[\w/]+\.py\b")


def issue_features(issue):
    issue = issue or ""
    return {
        "issue_chars": len(issue),
        "issue_lines": issue.count("\n") + 1,
        "issue_code_blocks": len(_CODEBLOCK.findall(issue)) // 2,
        "issue_traceback": int("Traceback" in issue),
        "issue_py_paths": len(set(_PYPATH.findall(issue))),
        "issue_expected": int(bool(re.search(r"expected|should", issue, re.I))),
        "issue_urls": issue.count("http"),
    }


class _Acc:
    """Running statistics over turns seen so far."""

    def __init__(self):
        self.n = 0
        self.act = dict.fromkeys(ACTIONS, 0)
        self.err = dict.fromkeys(ERR_KINDS, 0)
        self.any_err = 0
        self.streak = 0
        self.max_streak = 0
        self.recent = []          # any_err of last turns
        self.cmds = {}
        self.max_cmd = 0
        self.repeat = 0
        self.consec_repeat = 0
        self.prev_cmd = None
        self.kind = {"src": 0, "scratch": 0, "test": 0}
        self.src_files = set()
        self.viewed = set()
        self.edit_lines = 0
        self.first_edit = math.nan
        self.first_src_edit = math.nan
        self.first_run = math.nan
        self.run_after_edit = 0
        self.edited_since_run = False
        self.test_pass = 0
        self.test_fail = 0
        self.last_test_failed = math.nan
        self.repro_err = math.nan     # first scratch run errored (bug reproduced)
        self.last_run_err = math.nan  # most recent run/test errored
        self.out_chars = 0
        self.thought_chars = 0
        self.last_out = 0
        self.last_thought = 0
        self.thoughts = []
        self.lex = dict.fromkeys(LEXICON, 0)
        self.empty = 0
        self.malformed = 0
        self.ctx = 0                  # cumulative context chars (proxy for prefill cost)

    def add(self, x):
        self.n += 1
        a = x["action"]
        self.act[a] += 1
        e = 0
        for k in ERR_KINDS:
            if x[f"err_{k}"]:
                self.err[k] += 1
                e = 1
        self.any_err += e
        self.streak = self.streak + 1 if e else 0
        self.max_streak = max(self.max_streak, self.streak)
        self.recent.append(e)

        c = x["cmd"]
        if c:
            cnt = self.cmds.get(c, 0)
            if cnt:
                self.repeat += 1
            self.cmds[c] = cnt + 1
            self.max_cmd = max(self.max_cmd, cnt + 1)
            if c == self.prev_cmd:
                self.consec_repeat += 1
        self.prev_cmd = c

        pk = x["path_kind"]
        if a in ("edit", "create"):
            if pk in self.kind:
                self.kind[pk] += 1
            if pk == "src":
                self.src_files.add(x["path"])
                if math.isnan(self.first_src_edit):
                    self.first_src_edit = self.n
            if a == "edit" and math.isnan(self.first_edit):
                self.first_edit = self.n
            self.edit_lines += x["edit_lines"]
            if pk == "src" and a == "edit":
                self.edited_since_run = True
        if a == "view" and x["path"]:
            self.viewed.add(x["path"])
        if a in ("run", "test"):
            if math.isnan(self.first_run):
                self.first_run = self.n
            if self.edited_since_run:
                self.run_after_edit += 1
                self.edited_since_run = False
            run_err = int(bool(x["err_traceback"] or x["err_exception"] or x["err_exit"]))
            self.last_run_err = run_err
            if a == "run" and pk == "scratch" and math.isnan(self.repro_err):
                self.repro_err = run_err
        if a == "test":
            self.test_pass += x["test_passed"]
            self.test_fail += x["test_failed"]
            self.last_test_failed = int(x["test_failed"] > 0 or bool(x["err_traceback"]))

        self.out_chars += x["out_chars"]
        self.thought_chars += x["thought_chars"]
        self.last_out = x["out_chars"]
        self.last_thought = x["thought_chars"]
        self.thoughts.append(x["thought_chars"])
        for k, rx in LEXICON.items():
            if rx.search(x["thought"]):
                self.lex[k] += 1
        self.empty += x["empty_out"]
        self.malformed += x["malformed"]
        self.ctx += x["out_chars"] + x["thought_chars"] + len(x["cmd"])

    def snapshot(self):
        n = max(self.n, 1)
        f = {"k": self.n}
        for a in ACTIONS:
            f[f"n_{a}"] = self.act[a]
            f[f"frac_{a}"] = self.act[a] / n
        for k in ERR_KINDS:
            f[f"n_err_{k}"] = self.err[k]
        f.update({
            "n_err": self.any_err, "frac_err": self.any_err / n,
            "max_err_streak": self.max_streak, "cur_err_streak": self.streak,
            "err_last3": sum(self.recent[-3:]), "err_last5": sum(self.recent[-5:]),
            "n_repeat_cmd": self.repeat, "frac_repeat_cmd": self.repeat / n,
            "max_cmd_count": self.max_cmd, "n_consec_repeat": self.consec_repeat,
            "distinct_cmds": len(self.cmds),
            "n_src_edit": self.kind["src"], "n_scratch_edit": self.kind["scratch"],
            "n_test_edit": self.kind["test"], "n_src_files": len(self.src_files),
            "n_viewed_files": len(self.viewed), "edit_lines": self.edit_lines,
            "first_edit": self.first_edit, "first_src_edit": self.first_src_edit,
            "first_run": self.first_run, "run_after_edit": self.run_after_edit,
            "unverified_edit": int(self.edited_since_run),
            "test_pass": self.test_pass, "test_fail": self.test_fail,
            "last_test_failed": self.last_test_failed,
            "repro_err": self.repro_err, "last_run_err": self.last_run_err,
            "log_out_chars": math.log1p(self.out_chars), "log_out_mean": math.log1p(self.out_chars / n),
            "log_out_last": math.log1p(self.last_out),
            "log_thought_mean": math.log1p(self.thought_chars / n),
            "log_thought_last": math.log1p(self.last_thought),
            "thought_trend": (np.mean(self.thoughts[-3:]) - np.mean(self.thoughts[:3])) / 1000 if self.thoughts else 0.0,
            "n_empty_out": self.empty, "n_malformed": self.malformed,
            "log_ctx": math.log1p(self.ctx),
        })
        for k in LEXICON:
            f[f"lex_{k}"] = self.lex[k]
            f[f"frac_lex_{k}"] = self.lex[k] / n
        return f


def prefix_rows(turns, issue, checkpoints=CHECKPOINTS, full=False):
    """Feature dicts at each checkpoint where the run is still alive.

    With full=True, also returns the end-of-run snapshot (for the taxonomy).
    """
    base = issue_features(issue)
    n_turns = len(turns)
    acc = _Acc()
    rows = []
    cps = [k for k in checkpoints if k < n_turns]
    j = 0
    for k in cps:
        while j < k:
            acc.add(turns[j])
            j += 1
        rows.append({**base, **acc.snapshot()})
    if not full:
        return rows
    while j < n_turns:
        acc.add(turns[j])
        j += 1
    return rows, {**base, **acc.snapshot()}


# Per-turn vectors for the sequence model: one row per turn, no running sums.
STEP_COLS = ([f"act_{a}" for a in ACTIONS] + [f"err_{k}" for k in ERR_KINDS]
             + ["any_err", "log_out", "log_thought", "repeat", "consec_repeat", "kind_src",
                "kind_scratch", "kind_test", "log_edit_lines", "test_passed", "test_failed",
                "empty_out", "malformed"] + [f"lex_{k}" for k in LEXICON] + ["pos"])


def step_matrix(turns):
    seen = set()
    prev = None
    out = np.zeros((len(turns), len(STEP_COLS)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(STEP_COLS)}
    for t, x in enumerate(turns):
        r = out[t]
        r[idx[f"act_{x['action']}"]] = 1
        e = 0
        for k in ERR_KINDS:
            if x[f"err_{k}"]:
                r[idx[f"err_{k}"]] = 1
                e = 1
        r[idx["any_err"]] = e
        r[idx["log_out"]] = math.log1p(x["out_chars"]) / 10
        r[idx["log_thought"]] = math.log1p(x["thought_chars"]) / 10
        c = x["cmd"]
        r[idx["repeat"]] = float(bool(c) and c in seen)
        r[idx["consec_repeat"]] = float(bool(c) and c == prev)
        seen.add(c)
        prev = c
        pk = x["path_kind"]
        if pk in ("src", "scratch", "test") and x["action"] in ("edit", "create", "run", "view"):
            r[idx[f"kind_{pk}"]] = 1
        r[idx["log_edit_lines"]] = math.log1p(x["edit_lines"]) / 5
        r[idx["test_passed"]] = min(x["test_passed"], 50) / 50
        r[idx["test_failed"]] = min(x["test_failed"], 50) / 50
        r[idx["empty_out"]] = x["empty_out"]
        r[idx["malformed"]] = x["malformed"]
        for k, rx in LEXICON.items():
            r[idx[f"lex_{k}"]] = float(bool(rx.search(x["thought"])))
        r[idx["pos"]] = min(t, 100) / 100
    return out
