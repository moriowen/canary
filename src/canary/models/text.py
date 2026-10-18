"""
Fine-tune a long-context encoder on the raw transcript prefix.

    canary text swe

Each example is (run, k): the backbone tag plus the most recent turns that fit,
rendered as compact text. The issue text is left out: all of a task's ~22 attempts
share it, so including it lets the encoder memorise which issues are hopeless. Every epoch samples fresh checkpoints per run so one
model serves every k.
"""

import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from canary.data import RESULTS, processed

MODEL = "answerdotai/ModernBERT-base"
K_TEXT = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]
MAX_LEN = 2048
TURN_CHARS = {"cmd": 200, "thought": 240, "output": 360}


def render_turn(r):
    return (f"## {r.t} {r.action}{' !' if r.err_any else ''}\n"
            f"{r.thought[:TURN_CHARS['thought']]}\n$ {r.cmd[:TURN_CHARS['cmd']]}\n"
            f"> {r.output[:TURN_CHARS['output']]}\n")


def render(head, turn_txt, k, budget_chars):
    """Issue head, then as many of the last k turns as fit (most recent kept)."""
    keep, used = [], 0
    for s in reversed(turn_txt[:k]):
        if used + len(s) > budget_chars:
            break
        keep.append(s)
        used += len(s)
    gap = "[... earlier turns omitted ...]\n" if len(keep) < k else ""
    return head + gap + "".join(reversed(keep))


def load(scaffold):
    d = processed(scaffold)
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "instance_id", "target", "model", "issue", "n_turns"])
    splits = pd.read_parquet(d / "splits.parquet")
    runs = runs.merge(splits[["instance_id", "split"]], on="instance_id")
    t = pd.read_parquet(d / "turns", columns=["run_id", "t", "action", "cmd", "thought", "output"]
                        + [f"err_{k}" for k in ("traceback", "edit", "notfound", "exception", "exit")])
    t["err_any"] = t.filter(like="err_").any(axis=1)
    t = t.sort_values(["run_id", "t"])
    txt = {rid: [render_turn(r) for r in g.itertuples()] for rid, g in t.groupby("run_id", sort=False)}
    runs["head"] = [f"[{m}]\n" for m in runs.model]
    runs["turns"] = runs.run_id.map(lambda r: txt.get(r, []))
    return runs


def examples(runs, ks_per_run, rng):
    out = []
    for r in runs.itertuples():
        alive = [k for k in K_TEXT if k < r.n_turns]
        pick = alive if ks_per_run is None else rng.sample(alive, min(ks_per_run, len(alive)))
        out += [(r.Index, k) for k in pick]
    return out


def add_args(p):
    p.add_argument("scaffold")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--ks-per-run", type=int, default=3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--model", default=MODEL)


def run(args):
    rng = random.Random(0)
    torch.manual_seed(0)
    dev = "cuda"
    runs = load(args.scaffold)
    tok = AutoTokenizer.from_pretrained(args.model)
    net = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    budget = int((MAX_LEN - 20) * 3.2)   # ~3.2 chars per token on this text

    def enc(batch):
        texts = [render(runs.at[i, "head"], runs.at[i, "turns"], k, budget) for i, k in batch]
        e = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return {k: v.to(dev) for k, v in e.items()}

    @torch.no_grad()
    def predict(exs):
        net.eval()
        ps = []
        order = sorted(range(len(exs)), key=lambda i: exs[i][1])  # similar lengths together
        for i in range(0, len(order), args.bs * 4):
            b = [exs[j] for j in order[i:i + args.bs * 4]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps += torch.sigmoid(net(**enc(b)).logits.float().squeeze(-1)).cpu().tolist()
        p = np.empty(len(exs))
        p[order] = ps
        return p

    tr = runs[runs.split == "train"]
    va = runs[runs.split == "val"]
    va_ex = examples(va, None, rng)
    y = runs.target.to_numpy(np.float32)
    n_steps = args.epochs * math.ceil(len(tr) * args.ks_per_run / args.bs)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=n_steps, pct_start=0.06)
    best, log, step = -1, [], 0
    for ep in range(args.epochs):
        net.train()
        ex = examples(tr, args.ks_per_run, rng)
        rng.shuffle(ex)
        for i in range(0, len(ex), args.bs):
            b = ex[i:i + args.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(**enc(b)).logits.float().squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, torch.tensor([y[j] for j, _ in b], device=dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"ep {ep} step {step}/{n_steps} loss {loss.item():.4f}", flush=True)
        p = predict(va_ex)
        vdf = pd.DataFrame({"i": [i for i, _ in va_ex], "k": [k for _, k in va_ex], "p": p})
        vdf["y"] = y[vdf.i]
        auc = float(np.mean([roc_auc_score(g.y, g.p) for k, g in vdf.groupby("k")
                             if k in (1, 3, 5, 10) and g.y.nunique() == 2]))
        log.append({"epoch": ep, "val_auc_1_3_5_10": auc})
        print(log[-1], flush=True)
        if auc > best:
            best = auc
            net.save_pretrained(RESULTS / args.scaffold / "text_model")

    net = AutoModelForSequenceClassification.from_pretrained(RESULTS / args.scaffold / "text_model").to(dev)
    ev = runs[runs.split.isin(["val", "test"])]
    ev_ex = examples(ev, None, rng)
    p = predict(ev_ex)
    out = pd.DataFrame({"run_id": runs.run_id.to_numpy()[[i for i, _ in ev_ex]],
                        "k": [k for _, k in ev_ex], "split": runs.split.to_numpy()[[i for i, _ in ev_ex]],
                        "p": p})
    (RESULTS / args.scaffold / "preds").mkdir(parents=True, exist_ok=True)
    out.to_parquet(RESULTS / args.scaffold / "preds" / "text.parquet", index=False)
    (RESULTS / args.scaffold / "text.json").write_text(json.dumps({"best_val": best, "log": log}, indent=2))
    print("wrote text preds")
