"""
GRU over per-turn vectors: one network, a prediction after every turn.

    canary seq swe
    canary seq swe --eval-on oh

The static input (backbone size) initialises the hidden state. Issue statistics
are left out on purpose: they are identical across a task's ~22 attempts, so the
network uses them to memorise per-task solve rates (train loss collapses, val
stalls). Position 0 is the prior before any turn; every later position updates it.
"""

import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from canary.data import RESULTS, processed
from canary.features import CHECKPOINTS, STEP_COLS

T_MAX = 100
STATIC = ["issue_chars", "issue_lines", "issue_code_blocks", "issue_traceback",
          "issue_py_paths", "issue_expected", "issue_urls"]
MODELS = ["8b", "70b", "405b"]


def load(scaffold, use_model):
    d = processed(scaffold)
    runs = pd.read_parquet(d / "runs.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    steps = pd.read_parquet(d / "steps")
    k0 = pd.read_parquet(d / "prefix.parquet", columns=["run_id", "k"] + STATIC).query("k == 0")
    df = runs[["run_id", "instance_id", "target", "model", "n_turns"]].merge(
        splits[["instance_id", "split"]], on="instance_id").merge(steps, on="run_id").merge(
        k0.drop(columns="k"), on="run_id", how="left")
    S = np.ones((len(df), 1), np.float32)
    if use_model:
        S = np.concatenate([S, np.stack([df.model.str.endswith(m).to_numpy(np.float32)
                                         for m in MODELS], 1)], 1)
    D = len(STEP_COLS)
    X = [np.asarray(s, np.float32).reshape(-1, D)[:T_MAX] for s in df.steps]
    return df.drop(columns=["steps"]), X, S


class Net(nn.Module):
    def __init__(self, d_step, d_static, h=192, layers=2):
        super().__init__()
        self.layers, self.h = layers, h
        self.init = nn.Sequential(nn.Linear(d_static, h), nn.Tanh(), nn.Linear(h, layers * h), nn.Tanh())
        self.inp = nn.Sequential(nn.Linear(d_step, h), nn.GELU())
        self.gru = nn.GRU(h, h, layers, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(h + d_static, h), nn.GELU(), nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, x, s):
        B = x.shape[0]
        h0 = self.init(s).view(B, self.layers, self.h).transpose(0, 1).contiguous()
        y, _ = self.gru(self.inp(x), h0)
        y = torch.cat([h0[-1].unsqueeze(1), y], 1)            # position 0 = before any turn
        ss = s.unsqueeze(1).expand(-1, y.shape[1], -1)
        return self.head(torch.cat([y, ss], -1)).squeeze(-1)  # (B, T+1)


def batches(idx, X, S, y, n_turns, bs, dev, shuffle):
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(X[j]) for j in b)
        x = np.zeros((len(b), L, X[b[0]].shape[1]), np.float32)
        mask = np.zeros((len(b), L + 1), np.float32)
        for r, j in enumerate(b):
            x[r, :len(X[j])] = X[j]
            alive = min(n_turns[j], T_MAX + 1)   # positions k < n_turns are still running
            mask[r, :min(alive, L + 1)] = 1
        yield (b, torch.from_numpy(x).to(dev), torch.from_numpy(S[b]).to(dev),
               torch.from_numpy(y[b]).float().to(dev), torch.from_numpy(mask).to(dev))


@torch.no_grad()
def predict(net, idx, X, S, y, n_turns, dev):
    net.eval()
    out = {}
    for b, x, s, _, m in batches(idx, X, S, y, n_turns, 512, dev, False):
        p = torch.sigmoid(net(x, s)).cpu().numpy()
        mm = m.cpu().numpy()
        for r, j in enumerate(b):
            out[j] = p[r][mm[r] > 0]
    return out


def to_rows(df, probs):
    rows = []
    for j, p in probs.items():
        for k in CHECKPOINTS:
            if k < len(p):
                rows.append((df.run_id.iat[j], k, df.split.iat[j], float(p[k])))
    return pd.DataFrame(rows, columns=["run_id", "k", "split", "p"])


def add_args(p):
    p.add_argument("scaffold")
    p.add_argument("--eval-on")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_model = not args.eval_on
    df, X, S = load(args.scaffold, use_model)
    mu, sd = np.zeros(S.shape[1], np.float32), np.ones(S.shape[1], np.float32)
    y = df.target.to_numpy(np.float32)
    n = df.n_turns.to_numpy()
    tr = np.where(df.split == "train")[0]
    va = np.where(df.split == "val")[0]

    net = Net(len(STEP_COLS), S.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * math.ceil(len(tr) / 256))
    best, best_state, log = -1, None, []
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for _, x, s, yy, m in batches(tr, X, S, y, n, 256, dev, True):
            logit = net(x, s)
            l = nn.functional.binary_cross_entropy_with_logits(
                logit, yy.unsqueeze(1).expand_as(logit), reduction="none")
            loss = ((l * m).sum(1) / m.sum(1).clamp(min=1)).mean()   # each run weighs the same
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        vp = to_rows(df, predict(net, va, X, S, y, n, dev)).merge(df[["run_id", "target"]], on="run_id")
        auc = float(np.mean([roc_auc_score(g.target, g.p) for k, g in vp.groupby("k")
                             if k in (1, 3, 5, 10) and g.target.nunique() == 2]))
        log.append({"epoch": ep, "train_loss": tot / len(tr), "val_auc_1_3_5_10": auc})
        print(log[-1], flush=True)
        if auc > best:
            best = auc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    if args.eval_on:
        edf, EX, ES = load(args.eval_on, False)
        ES = (ES - mu) / sd
        ey, en = edf.target.to_numpy(np.float32), edf.n_turns.to_numpy()
        idx = np.where(edf.split.isin(["val", "test"]))[0]
        preds = to_rows(edf, predict(net, idx, EX, ES, ey, en, dev))
        out = RESULTS / f"{args.scaffold}_to_{args.eval_on}"
    else:
        idx = np.where(df.split.isin(["val", "test"]))[0]
        preds = to_rows(df, predict(net, idx, X, S, y, n, dev))
        out = RESULTS / args.scaffold
    (out / "preds").mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "preds" / "gru.parquet", index=False)
    (out / "gru.json").write_text(json.dumps({"best_val": best, "log": log}, indent=2))
    torch.save(net.state_dict(), out / "gru.pt")
    print(f"wrote {out}/preds/gru.parquet  best val {best:.4f}")
