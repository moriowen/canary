"""
Failure taxonomy for quiet failures: runs that submitted a patch and still failed.

exit_status says "submitted" for all of them, so it explains nothing. Three stages:

    canary taxonomy label swe     # LLM (vLLM) reads each run, writes a diagnosis
    canary taxonomy propose swe   # LLM proposes categories from the diagnoses' mode labels
    canary taxonomy assign swe    # LLM puts every diagnosis in exactly one category

`cluster` and `name` (k-means over embedded diagnoses) are kept for comparison. They
grouped runs by repository and by wording, not by failure type: silhouette ~0.05 on
full diagnoses, and on the short mode labels "misplaced fix" and "misplaced logic"
landed in different clusters. The propose/assign pass (as in TnT-LLM) fixes that.

The LLM sees the whole run, the final patch and the tests that still fail, so it
diagnoses with hindsight. The monitor never sees any of that; the point of the
taxonomy is to ask which kinds of failure are visible early.
"""

import json
import re

import numpy as np
import pandas as pd

from canary.data import RESULTS, processed

LLM = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DISCOVERY = 3000
STAGES = ["understanding", "localization", "reproduction", "implementation", "verification", "tooling"]

SYSTEM = """You review failed attempts by an AI coding agent to fix a GitHub issue.
The agent submitted a patch, but the hidden tests for the issue still fail.
Diagnose the root cause of the failure from the transcript, the final patch and the failing tests.

Reply with JSON only:
{"summary": "<one specific sentence: what the agent got wrong>",
 "stage": "<where it broke: understanding | localization | reproduction | implementation | verification | tooling>",
 "mode": "<2-6 word generic name for the failure pattern, not specific to this repository>"}

Stages: understanding = misread what the issue asks; localization = changed the wrong code;
reproduction = never reproduced the bug, or reproduced the wrong thing; implementation = right
place, wrong or incomplete change; verification = did not test, or misread test results;
tooling = fought the editor/shell and ran out of useful progress."""

NAMER = """Below are diagnoses of failed coding-agent runs that a clustering grouped together.
Name the failure mode they share. Reply with JSON only:
{"name": "<2-5 words>", "description": "<one sentence a software engineer would recognise>"}"""


def quiet_failures(runs, splits):
    q = runs.merge(splits[["instance_id", "split"]], on="instance_id")
    q = q[(~q.target) & q.exit_status.str.startswith("submitted") & (q.outcome == "tests_fail")]
    disc = q[q.split == "train"].sample(min(DISCOVERY, (q.split == "train").sum()), random_state=0)
    return pd.concat([disc, q[q.split == "test"]]).drop_duplicates("run_id")


def render(run, turns, max_chars=14000):
    lines = []
    for r in turns.itertuples():
        err = " ERROR" if (r.err_traceback or r.err_exception or r.err_edit or r.err_notfound) else ""
        out = " ".join(r.output.split())[:220]
        th = " ".join(r.thought.split())[:160]
        lines.append(f"[{r.t + 1}] {r.action}{err} | {th} | $ {' '.join(r.cmd.split())[:160]} -> {out}")
    if len(lines) > 44:
        lines = lines[:20] + [f"... {len(lines) - 40} turns omitted ..."] + lines[-20:]
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max_chars // 2] + "\n...\n" + body[-max_chars // 2:]
    return (f"ISSUE:\n{run.issue[:1500]}\n\nAGENT TRANSCRIPT ({run.n_turns} turns):\n{body}\n\n"
            f"FINAL PATCH:\n{run.patch[:2500]}\n\nTESTS STILL FAILING:\n{run.failed_tests or '(not listed)'}")


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def label(args):
    from vllm import LLM as VLLM, SamplingParams

    d = processed(args.scaffold)
    runs = pd.read_parquet(d / "runs.parquet")
    splits = pd.read_parquet(d / "splits.parquet")
    q = quiet_failures(runs, splits)
    if args.limit:
        q = q.head(args.limit)
    ids = set(q.run_id)
    t = pd.read_parquet(d / "turns", filters=[("run_id", "in", list(ids))]).sort_values(["run_id", "t"])
    by = dict(tuple(t.groupby("run_id")))
    llm = VLLM(args.model, max_model_len=12288, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": render(r, by[r.run_id])}],
                                       tokenize=False, add_generation_prompt=True)
               for r in q.itertuples()]
    # Keep prompts inside the context window.
    prompts = [p if len(p) < 36000 else p[:18000] + "\n...\n" + p[-18000:] for p in prompts]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
    rows = []
    for r, o in zip(q.itertuples(), outs):
        j = _json(o.outputs[0].text)
        stage = str(j.get("stage", "")).strip().lower()
        rows.append({"run_id": r.run_id, "split": r.split, "summary": str(j.get("summary", "")),
                     "stage": stage if stage in STAGES else "other", "mode": str(j.get("mode", "")),
                     "raw": o.outputs[0].text})
    out = RESULTS / args.scaffold / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "labels.parquet", index=False)
    print(f"labelled {len(rows):,} runs; parse failures {sum(not r['summary'] for r in rows)}")


def _embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED)
    m = AutoModel.from_pretrained(EMBED).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            e = tok(texts[i:i + 256], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            h = m(**e).last_hidden_state
            v = (h * e["attention_mask"].unsqueeze(-1)).sum(1) / e["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def cluster(args):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    out = RESULTS / args.scaffold / "taxonomy"
    lab = pd.read_parquet(out / "labels.parquet")
    lab = lab[lab.summary.str.len() > 0].reset_index(drop=True)
    X = _embed((lab["mode"] + ": " + lab.summary).tolist())
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(6000, len(X)), replace=False)
    scores = {}
    for k in range(6, 17):
        km = KMeans(k, n_init=10, random_state=0).fit(X[sub])
        scores[k] = float(silhouette_score(X[sub], km.labels_, metric="cosine"))
        print(f"k={k} silhouette={scores[k]:.4f}", flush=True)
    k = args.k or max(scores, key=scores.get)
    km = KMeans(k, n_init=20, random_state=0).fit(X)
    lab["cluster"] = km.labels_
    # Distance to centroid, so naming can show the most typical members first.
    lab["dist"] = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    lab.drop(columns=["raw"]).to_parquet(out / "clusters.parquet", index=False)
    (out / "silhouette.json").write_text(json.dumps({"scores": scores, "k": k}, indent=2))
    print(f"k={k}", lab.cluster.value_counts().sort_index().to_dict())


def name(args):
    from vllm import LLM as VLLM, SamplingParams

    out = RESULTS / args.scaffold / "taxonomy"
    lab = pd.read_parquet(out / "clusters.parquet")
    llm = VLLM(args.model, max_model_len=8192, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts, ids = [], []
    for c, g in lab.groupby("cluster"):
        ex = g.nsmallest(20, "dist").sample(frac=1, random_state=0)
        ex = pd.concat([ex, g.sample(min(10, len(g)), random_state=1)]).drop_duplicates("run_id")
        body = "\n".join(f"- [{r.stage}] {r.mode}: {r.summary}" for r in ex.itertuples())
        prompts.append(tok.apply_chat_template([{"role": "system", "content": NAMER},
                                                {"role": "user", "content": body}],
                                               tokenize=False, add_generation_prompt=True))
        ids.append(int(c))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=120))
    info = []
    for c, o in zip(ids, outs):
        j = _json(o.outputs[0].text)
        g = lab[lab.cluster == c]
        info.append({"cluster": c, "name": j.get("name", f"cluster {c}"),
                     "description": j.get("description", ""), "size": len(g),
                     "share": len(g) / len(lab),
                     "top_stage": g.stage.value_counts().index[0],
                     "examples": g.nsmallest(3, "dist").summary.tolist()})
    info.sort(key=lambda r: -r["size"])
    (out / "clusters.json").write_text(json.dumps(info, indent=2))
    for r in info:
        print(f"{r['share']:6.1%}  {r['name']}: {r['description']}")


PROPOSER = """You are building a taxonomy of why AI coding agents fail.
Each failed run was diagnosed separately; below are the most frequent short failure-mode
labels (with counts) and a sample of full diagnoses.

Propose 6 to 9 categories that together cover nearly all runs. Categories must be mutually
exclusive, about the KIND of mistake (not the repository or library), and useful to an
engineer deciding how to fix an agent. Merge near-synonyms. Prefer categories a reader
could apply consistently from a one-sentence diagnosis.

Reply with JSON only:
{"categories": [{"name": "<2-4 words>", "definition": "<one sentence>",
                 "includes": "<typical labels or cues>", "excludes": "<what belongs elsewhere>"}]}"""

ASSIGNER = """Classify one diagnosis of a failed AI coding-agent run into exactly one category.

Categories:
{cats}

Reply with the category number only."""


def propose(args):
    from vllm import LLM as VLLM, SamplingParams

    out = RESULTS / args.scaffold / "taxonomy"
    lab = pd.read_parquet(out / "labels.parquet")
    lab = lab[lab.summary.str.len() > 0]
    modes = lab["mode"].str.lower().str.strip().str.replace("-", " ").value_counts().head(300)
    sample = lab.sample(60, random_state=0)
    body = ("MOST FREQUENT LABELS:\n" + "\n".join(f"{m} ({n})" for m, n in modes.items())
            + "\n\nSAMPLE DIAGNOSES:\n" + "\n".join(f"- [{r.mode}] {r.summary}" for r in sample.itertuples()))
    llm = VLLM(args.model, max_model_len=16384, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template([{"role": "system", "content": PROPOSER}, {"role": "user", "content": body}],
                                     tokenize=False, add_generation_prompt=True)
    o = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4000))[0].outputs[0].text
    (out / "categories_raw.txt").write_text(o)
    cats = _json(o).get("categories", [])
    if not cats:
        raise SystemExit(f"no categories parsed; raw reply in {out / 'categories_raw.txt'}")
    (out / "categories.json").write_text(json.dumps(cats, indent=2))
    for i, c in enumerate(cats, 1):
        print(f"{i}. {c['name']}: {c['definition']}")
    return llm


def assign(args, llm=None):
    from vllm import LLM as VLLM, SamplingParams

    out = RESULTS / args.scaffold / "taxonomy"
    cats = json.loads((out / "categories.json").read_text())
    lab = pd.read_parquet(out / "labels.parquet")
    lab = lab[lab.summary.str.len() > 0].reset_index(drop=True)
    listing = "\n".join(f"{i}. {c['name']}: {c['definition']} (includes: {c.get('includes', '')}; "
                        f"excludes: {c.get('excludes', '')})" for i, c in enumerate(cats, 1))
    llm = llm or VLLM(args.model, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sys_prompt = ASSIGNER.format(cats=listing)
    prompts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                        {"role": "user", "content": f"[{r.mode}] {r.summary}"}],
                                       tokenize=False, add_generation_prompt=True) for r in lab.itertuples()]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4))
    idx = []
    for o in outs:
        m = re.search(r"\d+", o.outputs[0].text)
        k = int(m.group(0)) if m else 0
        idx.append(k if 1 <= k <= len(cats) else 0)
    lab["cluster"] = [i - 1 for i in idx]
    lab["name"] = [cats[i - 1]["name"] if i else "unassigned" for i in idx]
    lab.drop(columns=["raw"]).to_parquet(out / "assigned.parquet", index=False)
    info = []
    for i, c in enumerate(cats):
        g = lab[lab.cluster == i]
        info.append({"cluster": i, "name": c["name"], "description": c["definition"], "size": len(g),
                     "share": len(g) / len(lab), "examples": g.sample(min(3, len(g)), random_state=0).summary.tolist()})
    info.sort(key=lambda r: -r["size"])
    (out / "categories_assigned.json").write_text(json.dumps(info, indent=2))
    print(f"unassigned: {(lab.cluster < 0).mean():.2%}")
    for r in info:
        print(f"{r['share']:6.1%}  {r['name']}")


def propose_assign(args):
    assign(args, propose(args))


def add_args(p):
    p.add_argument("stage", choices=["label", "cluster", "name", "propose", "assign", "propose_assign"])
    p.add_argument("scaffold")
    p.add_argument("--model", default=LLM)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k", type=int, default=0, help="force the number of clusters")


def run(args):
    {"label": label, "cluster": cluster, "name": name, "propose": propose, "assign": assign,
     "propose_assign": propose_assign}[args.stage](args)
