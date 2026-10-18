"""
Print one trajectory as a readable transcript.

Examples:
    canary peek            # first failed trajectory
    canary peek --solved   # a solved one, for contrast
    canary peek --steps 8
"""

import re
import textwrap

import pandas as pd

from canary.data import SAMPLE
from canary.trajectory import command_of, reasoning_of


def add_args(p):
    p.add_argument("--solved", action="store_true", help="show a success instead")
    p.add_argument("--steps", type=int, default=6, help="agent turns to print")
    p.add_argument("--width", type=int, default=88)
    p.add_argument("--index", type=int, default=0, help="which matching row to show")
    p.add_argument("--sample", default=str(SAMPLE))


def run(a):
    df = pd.read_parquet(a.sample)
    row = df[df.target == a.solved].iloc[a.index]
    traj = list(row.trajectory)

    print("=" * a.width)
    print(f"TASK        {row.instance_id}")
    print(f"MODEL       {row.model_name}")
    print(f"EXIT STATUS {row.exit_status}")
    print(f"OUTCOME     {'SOLVED' if row.target else 'FAILED'}")
    print(f"LENGTH      {len(traj)} messages")
    print("=" * a.width)

    wrap = lambda s, i: textwrap.fill(
        " ".join(str(s).split()), a.width - 4, initial_indent=i, subsequent_indent="    "
    )

    # The issue the agent was asked to fix.
    first_user = next((s for s in traj if s["role"] == "user"), None)
    if first_user:
        print("\n### THE TASK GIVEN TO THE AGENT\n")
        print(wrap(first_user["text"][:600], "  "))

    print("\n### WHAT THE AGENT DID\n")
    turn = 0
    for step in traj:
        if step["role"] == "system":
            continue
        if step["role"] == "ai":
            turn += 1
            if turn > a.steps:
                break
            cmd = command_of(step["text"])
            print(f"\n--- turn {turn} ---")
            print(wrap(reasoning_of(step["text"])[:300], "  think: "))
            if cmd:
                print(f"  RUN  : {cmd.splitlines()[0][:a.width - 12]}")
        elif turn and turn <= a.steps:
            out = " ".join(str(step["text"]).split())
            # Naive: false-fires when file contents merely contain "error".
            flag = "  <-- ERROR?" if re.search(r"error|traceback|failed", out, re.I) else ""
            print(wrap(out[:260], "  saw  : ") + flag)

    if turn > a.steps:
        print(f"\n  ... {len(traj)} messages total, truncated at {a.steps} turns.")
