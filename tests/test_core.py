import json

import numpy as np
import pandas as pd

from canary.features import prefix_rows, step_matrix, STEP_COLS
from canary.metrics import within_task_auc
from canary.policy import Sim, threshold_kills
from canary.turns import classify_shell, parse_openhands, parse_sweagent, path_kind


def swe_traj():
    return [
        {"role": "system", "text": ""},
        {"role": "user", "text": "ISSUE:\nfoo() crashes\n\nINSTRUCTIONS:\nfix it"},
        {"role": "ai", "text": "Reproduce first.\n```\ncreate reproduce.py\n```"},
        {"role": "user", "text": "[File: /r/reproduce.py (1 lines total)]\n(Open file: /r/reproduce.py)\n(Current directory: /r)\nbash-$"},
        {"role": "ai", "text": "Look around.\n```\nfind. -name foo.py\n```"},
        {"role": "user", "text": "/bin/bash: find.: command not found"},
        {"role": "ai", "text": "Open it.\n```\nopen src/foo.py\n```"},
        {"role": "user", "text": "[File: /r/src/foo.py]\n(Open file: /r/src/foo.py)"},
        {"role": "ai", "text": "Fix.\n```\nedit 3:3\n    return 1\nend_of_edit\n```"},
        {"role": "user", "text": "Your proposed edit has introduced new syntax error(s)."},
        {"role": "ai", "text": "Done.\n```\nsubmit\n```"},
    ]


def test_parse_sweagent():
    issue, turns = parse_sweagent(swe_traj())
    assert issue == "foo() crashes"
    assert [t["action"] for t in turns] == ["create", "search", "view", "edit", "submit"]
    assert turns[0]["path_kind"] == "scratch"
    assert turns[1]["malformed"] == 1 and turns[1]["err_notfound"] == 1
    assert turns[3]["path"] == "/r/src/foo.py" and turns[3]["err_edit"] == 1


def test_parse_openhands():
    call = lambda i, name, args: {"id": i, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}
    traj = [
        {"role": "system", "content": "sys", "tool_calls": None},
        {"role": "user", "content": "<issue_description>\nbar is slow\n</issue_description>", "tool_calls": None},
        {"role": "assistant", "content": "run tests", "tool_calls": [call("a", "execute_bash", {"command": "cd /w && python -m pytest -q"})]},
        {"role": "tool", "content": "1 failed, 3 passed\n[Command finished with exit code 1]", "tool_call_id": "a", "name": "execute_bash", "tool_calls": None},
        {"role": "assistant", "content": "", "tool_calls": [call("b", "str_replace_editor", {"command": "str_replace", "path": "/w/bar.py", "old_str": "x", "new_str": "y\nz"})]},
        {"role": "tool", "content": "ERROR:\nNo replacement was performed", "tool_call_id": "b", "name": "str_replace_editor", "tool_calls": None},
        {"role": "assistant", "content": "", "tool_calls": [call("c", "finish", {})]},
    ]
    issue, turns = parse_openhands(traj)
    assert issue == "bar is slow"
    assert [t["action"] for t in turns] == ["test", "edit", "submit"]
    assert turns[0]["err_exit"] == 1 and turns[0]["test_failed"] == 1 and turns[0]["test_passed"] == 3
    assert turns[1]["err_edit"] == 1 and turns[1]["edit_lines"] == 2


def test_classify_and_paths():
    assert classify_shell("cd /x && grep -rn foo .")[0] == "search"
    assert classify_shell("python reproduce.py")[0] == "run"
    assert classify_shell("python tests/test_a.py")[0] == "test"
    assert path_kind("/r/reproduce_bug.py") == "scratch"
    assert path_kind("/r/tests/test_x.py") == "test"
    assert path_kind("/r/src/x.py") == "src"


def test_prefix_rows_only_alive_checkpoints():
    issue, turns = parse_sweagent(swe_traj())
    rows = prefix_rows(turns, issue, checkpoints=[0, 1, 3, 5, 8])
    # 5 turns: the run is still going at k = 0, 1, 3 but has submitted by k = 5.
    assert [r["k"] for r in rows] == [0, 1, 3]
    assert rows[-1]["n_malformed"] == 1 and rows[-1]["n_err"] == 1
    assert not any("target" in r or "exit_status" in r for r in rows)
    assert step_matrix(turns).shape == (5, len(STEP_COLS))


def test_within_task_auc_ignores_task_difficulty():
    # A predictor that only knows task difficulty: high p on the easy task, low on the
    # hard one. Pooled it looks good; within each task it is a coin flip.
    df = pd.DataFrame({
        "instance_id": ["easy"] * 4 + ["hard"] * 4,
        "target": [1, 1, 1, 0, 1, 0, 0, 0],
        "p": [0.9] * 4 + [0.1] * 4,
    })
    auc, pairs = within_task_auc(df)
    assert pairs == 3 + 3 and auc == 0.5
    df["p"] = [0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]
    assert within_task_auc(df)[0] == 1.0


def test_policy_sim():
    runs = pd.DataFrame({"run_id": ["a", "b"], "n_turns": [10, 4], "target": [False, True]})
    costs = {"a": np.ones(10), "b": np.ones(4)}
    sim = Sim(runs, costs)
    r = sim.evaluate({"a": 2})
    assert r["lost"] == 0 and r["saved_turns"] == 8 / 14
    r = sim.evaluate({"a": 2, "b": 1})
    assert r["lost"] == 1.0
    preds = pd.DataFrame({"run_id": ["a", "a", "b"], "k": [1, 2, 1], "p": [0.5, 0.05, 0.6]})
    assert threshold_kills(preds, 0.1) == {"a": 2}
