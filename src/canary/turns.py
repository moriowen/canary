"""
A scaffold-agnostic view of an agent run: one Turn per action.

SWE-agent and OpenHands look nothing alike on the wire (fenced commands vs.
function calls, footers vs. exit codes), but both reduce to the same loop:
the agent thinks, takes one action, and reads what the environment printed.
Everything downstream (features, models, the monitor) only sees Turns, which
is what makes the cross-scaffold transfer test possible.
"""

import json
import re
import shlex

ACTIONS = ["view", "search", "edit", "create", "run", "test", "install",
           "cleanup", "think", "submit", "other"]

ERR_KINDS = ["traceback", "edit", "notfound", "nomatch", "exception", "exit", "timeout"]

TXT_THOUGHT = 600   # chars kept per turn for text models; stats use full lengths
TXT_CMD = 300
TXT_OUT_HEAD, TXT_OUT_TAIL = 500, 300

_SCRATCH = re.compile(r"(^|/)[^/]*(repro|reproduc|debug|issue|bug|scratch|temp|tmp|verify|check|demo|minimal)[^/]*\.\w+$", re.I)
_TESTPATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")

_RE = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "exception": re.compile(r"^\s*\w*(Error|Exception)\b[:(]", re.M),
    "notfound": re.compile(r"command not found|No such file or directory|does not exist|not found\b", re.I),
    "nomatch": re.compile(r"^No matches found|^No (files|file) found|^Found 0 matches", re.M),
    "timeout": re.compile(r"timed out|TIMEOUT|no new output after", re.I),
}
_PASSED = re.compile(r"(\d+) passed")
_FAILED = re.compile(r"(\d+) (failed|error)")
_EXIT = re.compile(r"\[Command finished with exit code (-?\d+)\]")
_SWE_EDIT_ERR = re.compile(r"Your proposed edit has introduced new syntax error|Error: start_line|Usage: edit")
_SWE_OPEN = re.compile(r"\(Open file: ([^)\n]*)\)")
_FENCE = re.compile(r"```(?:[ \t]*\w+[ \t]*\n|\n)?(.*?)```", re.S)
_CD = re.compile(r"^\s*cd\s+\S+\s*(&&|;)\s*")
# Llama often drops the space before a dot ("find.", "cd..", "create.dvcignore").
# The shell rejects these; the flag keeps that signal, the fix keeps the category.
_GLUED = re.compile(r"^(find|cd|ls|create|rm|touch|open|cat|grep|mkdir)(\.\S*)")


def path_kind(path):
    if not path:
        return ""
    if _SCRATCH.search(path):
        return "scratch"
    if _TESTPATH.search(path):
        return "test"
    return "src"


def _clip(s, n):
    return s if len(s) <= n else s[:n]


def _clip_out(s):
    if len(s) <= TXT_OUT_HEAD + TXT_OUT_TAIL:
        return s
    return s[:TXT_OUT_HEAD] + "\n[...]\n" + s[-TXT_OUT_TAIL:]


def classify_shell(cmd):
    """(action, head, path) for a plain shell command."""
    cmd = _CD.sub("", cmd.strip())
    first = cmd.split("&&")[0].split("|")[0].strip()
    try:
        toks = shlex.split(first, posix=True)
    except ValueError:
        toks = first.split()
    if not toks:
        return "other", "", ""
    head = toks[0].rsplit("/", 1)[-1]
    args = [t for t in toks[1:] if not t.startswith("-")]
    path = next((a for a in args if "." in a or "/" in a), "")
    low = cmd.lower()
    if head in ("pytest", "tox", "nosetests") or "pytest" in low or "unittest" in low:
        return "test", head, path
    if head.startswith("python") or head in ("bash", "sh") or head.startswith("./"):
        script = next((a for a in args if a.endswith((".py", ".sh"))), "")
        if script and path_kind(script) == "test" or re.search(r"(^|/)test\w*\.py", script):
            return "test", head, script
        return "run", head, script
    if head in ("cat", "head", "tail", "less", "more", "sed", "nl", "wc", "diff"):
        return "view", head, path
    if head in ("grep", "rg", "find", "ls", "tree", "ag", "fd", "git", "which", "locate"):
        return "search", head, path
    if head in ("pip", "pip3", "conda", "apt", "apt-get", "poetry", "uv"):
        return "install", head, path
    if head in ("rm", "mv", "cp"):
        return "cleanup", head, path
    if head in ("echo", "printf", "touch", "tee") and ">" in cmd:
        return "create", head, path
    return "other", head, path


def _signals(out, exit_code=None):
    s = {f"err_{k}": 0 for k in ERR_KINDS}
    for k, rx in _RE.items():
        if rx.search(out):
            s[f"err_{k}"] = 1
    if exit_code is not None and exit_code != 0:
        s["err_exit"] = 1
    m = _PASSED.findall(out)
    f = _FAILED.findall(out)
    s["test_passed"] = sum(int(x) for x in m)
    s["test_failed"] = sum(int(x) for x, _ in f)
    return s


def make_turn(t, action, head, cmd, path, thought, out, exit_code=None, edit_lines=0,
              edit_err=False, malformed=0):
    out = out or ""
    thought = thought or ""
    sig = _signals(out, exit_code)
    if edit_err:
        sig["err_edit"] = 1
    return {
        "t": t, "action": action, "head": head[:40],
        "cmd": _clip(cmd.strip(), TXT_CMD), "path": path[:300], "path_kind": path_kind(path),
        "thought": _clip(thought.strip(), TXT_THOUGHT), "output": _clip_out(out.strip()),
        "thought_chars": len(thought), "out_chars": len(out),
        "empty_out": int(len(out.strip()) == 0 or "did not produce any output" in out),
        "edit_lines": edit_lines, "malformed": malformed, **sig,
    }


# ---------------------------------------------------------------- SWE-agent

SWE_ACTIONS = {
    "open": "view", "goto": "view", "scroll_down": "view", "scroll_up": "view",
    "search_dir": "search", "search_file": "search", "find_file": "search",
    "edit": "edit", "create": "create", "submit": "submit",
}


def swe_issue(text):
    m = re.search(r"ISSUE:\n(.*?)\n\s*INSTRUCTIONS:", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()


def parse_sweagent(traj):
    """-> (issue_text, [turn dicts])"""
    msgs = list(traj)
    issue = ""
    turns = []
    open_file = ""
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user" and not issue:
            issue = swe_issue(m["text"])
        if m["role"] != "ai":
            i += 1
            continue
        text = m["text"] or ""
        out = ""
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
            out = msgs[i + 1]["text"] or ""
        fm = _FENCE.search(text)
        cmd = fm.group(1).strip() if fm else ""
        thought = _FENCE.sub("", text) if fm else text
        malformed = 0
        g = _GLUED.match(cmd)
        if g:
            malformed = 1
            cmd = f"{g.group(1)} {cmd[len(g.group(1)):]}"
        toks = cmd.split()
        head = toks[0] if toks else ""
        edit_lines = 0
        edit_err = False
        if head in SWE_ACTIONS:
            action = SWE_ACTIONS[head]
            if head == "open" or head == "create":
                path = toks[1] if len(toks) > 1 else ""
            elif head == "edit":
                path = open_file
                edit_lines = max(0, cmd.count("\n") - 1)
                edit_err = bool(_SWE_EDIT_ERR.search(out))
            elif head in ("search_file", "goto", "scroll_down", "scroll_up"):
                path = open_file
            else:
                path = ""
        elif not cmd:
            action, head, path = "other", "", ""
        else:
            action, head, path = classify_shell(cmd)
        om = _SWE_OPEN.search(out)
        if om and om.group(1).strip() not in ("", "n/a"):
            open_file = om.group(1).strip()
        turns.append(make_turn(len(turns), action, head, cmd, path, thought, out,
                               edit_lines=edit_lines, edit_err=edit_err, malformed=malformed))
        i += 2 if out else 1
    return issue, turns


# ---------------------------------------------------------------- OpenHands

def oh_issue(text):
    m = re.search(r"<issue_description>\n?(.*?)</issue_description>", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()


def parse_openhands(traj):
    msgs = list(traj)
    issue = ""
    turns = []
    outputs = {}
    for m in msgs:
        if m["role"] == "tool" and m.get("tool_call_id"):
            outputs[m["tool_call_id"]] = m["content"] or ""
    for m in msgs:
        if m["role"] == "user" and not issue:
            issue = oh_issue(m["content"])
        if m["role"] != "assistant":
            continue
        calls = m["tool_calls"] if m["tool_calls"] is not None else []
        thought = m["content"] or ""
        for c in calls:
            name = c["function"]["name"]
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {}
            out = outputs.get(c.get("id"), "")
            exit_code = None
            edit_lines = 0
            edit_err = False
            if name == "execute_bash":
                cmd = a.get("command") or ""
                action, head, path = classify_shell(cmd)
                em = _EXIT.search(out)
                exit_code = int(em.group(1)) if em else None
            elif name == "str_replace_editor":
                sub = a.get("command") or ""
                path = a.get("path") or ""
                cmd = f"{sub} {path}"
                head = sub
                action = {"view": "view", "create": "create", "str_replace": "edit",
                          "insert": "edit", "undo_edit": "edit"}.get(sub, "other")
                if action == "edit":
                    edit_lines = (a.get("new_str") or "").count("\n") + 1
                    edit_err = out.lstrip().startswith("ERROR")
                elif action == "create":
                    edit_lines = (a.get("file_text") or "").count("\n") + 1
            elif name in ("think", "task_tracker"):
                action, head, path = "think", name, ""
                cmd = name
                thought = thought + "\n" + (a.get("thought") or "")
            elif name == "finish":
                action, head, path, cmd = "submit", "finish", "", "finish"
            else:
                action, head, path, cmd = "other", name, "", name
            turns.append(make_turn(len(turns), action, head, cmd, path, thought, out,
                                   exit_code=exit_code, edit_lines=edit_lines, edit_err=edit_err))
            thought = ""  # parallel calls share one message's reasoning
    return issue, turns
