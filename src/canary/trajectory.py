"""
Parsing for a single trajectory.

A trajectory is a list of step dicts:
  role="system" -> the standing instructions (text is None; see system_prompt)
  role="ai"     -> the agent thinking, with its command in a ``` fence
  role="user"   -> what the environment printed back (NOT a human)

They alternate ai/user, so "one step" of the agent = one ai + one user pair.
"""

import re

_FENCE = re.compile(r"```(?:\w+\n)?(.*?)```", re.S)


def command_of(text):
    """The agent's action lives in a fenced block inside its message."""
    m = _FENCE.search(text or "")
    return m.group(1).strip() if m else None


def reasoning_of(text):
    """The agent's message with its command block removed."""
    return _FENCE.sub("", text or "")
