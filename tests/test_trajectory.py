from canary.trajectory import command_of, reasoning_of


def test_command_with_language_tag():
    assert command_of("look first\n```bash\nls -la\n```") == "ls -la"


def test_command_without_tag():
    assert command_of("```\nopen foo.py\n```") == "open foo.py"


def test_no_command():
    assert command_of("just thinking") is None
    assert command_of(None) is None


def test_reasoning_strips_fence():
    assert reasoning_of("plan\n```\nls\n```\n").strip() == "plan"
