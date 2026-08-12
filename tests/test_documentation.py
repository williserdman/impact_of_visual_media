from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def _copyable_bash_blocks(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"^```bash\n(.*?)\n```$",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )


def test_readme_has_one_first_time_embedding_command_sequence():
    """Break caught: README publishes competing first-time embedding workflows."""

    text = README.read_text(encoding="utf-8")
    sections = tuple(
        re.finditer(
            r"^### First-time embedding workflow\n(?P<body>.*?)(?=^### |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )

    assert len(sections) == 1
    section = sections[0]
    commands = re.findall(
        r"(?:\.venv/bin/)?wsj-embeddings (smoke|pilot|inventory|run|validate)\b",
        "\n".join(_copyable_bash_blocks(section.group("body"))),
    )
    assert commands == ["smoke", "pilot", "inventory", "run", "validate"]
    outside_section = text[: section.start()] + text[section.end() :]
    assert not re.search(
        r"(?:\.venv/bin/)?wsj-embeddings (pilot|inventory|run|validate)\b",
        "\n".join(_copyable_bash_blocks(outside_section)),
    )


def test_readme_copyable_embedding_runs_use_pilot_returned_model():
    """Break caught: copyable production commands hard-code a hosted model label."""

    run_blocks = tuple(
        block
        for block in _copyable_bash_blocks(README.read_text(encoding="utf-8"))
        if re.search(r"(?:\.venv/bin/)?wsj-embeddings run\b", block)
    )

    assert run_blocks
    for block in run_blocks:
        observed_model = re.search(r"--observed-model\s+([^\s]+)", block)
        assert observed_model is not None
        assert observed_model.group(1) == "'<pilot-returned-model>'"
