from __future__ import annotations

import json

from wsj_embeddings.cli import main


def test_embedding_smoke_is_fixture_safe_and_deterministic(capsys) -> None:
    exit_code = main(["smoke"])
    line = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(line) == {
        "articles": 1,
        "embeddings": 1,
        "validation_ok": True,
    }
    assert line == f"{json.dumps(json.loads(line), sort_keys=True)}\n"
