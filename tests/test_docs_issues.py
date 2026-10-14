"""O índice de alterações (docs/issues/INDEX.md) e a pasta estão sincronizados: cada ficheiro
NNN-*.md aparece uma vez no índice, cada ligação do índice existe, e cada ficheiro tem a
tabela de cabeçalho (tipo, área, estado) e as secções mínimas."""

import re
from pathlib import Path

ISSUES = Path(__file__).resolve().parents[1] / "docs" / "issues"


def test_index_matches_folder():
    index = (ISSUES / "INDEX.md").read_text(encoding="utf-8")
    files = sorted(p.name for p in ISSUES.glob("[0-9][0-9][0-9]-*.md"))
    linked = re.findall(r"\]\((\d{3}-[^)]+\.md)\)", index)
    assert sorted(linked) == files, set(files) ^ set(linked)
    assert len(linked) == len(set(linked))
    for name in files:
        assert name[:3] == name[:3].zfill(3) and (ISSUES / name).exists()
    numbers = [int(n[:3]) for n in files]
    assert numbers == list(range(1, len(files) + 1))


def test_issue_files_have_header_and_sections():
    for p in sorted(ISSUES.glob("[0-9][0-9][0-9]-*.md")):
        text = p.read_text(encoding="utf-8")
        assert text.startswith(f"# {p.name[:3]} · "), p.name
        for key in ("| tipo |", "| área |", "| estado |", "| onde |"):
            assert key in text, (p.name, key)
        tipo = re.search(r"\| tipo \| (\w+)", text).group(1)
        assert tipo in ("feature", "bugfix", "estudo"), (p.name, tipo)
        assert "## Referências" in text, p.name
        assert ("## Antes" in text and "## Depois" in text) or "## Pergunta" in text, p.name
