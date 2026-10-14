"""Os dados de entrada versionados como binários (.mf2 legados, PDF) e os dados fora do git
(notebooks/data, scratch) continuam iguais ao MANIFEST.sha256."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _expected():
    p = ROOT / "MANIFEST.sha256"
    if not p.exists():
        pytest.skip("sem MANIFEST.sha256")
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            out[rel] = h
    return out


def test_tracked_inputs_unchanged():
    from manifest import sha256_of

    exp = _expected()
    tracked = {k: v for k, v in exp.items() if k.startswith("legacy/mfsim23/Files/") or k.endswith(".pdf")}
    assert tracked, "manifesto sem entradas legadas"
    for rel, h in tracked.items():
        assert (ROOT / rel).exists(), rel
        assert sha256_of(ROOT / rel) == h, f"{rel} alterado"


@pytest.mark.data
def test_untracked_data_unchanged():
    from manifest import sha256_of

    exp = _expected()
    data = {k: v for k, v in exp.items() if k.startswith("notebooks/data/")}
    present = [rel for rel in data if (ROOT / rel).exists()]
    if not present:
        pytest.skip("notebooks/data ausente")
    changed = [rel for rel in present if sha256_of(ROOT / rel) != data[rel]]
    assert not changed, f"alterados: {changed[:5]}"
