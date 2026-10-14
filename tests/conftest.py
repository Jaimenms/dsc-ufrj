"""Infraestrutura dos testes golden: carregamento dos ficheiros de referência e comparação
por nível (A = bit a bit pelo hash dos bytes; B = tolerância relativa registada em
policy.json). Marcadores: golden (usa tests/golden), nightly (lento, fora da rotina),
data (precisa de notebooks/data)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"
DATA = Path(__file__).resolve().parents[1] / "notebooks" / "data"


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: compara com tests/golden")
    config.addinivalue_line("markers", "nightly: lento; corre com -m nightly")
    config.addinivalue_line("markers", "data: precisa dos npz de notebooks/data")


def array_hash(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


class Golden:
    """Acesso a um grupo golden (`chapter10.npz`, ...) com asserções por política."""

    def __init__(self, fname):
        self.fname = fname
        path = GOLDEN / fname
        if not path.exists():
            pytest.skip(f"golden {fname} ausente (gerar com tools/golden_generate.py)")
        self.data = np.load(path)
        self.hashes = json.load(open(GOLDEN / "HASHES.json")).get(fname, {})
        self.policy = json.load(open(GOLDEN / "policy.json"))

    def __getitem__(self, key):
        return self.data[key]

    def has(self, key):
        return key in self.data.files

    def keys(self, prefix=""):
        return [k for k in self.data.files if k.startswith(prefix)]

    def assert_equal(self, key, value, what=""):
        pol = self.policy.get(f"{self.fname}:{key}", "A")
        value = np.asarray(value)
        ref = self.data[key]
        assert value.shape == ref.shape, f"{self.fname}:{key} forma {value.shape} != golden {ref.shape} {what}"
        if pol == "A":
            assert value.dtype == ref.dtype, f"{self.fname}:{key} dtype {value.dtype} != golden {ref.dtype}"
            h = array_hash(value)
            if h != self.hashes.get(key):
                dev = float(np.max(np.abs(value.astype(float) - ref.astype(float)))) if value.dtype.kind in "fi" else np.nan
                raise AssertionError(f"{self.fname}:{key} não é bit-idêntico ao golden (nível A){' ' + what if what else ''}; desvio máx. {dev:.3e}")
        else:
            rtol = float(pol.get("rtol", 1e-9)); atol = float(pol.get("atol", 0.0))
            np.testing.assert_allclose(value.astype(float), ref.astype(float), rtol=rtol, atol=atol,
                                       err_msg=f"{self.fname}:{key} (nível B, rtol {rtol}) {what}", equal_nan=True)


@pytest.fixture(scope="session")
def golden():
    return Golden


@pytest.fixture(scope="session")
def data_dir():
    if not DATA.exists() or not any(DATA.glob("*.npz")):
        pytest.skip("notebooks/data ausente")
    return DATA
