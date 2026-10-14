"""Compara os coeficientes F1..F12 do modelo a dois fluidos com os valores
gravados pelo MATLAB nos arquivos .mf2 do capítulo 10."""

import numpy as np
import pytest

from mfsim.fluid import Fluid
from thesis.mf2 import load_mf2, legacy_pipe_outputs
from mfsim.models.base import PipeParams
from mfsim.models.twofluid import TwoFluidModel


def _check_case(fname, regime, itimes=(0, 20, 100, -1), rtol=2e-3):
    raw = load_mf2(fname)
    d = raw["dados"]
    fluid = Fluid.from_legacy(d["fluido"])
    tubo = d["tubo"] if isinstance(d["tubo"], dict) else d["tubo"][0]
    PR = float(d["simulador"]["pressaoR"])
    params = PipeParams(D=float(tubo["diametro"]), eps=float(tubo["rugosidade"]) * 1e-6,
                        PR=PR, regime=regime)
    model = TwoFluidModel(fluid, params)
    out = legacy_pipe_outputs(raw, 0)
    if out is None or "F1" not in out:
        pytest.skip("arquivo sem saidas F1..F12")
    worst = 0.0
    for it in itimes:
        aG = out["alfaG"][:, it]
        PT = out["pressao"][:, it] * 1e5 / PR
        qL = out["vazaoMassaL"][:, it]
        qG = out["vazaoMassaG"][:, it]
        theta = out["inclinacao"][:, it]
        y = np.stack([aG, PT, qL, qG], axis=1)
        F = model.coefficients(y, theta)
        for i in range(12):
            ref = out[f"F{i+1}"][:, it]
            scale = np.maximum(np.abs(ref), 1e-12 * np.max(np.abs(ref)) + 1e-30)
            err = np.max(np.abs(F[i] - ref) / scale)
            worst = max(worst, err)
            assert err < rtol, f"{fname} t_idx={it} F{i+1}: erro relativo {err:.3e}"
    return worst


def test_stratified_10_1_1():
    _check_case("CASO10-1-1.mf2", regime=4)


def test_annular_10_1_2():
    _check_case("CASO10-1-2.mf2", regime=1)


def test_bubble_10_1_3():
    _check_case("CASO10-1-3.mf2", regime=2)


if __name__ == "__main__":
    for f, r in [("CASO10-1-1.mf2", 4), ("CASO10-1-2.mf2", 1), ("CASO10-1-3.mf2", 2)]:
        try:
            print(f, "erro max:", _check_case(f, r))
        except Exception as e:  # noqa
            print(f, "FALHOU:", e)
