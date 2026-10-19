"""Substitutos NN e PINN do artigo 02 (`tools/paper02_surrogate.py`): resíduos físicos, treino numa função suave,
gravação e leitura do modelo; nada simula."""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper02_surrogate", ROOT / "tools" / "paper02_surrogate.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)


def _consistent_rows(n=40, seed=0):
    """Pontos (x, y) que satisfazem R1–R3 exatamente e R4 pela lei da retenção nas médias."""
    rng = np.random.default_rng(seed)
    X = sg.latin_hypercube(n, seed=seed)
    P_bh = 110.0 - 25.0 * X[:, 0] - 0.15 * X[:, 1] + 0.05 * X[:, 0] * X[:, 1]  # função suave das decisões
    P_an = sg.KAPPA * X[:, 1]
    P_inj = P_bh - 2.0
    dP = P_an - P_inj
    P_mean = np.maximum(0.5 * (P_an + P_inj), 1.0)
    rho = sg.RHO_G0 * (P_mean / sg.P0_BAR) ** (1.0 / sg.GAMMA_G)
    gate = 0.5 * (1.0 + np.tanh(dP / 1.0))
    q_inj = 0.0075 * sg.CV * np.sqrt(rho) * gate * np.sqrt(np.abs(dP) + 1e-3)
    qL = sg.IPL * (sg.P_RES - P_bh)
    qG = sg.IPG * (sg.P_RES - P_bh) + q_inj
    spans = np.column_stack([3.0 * X[:, 0] + 0.1 * rng.random(n), 0.02 * X[:, 1], 0.5 * X[:, 0] * (X[:, 1] / 160.0)])
    Y = np.column_stack([qL, qG, q_inj, spans, P_bh, P_inj, P_an])
    return X, Y


def test_physics_residuals_vanish_on_consistent_data():
    import jax.numpy as jnp

    X, Y = _consistent_rows()
    r = np.asarray(sg.physics_residuals(jnp.asarray(Y), jnp.asarray(X), np.ones(4)))
    assert np.abs(r).max() < 1e-4  # float32 do JAX
    Y2 = Y.copy()
    Y2[:, 0] += 0.5  # viola o balanço do líquido
    r2 = np.asarray(sg.physics_residuals(jnp.asarray(Y2), jnp.asarray(X), np.ones(4)))
    assert np.allclose(r2[:, 0], 0.5, atol=1e-4) and np.abs(r2[:, 1:]).max() < 1e-4


def test_train_fits_a_smooth_map_and_roundtrips(tmp_path):
    X, Y = _consistent_rows(n=120, seed=1)
    tr, te = sg.split_indices(len(X), 0.7, seed=0)
    for lam in (0.0, 1.0):
        m, hist = sg.train(
            X[tr],
            Y[tr],
            n_hidden=16,
            n_layers=2,
            lr=0.01,
            epochs=1500,
            seed=0,
            lam=lam,
            Xc=sg.latin_hypercube(50, seed=3),
        )
        ev = sg.evaluate(m, X[te], Y[te])
        assert min(ev["r2"][:3]) > 0.95 and min(ev["r2"][6:]) > 0.95, ev["r2"]
        assert hist.shape == (1500, 3) and hist[-1, 0] < hist[0, 0]
        assert (sg.predict(m, X)[:, 3:6] >= 0).all()  # spans nunca negativos (log1p)
    names = sg.OUTPUTS + sg.AUX
    sg.save_model(m, tmp_path / "m.json", names)
    m2 = sg.load_model(tmp_path / "m.json")
    assert np.allclose(sg.predict(m2, X[te]), sg.predict(m, X[te])) and m2["outputs"] == names
    assert json.loads((tmp_path / "m.json").read_text())["hyperparameters"]["lam"] == 1.0


def test_latin_hypercube_within_ranges():
    Xc = sg.latin_hypercube(64, seed=5)
    assert Xc.shape == (64, 2)
    for j, k in enumerate(sg.INPUTS):
        assert sg.RANGES[k][0] <= Xc[:, j].min() and Xc[:, j].max() <= sg.RANGES[k][1]
    assert len(set(np.floor((Xc[:, 0] - 0.1) / 0.9 * 64).astype(int))) == 64  # um ponto por faixa
