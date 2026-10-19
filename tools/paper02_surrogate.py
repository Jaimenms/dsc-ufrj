"""Substitutos do artigo 02 sobre a massa de dados `base2d`: rede neuronal (NN) e rede com informação da física
(PINN), comparadas (abordagem do artigo II do projeto composicional, `research-factory/.../surrogate`).

Entradas: abertura do choke e pressão do gás na plataforma (o poço é o do caso base). Saídas pedidas pelo dono: as
vazões médias de líquido e de gás no topo do riser, a vazão média de gás de gas lift (pela retenção), e o *span*
(máximo − mínimo na janela de captura) das três, que mede a golfada severa. A rede prevê também três pressões médias
auxiliares (fundo do poço, tubing na injeção, base do anular), que são o que liga as saídas à física.

A física do PINN vem das médias dos estados (pedido do dono): num ciclo ou num estacionário, os balanços de massa
médios no tempo são exatos porque são lineares nas médias —
    R1  q̄_L = J_L (P_res − P̄_fundo)                      (IPR linear, líquido)
    R2  q̄_G = J_G (P_res − P̄_fundo) + q̄_inj               (IPR, gás do poço + gás de lift)
    R3  P̄_anular = κ P_gas                                 (coluna estática de gás da plataforma à base do anular)
    R4  q̄_inj = 0,0075 C_V √(ρ_G(P̄) ΔP̄) para ΔP̄ = P̄_anular − P̄_inj > 0, 0 para ΔP̄ < 0   (lei da retenção nas médias; aproximada)
avaliados em pontos de colocação (abertura, P_gas) sorteados no retângulo, sem simular. O gradiente passa pelas
relações, que são fechadas (não há chamada ao simulador: as médias não satisfazem o modelo dinâmico, só os balanços).
O estado final de cada ponto fica disponível para estudos posteriores (resíduo do modelo no estado final), mas não
entra nesta primeira comparação.

    uv run python tools/paper02_surrogate.py --dataset runs/paper02/base2d/dataset.csv --out runs/paper02/surrogate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUTS = ("choke_opening", "P_gas_bar")
OUTPUTS = ("qL_top", "qG_top", "q_inj", "qL_span", "qG_span", "q_inj_span")  # as seis saídas pedidas
AUX = ("P_bh", "P_inj", "P_an")  # médias auxiliares que a física liga às saídas
UNITS = {
    "qL_top": "kg/s",
    "qG_top": "kg/s",
    "q_inj": "kg/s",
    "qL_span": "kg/s",
    "qG_span": "kg/s",
    "q_inj_span": "kg/s",
    "P_bh": "bar",
    "P_inj": "bar",
    "P_an": "bar",
}
# constantes do caso base com gas lift (cases/paper01/gaslift/offshore_gl_P20.json)
P_RES = 200.0  # bar
IPL, IPG = 0.04051419143518518, 0.0012021868055555555  # kg/s/bar
CV = 1.5
RHO_G0, P0_BAR, GAMMA_G = 15.0, 20.0, 1.0  # rho_G = rho_G0 (P/P0)^(1/gamma_G)
KAPPA = 22.77 / 20.0  # base do anular / plataforma, coluna estática de gás (estado inicial do gerador)
RANGES = {"choke_opening": (0.1, 1.0), "P_gas_bar": (20.0, 160.0)}


# ------------------------------------------------------------------------------------------------ dados
def load_table(path, exclude_reasons=("failure",)):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    keep = [r for r in rows if r["reason"] not in exclude_reasons]
    X = np.array([[float(r[k]) for k in INPUTS] for r in keep])
    Y = np.array(
        [
            [
                float(r["riser_top_qL_mean"]),
                float(r["riser_top_qG_mean"]),
                float(r["q_inj_mean"]),
                float(r["riser_top_qL_max"]) - float(r["riser_top_qL_min"]),
                float(r["riser_top_qG_max"]) - float(r["riser_top_qG_min"]),
                float(r["q_inj_max"]) - float(r["q_inj_min"]),
                float(r["well_bottom_P_mean"]),
                float(r["injection_P_mean"]),
                float(r["annulus_base_P_mean"]),
            ]
            for r in keep
        ]
    )
    meta = dict(
        n_rows=len(rows),
        n_used=len(keep),
        reasons={rs: sum(1 for r in rows if r["reason"] == rs) for rs in set(r["reason"] for r in rows)},
    )
    cls = np.array([r["class"] for r in keep])
    return X, Y, cls, meta


def split_indices(n, train_fraction=0.7, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(train_fraction * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def latin_hypercube(n, seed=0):
    rng = np.random.default_rng(seed)
    u = (rng.permutation(n)[:, None] + rng.random((n, 1))) / n
    v = (rng.permutation(n)[:, None] + rng.random((n, 1))) / n
    lo = np.array([RANGES[k][0] for k in INPUTS])
    hi = np.array([RANGES[k][1] for k in INPUTS])
    return lo + np.hstack([u, v]) * (hi - lo)


# ------------------------------------------------------------------------------------------------ rede
def init_params(key, sizes):
    import jax

    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k1, k2 = jax.random.split(key, 3)
        bound = 1.0 / np.sqrt(fan_in)
        params.append(
            (
                jax.random.uniform(k1, (fan_in, fan_out), minval=-bound, maxval=bound),
                jax.random.uniform(k2, (fan_out,), minval=-bound, maxval=bound),
            )
        )
    return params


def forward(params, x):
    import jax.numpy as jnp

    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def physics_residuals(y, x, scale):
    """Resíduos R1–R4 (n, 4) nas unidades físicas divididos por `scale`; `y` (n, 9) físico, `x` (n, 2) físico."""
    import jax.numpy as jnp

    qL, qG, q_inj = y[:, 0], y[:, 1], y[:, 2]
    P_bh, P_inj, P_an = y[:, 6], y[:, 7], y[:, 8]
    P_gas = x[:, 1]
    r1 = qL - IPL * (P_RES - P_bh)
    r2 = qG - IPG * (P_RES - P_bh) - q_inj
    r3 = P_an - KAPPA * P_gas
    dP = P_an - P_inj
    P_mean = jnp.maximum(0.5 * (P_an + P_inj), 1.0)
    rho_G = RHO_G0 * (P_mean / P0_BAR) ** (1.0 / GAMMA_G)
    eps = 1.0  # bar: abertura suave da retenção nas médias
    gate = 0.5 * (1.0 + jnp.tanh(dP / eps))
    q_valve = 0.0075 * CV * jnp.sqrt(rho_G) * gate * jnp.sqrt(jnp.abs(dP) + 1e-3)
    r4 = q_inj - q_valve
    return jnp.stack([r1, r2, r3, r4], axis=1) / jnp.asarray(scale)


def train(X, Y, n_hidden=32, n_layers=2, lr=0.01, epochs=4000, seed=0, lam=0.0, Xc=None, test=None):
    """Adam em lote completo num `lax.scan`; `lam` > 0 liga a física nos pontos de colocação `Xc` (PINN).
    Devolve (modelo dict com pesos numpy e padronização, histórico (épocas, 3): perda de dados, de física, teste)."""
    import jax
    import jax.numpy as jnp

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xm, xs = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    ym, ys = Y.mean(0), np.where(Y.std(0) > 0, Y.std(0), 1.0)
    Xn, Yn = jnp.asarray((X - xm) / xs), jnp.asarray((Y - ym) / ys)
    Xt = Yt = None
    if test is not None:
        Xt, Yt = jnp.asarray((np.asarray(test[0]) - xm) / xs), jnp.asarray((np.asarray(test[1]) - ym) / ys)
    scale = np.array([ys[0], ys[1], ys[8], ys[2]])  # escalas de R1..R4: as das grandezas que comparam
    Xc_phys = jnp.asarray(Xc if Xc is not None else X)
    Xcn = (Xc_phys - jnp.asarray(xm)) / jnp.asarray(xs)
    ymj, ysj = jnp.asarray(ym), jnp.asarray(ys)
    sizes = [X.shape[1]] + [n_hidden] * n_layers + [Y.shape[1]]
    params0 = init_params(jax.random.PRNGKey(seed), sizes)

    def loss_parts(params):
        ld = jnp.mean((forward(params, Xn) - Yn) ** 2)
        y_phys = forward(params, Xcn) * ysj + ymj
        lp = jnp.mean(physics_residuals(y_phys, Xc_phys, scale) ** 2)
        return ld, lp

    def loss(params):
        ld, lp = loss_parts(params)
        return ld + lam * lp

    grad = jax.grad(loss)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def step(carry, t):
        params, m, v = carry
        g = grad(params)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_**2, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree_util.tree_map(lambda p, mh_, vh_: p - lr * mh_ / (jnp.sqrt(vh_) + eps), params, mh, vh)
        ld, lp = loss_parts(params)
        lt = jnp.mean((forward(params, Xt) - Yt) ** 2) if Xt is not None else jnp.nan
        return (params, m, v), jnp.stack([ld, lp, lt])

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params0)

    @jax.jit
    def run(p0):
        (params, _, _), hist = jax.lax.scan(step, (p0, zeros, zeros), jnp.arange(1, epochs + 1, dtype=float))
        return params, hist

    params, hist = run(params0)
    model = dict(
        params=[(np.asarray(W), np.asarray(b)) for W, b in params],
        x_mean=xm,
        x_std=xs,
        y_mean=ym,
        y_std=ys,
        lam=lam,
        n_hidden=n_hidden,
        n_layers=n_layers,
        lr=lr,
        epochs=epochs,
        seed=seed,
    )
    return model, np.asarray(hist)


def predict(model, X):
    X = np.atleast_2d(np.asarray(X, float))
    h = (X - model["x_mean"]) / model["x_std"]
    for W, b in model["params"][:-1]:
        h = np.tanh(h @ W + b)
    W, b = model["params"][-1]
    return (h @ W + b) * model["y_std"] + model["y_mean"]


def r2(y, yp):
    ss_res = ((y - yp) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)


def evaluate(model, X, Y):
    Yp = predict(model, X)
    err = Yp - Y
    return dict(
        r2=r2(Y, Yp).tolist(),
        rmse=np.sqrt((err**2).mean(0)).tolist(),
        max_abs=np.abs(err).max(0).tolist(),
        n=int(len(Y)),
    )


def residual_rms(model, Xc):
    """RMS dos quatro resíduos físicos escalados nos pontos `Xc` (unidades: desvios padrão das grandezas)."""
    import jax.numpy as jnp

    Yp = predict(model, Xc)
    scale = np.array([model["y_std"][0], model["y_std"][1], model["y_std"][8], model["y_std"][2]])
    r = np.asarray(physics_residuals(jnp.asarray(Yp), jnp.asarray(Xc), scale))
    return np.sqrt((r**2).mean(0)).tolist()


def save_model(model, path, names):
    d = dict(
        format="paper02-surrogate-mlp/1",
        inputs=list(INPUTS),
        outputs=list(names),
        hyperparameters={k: model[k] for k in ("n_hidden", "n_layers", "lr", "epochs", "seed", "lam")},
        normalization={k: np.asarray(model[k]).tolist() for k in ("x_mean", "x_std", "y_mean", "y_std")},
        layers=[{"W": W.tolist(), "b": b.tolist()} for W, b in model["params"]],
    )
    Path(path).write_text(json.dumps(d) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ estudo
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(ROOT / "runs" / "paper02" / "base2d" / "dataset.csv"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "paper02" / "surrogate"))
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=1.0, help="peso da física (PINN)")
    ap.add_argument("--n-colloc", type=int, default=200)
    ap.add_argument("--sparse", default="10,20,40", help="tamanhos de treino do estudo com poucos dados")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, Y, cls, meta = load_table(a.dataset)
    names = OUTPUTS + AUX
    tr, te = split_indices(len(X), 0.7, seed=0)
    Xc = latin_hypercube(a.n_colloc, seed=1)
    Xc_test = latin_hypercube(400, seed=2)
    print(
        f"dados: {meta['n_used']} de {meta['n_rows']} pontos ({meta['reasons']}); treino {len(tr)}, teste {len(te)}",
        flush=True,
    )
    results = dict(meta=meta, n_train=int(len(tr)), n_test=int(len(te)), outputs=list(names), full={}, sparse=[])
    kw = dict(n_hidden=a.hidden, n_layers=a.layers, lr=a.lr, epochs=a.epochs)
    # ---- todos os dados de treino: NN e PINN
    for label, lam in (("nn", 0.0), ("pinn", a.lam)):
        t0 = time.time()
        model, hist = train(X[tr], Y[tr], seed=0, lam=lam, Xc=Xc, test=(X[te], Y[te]), **kw)
        ev = evaluate(model, X[te], Y[te])
        res = residual_rms(model, Xc_test)
        results["full"][label] = dict(
            test=ev,
            train=evaluate(model, X[tr], Y[tr]),
            residual_rms=res,
            wall=time.time() - t0,
            history=hist[:: max(1, a.epochs // 200)].tolist(),
        )
        save_model(model, out / f"{label}.json", names)
        np.savez(out / f"{label}_history.npz", history=hist, train_idx=tr, test_idx=te)
        print(
            f"{label:5s}: teste R² "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, ev["r2"]))
            + f" | resíduos {np.round(res, 3).tolist()} | {time.time() - t0:.0f} s",
            flush=True,
        )
        # mapa no plano das decisões
        ga, gp = np.meshgrid(np.linspace(*RANGES["choke_opening"], 60), np.linspace(*RANGES["P_gas_bar"], 60))
        G = np.column_stack([ga.ravel(), gp.ravel()])
        np.savez(out / f"{label}_map.npz", a=ga, P=gp, Y=predict(model, G).reshape(60, 60, -1), names=np.array(names))
    # ---- poucos dados: NN × PINN, várias sementes de divisão
    for n_train in [int(v) for v in a.sparse.split(",")]:
        for seed in range(a.seeds):
            rng = np.random.default_rng(100 + seed)
            sub = np.sort(rng.choice(tr, size=min(n_train, len(tr)), replace=False))
            row = dict(n_train=int(len(sub)), seed=seed)
            for label, lam in (("nn", 0.0), ("pinn", a.lam)):
                model, _ = train(X[sub], Y[sub], seed=seed, lam=lam, Xc=Xc, **kw)
                row[label] = dict(test=evaluate(model, X[te], Y[te]), residual_rms=residual_rms(model, Xc_test))
            results["sparse"].append(row)
            print(
                f"n_train {len(sub):3d} semente {seed}: RMSE teste (qL, qG, q_inj, spans) NN "
                + " ".join(f"{v:.3f}" for v in row["nn"]["test"]["rmse"][:6])
                + " | PINN "
                + " ".join(f"{v:.3f}" for v in row["pinn"]["test"]["rmse"][:6]),
                flush=True,
            )
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    np.savez(out / "data.npz", X=X, Y=Y, cls=cls, train_idx=tr, test_idx=te, names=np.array(names))
    print(f"{out}/results.json")


if __name__ == "__main__":
    main()
