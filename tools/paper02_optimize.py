"""Otimização operacional do artigo 02 sobre o substituto (PINN): escolhe a abertura do choke e a pressão do gás de
lift na plataforma que maximizam a função objetivo (FOB)

    FOB [$/d] = preço do óleo × óleo produzido − custo de compressão do gás de lift

com o óleo tirado da vazão média de líquido no topo do riser (massa de líquido por Sm³ de óleo do caso base) e a
compressão do gás de lift da pressão de sucção (5 bar por omissão) até P_gas, politrópica em estágios com
arrefecimento intermédio (razão por estágio ≤ 4, k = 1,3, T de sucção 313 K, eficiência 0,75), paga ao preço da
energia. Restrições (pedido do dono): a vazão máxima de gás de lift prevista pelo substituto não excede a capacidade do
compressor (`--max-gl`, kg/s) e a vazão máxima de líquido no topo não excede a capacidade do separador
(`--max-oil`, kg/s); ambas usam os máximos que o substituto prevê (não as médias). Resolve por enumeração numa
grelha fina do plano (a, P_gas) sobre o substituto (só nós admissíveis), refina com SLSQP com as restrições e
reporta o ótimo com as saídas previstas e as restrições ativas. Traça também a curva operacional (pedido do dono,
2026-10-20, no lugar da frente de Pareto): para cada pressão do gás fixa, a abertura do choke que maximiza a FOB
entre as admissíveis, com a FOB, a receita e o custo ao longo dela (`operating_curve.npz` e `curve` no results.json).

    uv run python tools/paper02_optimize.py [--model runs/paper02/surrogate/pinn.json] [--oil-price 70]
        [--energy-price 0.10] [--p-suction 5] [--out runs/paper02/optimize]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("paper02_surrogate", ROOT / "tools" / "paper02_surrogate.py")
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

# fluido e poço do caso base (cases/paper01/gaslift/offshore_gl_P20.json, meta.mass)
KG_PER_SM3_OIL = 720.859016  # massa total (óleo + gás dissolvido + gás livre) por Sm³ de óleo
W_GAS = 0.028818101097316372  # fração mássica de gás livre
KG_LIQ_PER_SM3_OIL = KG_PER_SM3_OIL * (1.0 - W_GAS)  # massa de líquido (óleo + gás dissolvido) por Sm³ de óleo
BBL_PER_M3 = 6.28981
R_GAS = 8.314  # J/mol/K
M_GAS = 15.0 * R_GAS * 313.0 / 2.0e6  # kg/mol: rho_G0 = 15 kg/m³ a 20 bar e 313 K → ~19,5 g/mol
K_GAS = 1.3  # expoente politrópico
T_SUCTION = 313.0  # K
ETA = 0.75
MAX_STAGE_RATIO = 4.0


def oil_rate_bbl_d(qL_kg_s):
    """Vazão de óleo em bbl/d a partir da vazão mássica média de líquido (kg/s)."""
    return np.asarray(qL_kg_s) * 86400.0 / KG_LIQ_PER_SM3_OIL * BBL_PER_M3


def compression_power_kw(q_gas_kg_s, P_discharge_bar, P_suction_bar=5.0):
    """Potência de compressão (kW) do gás de lift, politrópica em n estágios iguais com arrefecimento à T de sucção."""
    q = np.asarray(q_gas_kg_s, float)
    ratio = np.maximum(np.asarray(P_discharge_bar, float) / P_suction_bar, 1.0)
    n_st = np.maximum(np.ceil(np.log(ratio) / np.log(MAX_STAGE_RATIO)), 1.0)
    r_st = ratio ** (1.0 / n_st)
    kk = K_GAS / (K_GAS - 1.0)
    w = n_st * kk * (R_GAS / M_GAS) * T_SUCTION * (r_st ** (1.0 / kk) - 1.0) / ETA  # J/kg
    return np.maximum(q, 0.0) * w / 1000.0


def objective(model, A, P, oil_price=70.0, energy_price=0.10, p_suction=5.0, gas_price=0.0):
    """FOB ($/d) e parcelas para vetores de aberturas A e pressões P (mesma forma)."""
    A, P = np.asarray(A, float), np.asarray(P, float)
    Y = sg.predict(model, np.column_stack([A.ravel(), P.ravel()]))
    qL, qG, q_gl = Y[:, sg.IDX["qL_top"]], Y[:, sg.IDX["qG_top"]], Y[:, sg.IDX["q_inj"]]
    revenue = oil_price * oil_rate_bbl_d(qL)
    power = compression_power_kw(q_gl, P.ravel(), p_suction)
    cost = power * 24.0 * energy_price
    gas_sales = gas_price * np.maximum(qG - q_gl, 0.0) * 86400.0  # gás do poço vendido (opcional), $/kg
    fob = revenue - cost + gas_sales
    out = dict(fob=fob, revenue=revenue, cost=cost, power_kw=power, gas_sales=gas_sales, Y=Y)
    return {k: (v.reshape(A.shape) if k != "Y" else v) for k, v in out.items()}


def constraints_values(Y):
    """Máximos previstos que as restrições comparam com as capacidades: (q_gl máx, q_L máx no topo), kg/s."""
    return Y[:, sg.IDX["q_inj_max"]], Y[:, sg.IDX["qL_max"]]


def optimise(model, n_grid=201, max_gl=np.inf, max_oil=np.inf, **prices):
    """Enumeração na grelha (a ∈ [0,1; 1], P_gas ∈ [20; 160]) só nos nós admissíveis (q_gl máx ≤ max_gl, q_L máx
    ≤ max_oil) e refinamento por SLSQP com as duas restrições e as faixas."""
    from scipy.optimize import minimize

    a_lo, a_hi = sg.RANGES["choke_opening"]
    p_lo, p_hi = sg.RANGES["P_gas_bar"]
    A, P = np.meshgrid(np.linspace(a_lo, a_hi, n_grid), np.linspace(p_lo, p_hi, n_grid), indexing="ij")
    res = objective(model, A, P, **prices)
    gl_max, oil_max = constraints_values(res["Y"])
    feasible = (gl_max <= max_gl) & (oil_max <= max_oil)
    if not feasible.any():
        raise SystemExit("nenhum nó da grelha satisfaz as restrições")
    fob_f = np.where(feasible, res["fob"].ravel(), -np.inf)
    k = int(np.argmax(fob_f))
    a0, p0 = float(A.ravel()[k]), float(P.ravel()[k])

    n_calls = [0]

    def f(x):
        n_calls[0] += 1
        return -float(objective(model, np.array([x[0]]), np.array([x[1]]), **prices)["fob"][0]) / 1e5

    def g(x):
        n_calls[0] += 1
        Y = sg.predict(model, np.array([[x[0], x[1]]]))
        gl, oil = constraints_values(Y)
        return np.array(
            [max_gl - gl[0] if np.isfinite(max_gl) else 1.0, max_oil - oil[0] if np.isfinite(max_oil) else 1.0]
        )

    sl = minimize(
        f,
        [a0, p0],
        method="SLSQP",
        bounds=[(a_lo, a_hi), (p_lo, p_hi)],
        constraints=[dict(type="ineq", fun=g)],
        options=dict(ftol=1e-9, maxiter=200),
    )
    a_opt, p_opt = a0, p0
    if sl.success and np.all(g(sl.x) >= -1e-6) and f(sl.x) <= f([a0, p0]):
        a_opt, p_opt = float(np.clip(sl.x[0], a_lo, a_hi)), float(np.clip(sl.x[1], p_lo, p_hi))
    best = objective(model, np.array([a_opt]), np.array([p_opt]), **prices)
    gl_b, oil_b = constraints_values(best["Y"])
    active = []
    if np.isfinite(max_gl) and gl_b[0] >= max_gl - 1e-3 * max(abs(max_gl), 1.0):
        active.append("max_gl")
    if np.isfinite(max_oil) and oil_b[0] >= max_oil - 1e-3 * max(abs(max_oil), 1.0):
        active.append("max_oil")
    return dict(
        a=a_opt,
        P_gas_bar=p_opt,
        fob=float(best["fob"][0]),
        revenue=float(best["revenue"][0]),
        cost=float(best["cost"][0]),
        power_kw=float(best["power_kw"][0]),
        outputs={n: float(v) for n, v in zip(sg.NAMES, best["Y"][0])},
        constraints=dict(
            max_gl=float(max_gl) if np.isfinite(max_gl) else None,
            max_oil=float(max_oil) if np.isfinite(max_oil) else None,
            active=active,
            n_feasible=int(feasible.sum()),
            n_grid=int(feasible.size),
        ),
        grid=dict(
            a=A,
            P=P,
            fob=res["fob"],
            revenue=res["revenue"],
            cost=res["cost"],
            feasible=feasible.reshape(A.shape),
            qL=res["Y"][:, sg.IDX["qL_top"]].reshape(A.shape),
            q_gl=res["Y"][:, sg.IDX["q_inj"]].reshape(A.shape),
            qL_span=res["Y"][:, sg.IDX["qL_span"]].reshape(A.shape),
            qL_max=oil_max.reshape(A.shape),
            q_gl_max=gl_max.reshape(A.shape),
        ),
        grid_best=dict(a=a0, P_gas_bar=p0, fob=float(res["fob"].ravel()[k])),
        n_eval=dict(grid=int(A.size), refine=int(n_calls[0])),
    )


def operating_curve(model, n_p=121, n_a=201, max_gl=np.inf, max_oil=np.inf, **prices):
    """Curva operacional: para cada pressão do gás na plataforma fixa (n_p valores na faixa), a abertura do choke que
    maximiza a FOB entre as admissíveis (q_gl máx ≤ max_gl, q_L máx ≤ max_oil), por enumeração em n_a aberturas e
    refinamento unidimensional entre os nós vizinhos; devolve as pressões, a abertura ótima, a FOB, a receita, o custo,
    a potência e as saídas previstas ao longo da curva (NaN e `feasible` falso onde nenhuma abertura é admissível)."""
    from scipy.optimize import minimize_scalar

    a_lo, a_hi = sg.RANGES["choke_opening"]
    p_lo, p_hi = sg.RANGES["P_gas_bar"]
    Ps, As = np.linspace(p_lo, p_hi, n_p), np.linspace(a_lo, a_hi, n_a)
    A, P = np.meshgrid(As, Ps, indexing="ij")
    res = objective(model, A, P, **prices)
    gl_max, oil_max = constraints_values(res["Y"])
    feasible = ((gl_max <= max_gl) & (oil_max <= max_oil)).reshape(A.shape)
    fob = np.where(feasible, np.asarray(res["fob"]).reshape(A.shape), -np.inf)
    keys = ("a", "fob", "revenue", "cost", "power_kw", "qL", "q_gl", "qL_max", "q_gl_max")
    out = {k: np.full(n_p, np.nan) for k in keys}
    ok = feasible.any(axis=0)
    n_calls = [0]
    for j in range(n_p):
        if not ok[j]:
            continue
        i = int(np.argmax(fob[:, j]))
        a_b, p = float(As[i]), float(Ps[j])

        def f(x):
            n_calls[0] += 1
            return -float(objective(model, np.array([x]), np.array([p]), **prices)["fob"][0])

        def feas(x):
            n_calls[0] += 1
            gl, oil = constraints_values(sg.predict(model, np.array([[x, p]])))
            return bool(gl[0] <= max_gl and oil[0] <= max_oil)

        lo, hi = float(As[max(i - 1, 0)]), float(As[min(i + 1, n_a - 1)])
        if hi > lo:
            r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-4))
            if r.success and feas(float(r.x)) and r.fun <= f(a_b):
                a_b = float(r.x)
        best = objective(model, np.array([a_b]), np.array([p]), **prices)
        gl_b, oil_b = constraints_values(best["Y"])
        out["a"][j], out["fob"][j] = a_b, float(best["fob"][0])
        out["revenue"][j], out["cost"][j], out["power_kw"][j] = (
            float(best["revenue"][0]),
            float(best["cost"][0]),
            float(best["power_kw"][0]),
        )
        out["qL"][j], out["q_gl"][j] = float(best["Y"][0, sg.IDX["qL_top"]]), float(best["Y"][0, sg.IDX["q_inj"]])
        out["qL_max"][j], out["q_gl_max"][j] = float(oil_b[0]), float(gl_b[0])
    out["n_eval"] = np.array([A.size, n_calls[0] + 2 * int(ok.sum())])  # grelha; refinamento + avaliação final
    return dict(P_gas_bar=Ps, feasible=ok, **out)


def timing(model, n_grid=201, n_single=50, **prices):
    """Custo de avaliar o substituto (secção do custo computacional do artigo): segundos de uma avaliação num ponto
    (mediana de n_single chamadas depois do aquecimento) e de uma grelha n_grid por n_grid num só lote."""
    a_lo, a_hi = sg.RANGES["choke_opening"]
    p_lo, p_hi = sg.RANGES["P_gas_bar"]
    objective(model, np.array([0.5]), np.array([100.0]), **prices)  # aquecimento
    ts = []
    for _ in range(n_single):
        t = time.perf_counter()
        objective(model, np.array([0.5]), np.array([100.0]), **prices)
        ts.append(time.perf_counter() - t)
    A, P = np.meshgrid(np.linspace(a_lo, a_hi, n_grid), np.linspace(p_lo, p_hi, n_grid), indexing="ij")
    t = time.perf_counter()
    objective(model, A, P, **prices)
    t_grid = time.perf_counter() - t
    return dict(single_s=float(np.median(ts)), grid_s=float(t_grid), n_grid=int(A.size))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(ROOT / "runs" / "paper02" / "surrogate" / "pinn.json"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "paper02" / "optimize"))
    ap.add_argument("--oil-price", type=float, default=70.0, help="$/bbl")
    ap.add_argument("--energy-price", type=float, default=0.10, help="$/kWh da energia de compressão")
    ap.add_argument("--p-suction", type=float, default=5.0, help="pressão de sucção do compressor [bar]")
    ap.add_argument("--gas-price", type=float, default=0.0, help="$/kg do gás do poço vendido (omissão: não conta)")
    ap.add_argument("--n-grid", type=int, default=201)
    ap.add_argument(
        "--max-gl", type=float, default=np.inf, help="capacidade do compressor: q_gl máximo previsto [kg/s]"
    )
    ap.add_argument(
        "--max-oil", type=float, default=np.inf, help="capacidade do separador: q_L máximo previsto no topo [kg/s]"
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = sg.load_model(a.model)
    # faixas das decisões: as do estudo do substituto (results.json ao lado do modelo), para o ótimo não ficar preso
    # a uma faixa menor do que a dos dados
    res_path = Path(a.model).parent / "results.json"
    if res_path.exists():
        rr = json.load(open(res_path, encoding="utf-8")).get("ranges")
        if rr:
            sg.RANGES.update({k: tuple(v) for k, v in rr.items()})
    prices = dict(oil_price=a.oil_price, energy_price=a.energy_price, p_suction=a.p_suction, gas_price=a.gas_price)
    t0 = time.time()
    r = optimise(model, n_grid=a.n_grid, max_gl=a.max_gl, max_oil=a.max_oil, **prices)
    t_opt = time.time() - t0
    grid = r.pop("grid")
    np.savez(out / "fob_map.npz", **grid)
    t1 = time.time()
    curve = operating_curve(model, max_gl=a.max_gl, max_oil=a.max_oil, **prices)
    t_curve = time.time() - t1
    np.savez(out / "operating_curve.npz", **curve)
    tm = timing(model, n_grid=a.n_grid, **prices)
    r["timing"] = dict(
        optimise_s=t_opt,
        curve_s=t_curve,
        n_eval_optimise=r.pop("n_eval"),
        n_eval_curve=dict(grid=int(curve["n_eval"][0]), refine=int(curve["n_eval"][1])),
        single_eval_s=tm["single_s"],
        grid_eval_s=tm["grid_s"],
        grid_eval_n=tm["n_grid"],
    )
    ok = curve["feasible"]
    r["curve"] = dict(
        P_gas_bar=curve["P_gas_bar"].tolist(),
        a=[None if not np.isfinite(v) else float(v) for v in curve["a"]],
        fob=[None if not np.isfinite(v) else float(v) for v in curve["fob"]],
        P_feasible=[float(curve["P_gas_bar"][ok].min()), float(curve["P_gas_bar"][ok].max())] if ok.any() else None,
    )
    r.update(prices=prices, model=str(a.model), wall=time.time() - t0, n_grid=a.n_grid)
    (out / "results.json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    o = r["outputs"]
    print(
        f"ótimo: abertura {r['a']:.3f}, P_gas {r['P_gas_bar']:.1f} bar | FOB {r['fob']:,.0f} $/d = receita {r['revenue']:,.0f} − "
        f"compressão {r['cost']:,.0f} ({r['power_kw']:.0f} kW)"
    )
    print(
        f"  previsto: q_L {o['qL_top']:.2f} kg/s ({oil_rate_bbl_d(o['qL_top']):,.0f} bbl/d), q_G {o['qG_top']:.2f}, gás de lift "
        f"{o['q_inj']:.2f} kg/s; máximos q_L {o['qL_max']:.1f}, q_gl {o['q_inj_max']:.2f}; spans q_L {o['qL_span']:.1f}, "
        f"q_G {o['qG_span']:.2f}; P_fundo {o['P_bh']:.1f} bar"
    )
    c = r["constraints"]
    print(
        f"  restrições: q_gl máx ≤ {c['max_gl']}, q_L máx ≤ {c['max_oil']}; ativas {c['active'] or 'nenhuma'}; "
        f"nós admissíveis {c['n_feasible']}/{c['n_grid']}"
    )
    print(
        f"  grelha {a.n_grid}×{a.n_grid}: melhor nó ({r['grid_best']['a']:.3f}, {r['grid_best']['P_gas_bar']:.1f}) FOB {r['grid_best']['fob']:,.0f}; {r['wall']:.1f} s"
    )
    if ok.any():
        Pc, ac, fc = curve["P_gas_bar"], curve["a"], curve["fob"]
        print(
            f"  curva operacional (abertura ótima a P_gas fixa): admissível de {Pc[ok].min():.0f} a {Pc[ok].max():.0f} bar"
        )
        tm = r["timing"]
        print(
            f"  custo: otimização {tm['optimise_s']:.2f} s ({tm['n_eval_optimise']['grid']} + {tm['n_eval_optimise']['refine']} avaliações), "
            f"curva {tm['curve_s']:.2f} s ({tm['n_eval_curve']['grid']} + {tm['n_eval_curve']['refine']}); uma avaliação "
            f"{tm['single_eval_s']*1e3:.2f} ms, grelha de {tm['grid_eval_n']} em {tm['grid_eval_s']:.3f} s"
        )
        for p_show in (20.0, 60.0, 100.0, 140.0, 180.0, 220.0, 260.0):
            j = int(np.argmin(np.abs(Pc - p_show)))
            if ok[j]:
                print(
                    f"    P_gas {Pc[j]:5.0f} bar: a* {ac[j]:.2f}, FOB {fc[j]:,.0f} $/d, q_L {curve['qL'][j]:.2f}, "
                    f"pico q_L {curve['qL_max'][j]:.1f}, q_gl {curve['q_gl'][j]:.2f} kg/s"
                )
    print(out / "results.json")


if __name__ == "__main__":
    main()
