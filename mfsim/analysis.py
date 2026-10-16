"""Análises: Jacobiana densa e autovalores do sistema discretizado."""

from __future__ import annotations

import numpy as np

G = 9.80665


def boe_criterion(vSL, vSG, P_base, rhoL, alphaG, L_gas):
    """Critério de Bøe (1981) na forma citada por Taitel (1986) e Jansen et al. (1996):

        vSL >= P / (rhoL g (1 - alphaG) L) * vSG

    Compara a taxa de subida da coluna hidrostática no riser (rhoL g vSL) com a taxa
    de compressão do gás da linha (P vSG / (alphaG L)); L é o comprimento de linha
    ocupado pelo gás (bolsão + volume a montante), sem fator de inclinação. Uma
    versão anterior desta função multiplicava L por sin|theta|, o que não consta
    das fontes e superestimava vSL_min por 1/sin|theta| (19x a -3 graus).
    Retorna (vSL_min, severe) com severe = vSL >= vSL_min."""
    vSL_min = P_base / (rhoL * G * (1.0 - alphaG) * L_gas) * vSG
    return vSL_min, vSL >= vSL_min


def superficial_velocities(case):
    """(vSL, vSG) à pressão de saída de um caso gerado (duto único; `meta` do gerador)."""
    p = case.pipes[0]
    A = np.pi * p.D**2 / 4
    fl = case.fluid
    P = case.meta["P_out_bar"] * 1e5
    rhoL, rhoG, _, _ = fl.props(P)
    return case.meta["qL"] / rhoL / A, case.meta["qG"] / rhoG / A


def dense_jacobian(model, X, t=0.0, rel=1e-7, abs_=1e-9):
    X = np.asarray(X, float)
    f0 = model.rhs(t, X)
    n = X.size
    J = np.empty((n, n))
    for j in range(n):
        h = rel * abs(X[j]) + abs_
        xp = X.copy()
        xp[j] += h
        J[:, j] = (model.rhs(t, xp) - f0) / h
    return J


def jacobian_eigenvalues(model, X, t=0.0):
    """Autovalores da Jacobiana dX/dt em (t, X), ordenados por parte real decrescente."""
    J = dense_jacobian(model, X, t)
    ev = np.linalg.eigvals(J)
    return ev[np.argsort(-ev.real)]


def stability_report(model, X, t=0.0, tol=1e-9):
    ev = jacobian_eigenvalues(model, X, t)
    unstable = ev[ev.real > tol]
    return dict(
        eigenvalues=ev,
        n_unstable=int(unstable.size),
        max_real=float(ev.real.max()),
        dominant=unstable[:5],
        period=(2 * np.pi / abs(unstable[0].imag) if unstable.size and unstable[0].imag != 0 else None),
    )


def mass_balance(res, k=0, phase="L"):
    """Balanço de massa de uma fase num duto: estoque(t) - estoque(0) e integral
    acumulada de (entrada - saída) [kg].

    Usa os fluxos de massa efetivos nas faces de contorno: no modo conservativo do
    esquema fv_upwind são as vazões vezes o limitador da célula doadora (os mesmos
    fluxos das equações de massa); nos demais, as próprias vazões das faces.
    Devolve (mass, net, err) com err = (mass - mass[0]) - net."""
    model = res.model
    pb = model.pipes[k]
    p = res.pipe(k)
    T = res.T
    Y = p["Y"]  # (nt, n_nodes, 4), primitivas
    sch = pb.scheme
    A = np.pi * pb.pipe.D**2 / 4
    h = pb.mesh["h"]
    if sch.name.startswith("fv_"):
        cells = slice(1, sch.N + 1)
        P = Y[:, cells, 1] * model.PR
        aG = np.clip(Y[:, cells, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = model.fluid.props(P)
        rho_a = rhoG * aG if phase == "G" else rhoL * (1.0 - aG)
        mass = np.sum(rho_a * A * h[None, :], axis=1)
        if getattr(sch, "conservative", False):
            F = np.array([sch.mass_fluxes_nodal(Y[it])[0 if phase == "L" else 1] for it in range(T.size)])
            qin, qout = F[:, 0], F[:, -1]
        else:
            slot = 2 if phase == "L" else 3
            qin, qout = Y[:, 0, slot], Y[:, sch.N, slot]
    else:
        # esquemas nodais: regra do trapézio nos nós
        P = Y[:, :, 1] * model.PR
        aG = np.clip(Y[:, :, 0], 0.0, 1.0)
        rhoL, rhoG, _, _ = model.fluid.props(P)
        rho_a = rhoG * aG if phase == "G" else rhoL * (1.0 - aG)
        mass = np.sum(0.5 * (rho_a[:, 1:] + rho_a[:, :-1]) * A * h[None, :], axis=1)
        slot = 2 if phase == "L" else 3
        qin, qout = Y[:, 0, slot], Y[:, -1, slot]
    dT = np.diff(T)
    net = np.concatenate([[0.0], np.cumsum(0.5 * ((qin[1:] - qout[1:]) + (qin[:-1] - qout[:-1])) * dT)])
    return mass, net, (mass - mass[0]) - net


def blowdowns(T, P_in, min_gap=200.0):
    """Instantes dos blowdowns de uma série de pressão na entrada: mínimos locais abaixo de
    30 % da faixa, separados por pelo menos `min_gap` s (um blowdown tem mínimos
    secundários nos segundos seguintes). Mais robusto do que contar picos de vazão.
    (Era `blowdowns` no notebook nb_severe_slug e em scratch/cycle_stats.py.)"""
    T = np.asarray(T, float)
    P_in = np.asarray(P_in, float)
    lvl = P_in.min() + 0.3 * (P_in.max() - P_in.min())
    low = P_in < lvl
    out = []
    i = 0
    while i < T.size:
        if low[i]:
            j = i
            while j < T.size and low[j]:
                j += 1
            k = i + int(np.argmin(P_in[i:j]))
            if not out or T[k] - out[-1] > min_gap:
                out.append(T[k])
            i = j
        else:
            i += 1
    return np.array(out)


def cycle_signature(T, P_in, q_out, min_gap=200.0, t_skip=0.0):
    """Assinatura de um ciclo de intermitência severa (para comparações tolerantes ao
    caos): instante do primeiro blowdown, intervalo médio, número de blowdowns, P_in
    mínima e máxima, pico de q_out depois de t_skip. P_in em bar, q_out em kg/s."""
    T = np.asarray(T, float)
    P_in = np.asarray(P_in, float)
    q_out = np.asarray(q_out, float)
    tb = blowdowns(T, P_in, min_gap)
    sel = T >= t_skip
    return dict(
        t_first=float(tb[0]) if tb.size else np.nan,
        interval=float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan,
        n_blowdowns=int(tb.size),
        P_min=float(P_in[sel].min()),
        P_max=float(P_in[sel].max()),
        q_peak=float(q_out[sel].max()),
    )


def gas_front_widths(x, aG, ib):
    """Largura (m) da frente de gás no riser em cada instante em que há gás acima da base
    (fração > 0.5 em algum ponto e < 0.05 noutro): distância entre os pontos em que a
    fração cruza 0.1 e 0.9. (Era parte de scratch/cycle_stats.py.)"""
    widths = []
    xs = x[ib:]
    for it in range(aG.shape[1]):
        seg = aG[ib:, it]
        if seg.max() > 0.5 and seg.min() < 0.05:
            j1 = np.argmax(seg > 0.1)
            j9 = np.argmax(seg > 0.9) if (seg > 0.9).any() else len(seg) - 1
            if j9 > j1:
                widths.append(xs[j9] - xs[j1])
    return np.array(widths)


def cycle_stats(run, t_skip=100.0, min_gap=200.0):
    """Estatísticas de ciclo de uma corrida gravada do severe slug (partida bloqueada, duto
    único): instantes dos blowdowns, intervalo médio, pico e média da vazão de líquido na
    saída, faixa da pressão de entrada (bar) depois de `t_skip`, largura mediana da frente
    de gás no riser, passos e CPU do integrador. (Era scratch/cycle_stats.py.)"""
    p = run.pipe(0)
    T = run.T
    P1 = p["out"]["P_bar"][1]
    q = p["out"]["qL"][-1]
    aG = p["out"]["alphaG"]
    x = p["x"]
    tb = blowdowns(T, P1, min_gap)
    per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    sel = T > t_skip
    widths = gas_front_widths(x, aG, run.ib)
    return dict(
        blowdowns=tb,
        interval=per,
        q_peak=float(q[sel].max()),
        q_mean=float(q[sel].mean()),
        P_min=float(P1[sel].min()),
        P_max=float(P1[sel].max()),
        front_width=float(np.median(widths)) if widths.size else np.nan,
        n_steps=run.meta.get("n_steps", run.info.get("n_steps")),
        cpu=run.meta.get("cpu", run.info.get("cpu")),
    )


CYCLE_STATS_HEADER = (
    f"{'trajetória':48s} {'passos':>7s} {'CPU':>6s} {'blowdowns':>24s} {'interv.':>7s} {'pico':>6s} "
    f"{'P_in':>11s} {'prod.':>6s} {'frente':>6s}"
)


def format_cycle_stats(name, st):
    """Linha da tabela de `cycle_stats` (formato de scratch/cycle_stats.py)."""
    short = name.replace("severe_slug_blocked_", "")
    n_steps = int(st["n_steps"]) if st["n_steps"] is not None else -1
    cpu = float(st["cpu"]) if st["cpu"] is not None else np.nan
    tb = np.asarray(st["blowdowns"], float)
    return (
        f"{short:48s} {n_steps:7d} {cpu:5.0f}s {str(tb.round(0).astype(int).tolist()):>24s} {st['interval']:6.0f}s "
        f"{st['q_peak']:5.2f} {st['P_min']:5.2f}-{st['P_max']:4.2f} {st['q_mean']:6.3f} {st['front_width']:5.1f}m"
    )


# ----------------------------------------------------------------------------- varredura (era tools/scan_summary.py)
SCAN_HEADER = (
    f"{'trajetória':34s} {'vSL':>5s} {'vSG':>5s} {'P_out':>5s} {'L':>5s} {'V_up':>5s} {'Bøe mín':>8s} {'linha (IPE)':>12s} "
    f"{'passos':>7s} {'ΔP_in':>6s} {'blowdowns':>24s} {'interv.':>8s} {'pico':>6s}"
)
_REGIME_SHORT = {1: "anular", 2: "bolhas", 3: "intermit.", 4: "estrat.", 5: "estrat.ond."}


def scan_summary(run, t_skip=300.0):
    """Resumo de uma corrida da varredura da fronteira de intermitência severa: vazões,
    pressão, critério de Bøe, padrão da linha pela identificação da tese (estado médio da
    célula no meio do trecho descendente), amplitude da pressão de entrada depois de
    `t_skip`, blowdowns e intervalo."""
    from .closures import equilibrium_fractions, identify_regime

    mb = run.meta
    fluid = run.case.fluid
    D = run.case.pipes[0].D
    A = np.pi * D**2 / 4
    P_out = mb.get("P_out", 4.0)
    vSL = mb.get("vSL", 0.3)
    L = mb.get("L_down", 300.0)
    H = mb.get("H_riser", 20.0)
    qG = mb.get("vSG", 0.1) * fluid.rhoG((P_out + 1) * 1e5) * A
    model = run.model
    p = run.pipe(0)
    T, X = run.T, run.X
    P1 = p["out"]["P_bar"][1]
    qout = p["out"]["qL"][-1]
    sel = T > t_skip
    amp = P1[sel].max() - P1[sel].min()
    tb = blowdowns(T[sel], P1[sel]) if amp > 0.3 else np.array([])
    per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    # Bøe com a pressão média na base do riser e vSG à pressão do separador
    P_base = P_out * 1e5 + fluid.rhoL0 * 9.81 * H
    vSG_out = qG / (fluid.rhoG(P_out * 1e5) * A)
    vmin, severe = boe_criterion(
        vSL, vSG_out, P_base, fluid.rhoL0, 0.5, mb.get("L_h1", 20.0) + L
    )  # L de gás = entrada + descendente
    # padrão da linha: IPE da tese no estado médio (fração e vazões) de uma célula do descendente
    pb = model.pipes[0]
    sch, mod = pb.scheme, pb.model
    it = np.flatnonzero(sel)[::10]
    Ymean = np.mean([model.assemble(T[i], X[:, i])[1][0] for i in it], axis=0)
    inp, _, _ = mod.local_state(Ymean, sch.theta_nodes)
    fl, _ = identify_regime(inp, k_switch=1e6, eq=equilibrium_fractions(inp))
    th_ = np.rad2deg(sch.theta_nodes)
    down = np.flatnonzero(np.abs(th_ - mb.get("slope", -3.0)) < 0.5)
    k = int(down[len(down) // 2])
    line = _REGIME_SHORT.get(int(round(float(fl.reg2[k]))), f"{float(fl.reg2[k]):.1f}")
    return dict(
        vSL=vSL,
        vSG_out=vSG_out,
        P_out=P_out,
        L=L,
        V_up=mb.get("upstream_volume", 0.0),
        boe_vmin=vmin,
        boe_severe=bool(severe),
        line=line,
        n_steps=mb["n_steps"],
        amp=amp,
        blowdowns=tb,
        interval=per,
        q_peak=float(qout[sel].max()),
    )


def format_scan(name, st):
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    return (
        f"{short:34s} {st['vSL']:5.2f} {st['vSG_out']:5.2f} {st['P_out']:5.1f} {st['L']:5.0f} {st['V_up']:5.1f} {st['boe_vmin']:8.2f} "
        f"{st['line']:>12s} {st['n_steps']:7d} {st['amp']:6.2f} {str(np.round(st['blowdowns']).astype(int).tolist()):>24s} "
        f"{st['interval']:7.0f}s {st['q_peak']:6.2f}"
    )


# ----------------------------------------------------------------------------- filme no riser (era tools/riser_cc.py)
def riser_countercurrent(run):
    """Filme no riser de uma corrida (partida bloqueada, duto único): fração do tempo com o
    riser quase cheio de gás (> 90 %), e nessas células a fração em contracorrente (filme
    descendo, gás subindo), com líquido subindo depressa, e as velocidades médias."""
    model = run.model
    p = run.pipe(0)
    th = np.rad2deg(model.pipes[0].scheme.theta_nodes)
    riser = np.flatnonzero(th > 80)
    aG = p["out"]["alphaG"][riser]
    vL = p["out"]["velocidadeL"][riser]
    vG = p["out"]["velocidadeG"][riser]
    gasfull = aG > 0.9
    cc = gasfull & (vL < -0.02) & (vG > 0.02)
    up = gasfull & (vL > 0.5)
    n = max(1, gasfull.sum())
    return dict(
        gasfull=float(gasfull.mean()),
        countercurrent=cc.sum() / n,
        liquid_up=up.sum() / n,
        vL_mean=float(vL[gasfull].mean()) if gasfull.any() else np.nan,
        vG_mean=float(vG[gasfull].mean()) if gasfull.any() else np.nan,
    )


def format_riser(name, st):
    short = name.replace("severe_slug_blocked_conservative_arc_", "")
    return (
        f"{short:28s}: riser com >90% gás {st['gasfull']:5.1%} do tempo | nessas células: filme descendo {st['countercurrent']:5.1%}, "
        f"líquido subindo > 0.5 m/s {st['liquid_up']:5.1%} | vL médio {st['vL_mean']:+.2f} m/s, vG médio {st['vG_mean']:+.2f} m/s"
    )


# ----------------------------------------------------------------------------- validação de laboratório (era tools/val_summary.py)
# referências experimentais: período (s), pressão na linha mín-máx (kPa abs), fonte
LAB_REF = {
    "fabre2": dict(
        periodo=137.0,
        P_min=126.0,
        P_max=235.0,
        fonte="Fabre et al. 1990 caso 2 (Tengesdal 2002, fig. 3-5): vSL 0.13, vSG0 0.20",
    ),
    "fabre1": dict(
        periodo=70.0,
        P_min=128.0,
        P_max=210.0,
        fonte="Fabre et al. 1990 caso 1 (Tengesdal 2002, fig. 3-3): vSL 0.13, vSG0 0.45",
    ),
    "jansen": dict(
        periodo=46.6,
        P_min=111.0,
        P_max=133.0,
        fonte="Jansen et al. 1996 tab. 1 (choke C = 120 kPa s2/m2): ULS 0.0959, UGSO 0.0753",
    ),
    "teng1": dict(
        periodo=79.0,
        P_min=121.0,
        P_max=222.0,
        fonte="Tengesdal 2002 teste 1 (fig. 4-12/4-13, -1 grau): vSL 0.32, vSG 0.35",
    ),
}


def lab_reference(tag):
    """Referência experimental cujo nome aparece em `tag` (ou None)."""
    key = next((k for k in LAB_REF if k in tag), None)
    return key, LAB_REF.get(key, {})


def lab_validation(run, tag):
    """Assinatura de uma corrida de validação contra os valores publicados: período (blowdowns
    separados por 30 % do período medido), pressão de entrada em kPa, pico de líquido, fração
    de gás na base do riser e critério de Bøe com a fração média da linha."""
    case = run.case
    fluid = case.fluid
    m = case.meta
    A = np.pi * case.pipes[0].D ** 2 / 4
    qL, qG = m["qL"], m["qG"]
    T = run.T
    p = run.pipe(0)
    P1 = p["out"]["P_bar"][1] * 100.0
    q = p["out"]["qL"][-1]
    x = p["x"]
    aG = p["out"]["alphaG"]
    key, ref = lab_reference(tag)
    per_ref = ref.get("periodo", 100.0)
    sel = T > 0.2 * T[-1]
    tb = blowdowns(T[sel], P1[sel], 0.3 * per_ref)
    per = float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan
    P_base = m["P_out_bar"] * 1e5 + fluid.rhoL0 * 9.81 * m["H_riser"]
    vSL = qL / fluid.rhoL0 / A
    vSG_out = qG / fluid.rhoG(m["P_out_bar"] * 1e5) / A
    gp = run.meta
    vmin, sev = boe_criterion(
        vSL,
        vSG_out,
        P_base,
        fluid.rhoL0,
        1 - float(np.mean(1 - aG[1 : gp.get("n_h1", 2) + gp["n_down"], sel])),
        m["L_h1"] + m["L_down"],
    )
    ib = int(np.argmin(np.abs(x - m["s_elbow0"])))
    return dict(
        tag=tag,
        ref=ref,
        n_steps=gp.get("n_steps"),
        blowdowns=tb,
        interval=per,
        P_min=float(P1[sel].min()),
        P_max=float(P1[sel].max()),
        q_peak=float(q[sel].max()),
        q_feed=qL,
        aG_base_min=float(aG[ib + 1, sel].min()),
        aG_base_max=float(aG[ib + 1, sel].max()),
        boe_vmin=vmin,
        boe_severe=bool(sev),
    )


def format_lab(st):
    ref = st["ref"]
    n = st["n_steps"] if st["n_steps"] is not None else "?"
    return (
        f"{st['tag']:18s} passos {n:>6} | blowdowns {len(st['blowdowns'])} | período {st['interval']:6.1f} s (exp {ref.get('periodo', np.nan):6.1f}) "
        f"| P_in {st['P_min']:6.1f}-{st['P_max']:6.1f} kPa (exp {ref.get('P_min', np.nan):.0f}-{ref.get('P_max', np.nan):.0f}) "
        f"| pico líquido {st['q_peak']:.3f} kg/s ({st['q_peak']/st['q_feed']:.1f}x) | αG base riser {st['aG_base_min']:.2f}-{st['aG_base_max']:.2f} "
        f"| Bøe vSL_min {st['boe_vmin']:.3f} ({'sim' if st['boe_severe'] else 'não'})"
    )


# ----------------------------------------------------------------------------- classe do ciclo (issue 043)
CYCLE_CLASSES = ("stable", "cycling", "slugging")


def cycle_class(T, P_in, q_out, amp_stable=0.1, amp_rel=0.01, surge=2.0):
    """Classe de uma janela de corrida a partir da pressão na entrada da linha (bar) e da vazão
    de líquido na saída (kg/s):

    - "stable": amplitude da pressão menor do que `amp_stable` bar ou do que `amp_rel` da
      pressão média (o que for maior), ou sem ciclo na janela (no máximo um cruzamento da média:
      uma deriva que não chegou a ciclar conta como estável);
    - "slugging": golfada severa, pico da vazão de saída de pelo menos `surge` vezes a média;
    - "cycling": ciclo de pressão com a saída suave (o choke absorve o pico).

    Devolve a classe e a assinatura: pressão mínima, média e máxima, amplitude, período
    (entre cruzamentos ascendentes, `mfsim.simulation.window_signature`), vazão média e de pico
    e a razão pico/média."""
    from .simulation import window_signature

    T, P_in, q_out = (np.asarray(a, float) for a in (T, P_in, q_out))
    w = window_signature(T, P_in)
    q_mean, q_peak = float(np.mean(q_out)), float(np.max(q_out))
    ratio = q_peak / q_mean if q_mean > 0 else float("inf")
    if w["amp"] < max(amp_stable, amp_rel * abs(w["mean"])) or w["crossings"] <= 1:
        cls = "stable"
    elif ratio >= surge:
        cls = "slugging"
    else:
        cls = "cycling"
    return dict(
        cls=cls,
        P_min=float(np.min(P_in)),
        P_mean=w["mean"],
        P_max=float(np.max(P_in)),
        amp=w["amp"],
        period=w["period"] if cls != "stable" else float("nan"),
        crossings=w["crossings"],
        q_mean=q_mean,
        q_peak=q_peak,
        surge=ratio,
    )


def run_cycle_class(run, window=None, pipe=0, **kw):
    """`cycle_class` de uma corrida gravada (`mfsim.runs.Run`). Pressão no nó 1 do duto, vazão de
    líquido no último nó.

    Com a paragem adaptativa (`info["settle"]`) e pelo menos duas janelas, o registo depois do
    transitório (todas as janelas menos a primeira) decide se houve ciclo e dá o período, os picos e
    as vazões (ciclos longos, de uma golfada por janela ou menos, só aparecem aí); a amplitude da
    última janela decide se a corrida terminou permanente. Sem paragem adaptativa, ou com uma só
    janela, usa a janela final (`window` segundos; omissão: a segunda metade)."""
    p = run.pipe(pipe)
    T = run.T
    P1, q = p["out"]["P_bar"][1], p["out"]["qL"][-1]
    ws = ((run.info or {}).get("settle") or {}).get("windows") or []
    if window is None:
        if ws:
            window = ws[-1]["t_end"] - (ws[-2]["t_end"] if len(ws) >= 2 else T[0])
        else:
            window = 0.5 * (T[-1] - T[0])
    last = T >= T[-1] - window - 1e-9
    out = cycle_class(T[last], P1[last], q[last], **kw)
    if len(ws) >= 2:
        long = T >= ws[0]["t_end"] - 1e-9
        c = cycle_class(T[long], P1[long], q[long], **kw)
        amp_ok = out["amp"] < max(kw.get("amp_stable", 0.1), kw.get("amp_rel", 0.01) * abs(out["P_mean"]))
        if amp_ok or c["cls"] == "stable":
            out["cls"] = "stable"
            out["period"] = float("nan")
        else:
            out = dict(c, amp=out["amp"], P_mean=out["P_mean"])
    out.update(t_end=float(T[-1]), window=float(window))
    return out


def jansen_choke_dp_crit(alpha=0.74, alpha_p=0.6, L=320.0, H=20.0, P_S=4e5, Phi=1.0, rhoL=900.0, g=9.81):
    """Queda no choke (bar, relação monofásica do líquido, Δp = C·U_LS²) acima da qual o critério
    de Jansen, Shoham e Taitel (1996) prevê estabilidade do riser: eq. (5) com Phi = 1 (o gás não
    acelera o tampão, sem blowout) e eq. (10) com Phi a fração de líquido média do riser em
    permanente. alpha: fração de gás da linha; alpha_p: fração de gás da frente que entra no riser
    (0,5–0,75 nos ensaios); L: comprimento da linha com gás; H: altura do riser; P_S: pressão do
    separador (Pa). Com K = 2 C U_LS²/H, a condição dá C U_LS² (1 + 2 alpha L/(alpha_p H)) >
    Phi rhoL g (alpha L/alpha_p - H) - P_S."""
    aL = alpha * L / alpha_p
    return (Phi * rhoL * g * (aL - H) - P_S) / (1.0 + 2.0 * aL / H) / 1e5


def jansen_choke_opening(ab, q_liq, dp_crit, CV, rhoL=900.0):
    """Abertura em que a queda monofásica do líquido no choke (dp = q²/rhoL/(0,0075 CV ab)², em bar,
    a relação do choke do mfsim em líquido puro) iguala `dp_crit`, interpolada em log numa série de
    aberturas `ab` com as vazões médias `q_liq` (kg/s); None se a série não cruza o limiar."""
    ab, q_liq = np.asarray(ab, float), np.asarray(q_liq, float)
    o = np.argsort(ab)
    ab, q_liq = ab[o], q_liq[o]
    dpl = q_liq**2 / rhoL / (0.0075 * CV * ab) ** 2
    if not (dpl.max() > dp_crit > dpl.min()):
        return None
    return float(np.exp(np.interp(-np.log(dp_crit), -np.log(dpl), np.log(ab))))


# ----------------------------------------------------------------------------- diagnósticos de trajetória (artigo 01)
PATTERN_FLAGS = (("stratified", "Est"), ("intermittent", "Int"), ("bubble", "Bub"), ("annular", "Anu"))


def cell_states(run, pipe=0, every=1):
    """Estados nas células de um duto do esquema escalonado ao longo de uma corrida gravada: dict com `T` (nt,),
    `s` e `theta` dos centros das células (N,) e `Y` (nt, N, 4) = (alphaG, PT, qL, qG) com as vazões médias das
    duas faces da célula (as mesmas entradas com que o esquema avalia os fechamentos nas células)."""
    model = run.model
    pb = model.pipes[pipe]
    sch = pb.scheme
    N = sch.N
    Yn = run.pipe(pipe)["Y"][::every]  # (nt, N + 2, 4) primitivas
    Y = Yn[:, 1 : N + 1].copy()
    Y[:, :, 2] = 0.5 * (Yn[:, 0:N, 2] + Yn[:, 1 : N + 1, 2])
    Y[:, :, 3] = 0.5 * (Yn[:, 0:N, 3] + Yn[:, 1 : N + 1, 3])
    return dict(
        T=np.asarray(run.T)[::every], s=np.asarray(sch.x_cells), theta=np.asarray(sch.theta_nodes[1 : N + 1]), Y=Y
    )


def flow_pattern_weights(run, pipe=0, every=1, cells=None):
    """Pesos dos quatro fechamentos (estratificado, intermitente, bolhas, anular) em cada célula e instante, com
    a política de padrão do caso (fixa com misturas ou identificação automática): dict com `T`, `s`, `theta` e
    `W` (nt, N, 4) na ordem de `PATTERN_FLAGS`; o padrão dominante é `W.argmax(-1)`."""
    cs = cells or cell_states(run, pipe, every)
    mod = run.model.pipes[pipe].model
    p = mod.params
    W = np.zeros(cs["Y"].shape[:2] + (len(PATTERN_FLAGS),))
    for it, Y in enumerate(cs["Y"]):
        Y = Y.copy()
        Y[:, 0] = np.clip(Y[:, 0], p.alpha_min, p.alpha_max)
        inp, _, _ = mod.local_state(Y, cs["theta"])
        _, flags, _ = mod.closures(inp)
        W[it] = np.stack([np.asarray(getattr(flags, f), float) for _, f in PATTERN_FLAGS], axis=1)
    return dict(T=cs["T"], s=cs["s"], theta=cs["theta"], W=W)


def characteristic_imag(run, pipe=0, every=1, cells=None, **params):
    """max |Im λ| (m/s) das velocidades características do modelo em cada célula e instante (0 = hiperbólico),
    com parâmetros do modelo opcionalmente trocados (`virtual_mass`, `low_void_regime`, `interfacial_pressure`,
    ...) para avaliar, nos mesmos estados, o que dariam outras regularizações. Devolve (nt, N)."""
    from dataclasses import replace

    cs = cells or cell_states(run, pipe, every)
    mod = run.model.pipes[pipe].model
    m2 = type(mod)(mod.fluid, replace(mod.params, **params)) if params else mod
    p = m2.params
    out = np.zeros(cs["Y"].shape[:2])
    for it, Y in enumerate(cs["Y"]):
        Y = Y.copy()
        Y[:, 0] = np.clip(Y[:, 0], p.alpha_min, p.alpha_max)
        out[it] = np.abs(m2.eigenvalues(Y, cs["theta"]).imag).max(axis=1)
    return out


def _crossing(s, a, thr):
    """Primeira posição (interpolada entre centros de células) em que a série `a(s)` passa a `thr` ou mais;
    `s[-1]` se nunca passa."""
    k = np.flatnonzero(a >= thr)
    if k.size == 0:
        return float(s[-1])
    j = int(k[0])
    if j == 0:
        return float(s[0])
    return float(s[j - 1] + (thr - a[j - 1]) * (s[j] - s[j - 1]) / (a[j] - a[j - 1]))


def slug_fronts(run, pipe=0, every=1, thr=0.5, cells=None):
    """Frentes da golfada severa ao longo de uma corrida (duto único com descendente, curva e riser): a partir
    do ponto mais baixo do traçado, o comprimento de duto cheio de líquido a montante (`seal`: até onde o
    tampão penetra no descendente, fração de gás < `thr`), a altura da coluna de líquido no riser acima do
    ponto baixo (`level`, m, até à primeira célula com fração de gás >= `thr`) e a fração de líquido média do
    riser (`riser_holdup`). Devolve dict com `T`, `seal` (m), `level` (m), `riser_holdup` e `H` (altura do
    riser acima do ponto baixo, m)."""
    from .geometry import pipe_mesh

    cs = cells or cell_states(run, pipe, every)
    pipe_ = run.model.pipes[pipe].pipe
    pts = np.asarray(pipe_.points, float)
    mesh = pipe_mesh(pts[:, 0], pts[:, 1], pts[:, 2], np.asarray(pipe_.n_elements, int), grading=pipe_.grading)
    z_nodes = mesh["xyz"][:, 2]
    s_nodes = mesh["s"]
    s, aG = cs["s"], cs["Y"][:, :, 0]
    z = np.interp(s, s_nodes, z_nodes)
    k_low = int(np.argmin(z))
    th = np.rad2deg(cs["theta"])
    riser = np.flatnonzero(th > 45.0)
    z_top = float(z[riser].max()) if riser.size else float(z.max())
    H = z_top - z[k_low]
    seal, level, hold = [], [], []
    for a in aG:
        up = a[k_low::-1]  # do ponto baixo para montante
        d_up = s[k_low] - s[k_low::-1]
        seal.append(_crossing(d_up, up, thr) if up[0] < thr else 0.0)
        dn = a[k_low:]
        zz = z[k_low:] - z[k_low]
        level.append(min(_crossing(zz, dn, thr), H) if dn[0] < thr else 0.0)
        hold.append(float(np.mean(1.0 - a[riser])) if riser.size else np.nan)
    return dict(T=cs["T"], seal=np.array(seal), level=np.array(level), riser_holdup=np.array(hold), H=H)


def inventory_error(run, pipe=0):
    """Erro do balanço de massa de cada fase ao longo de uma corrida do esquema escalonado com o acumulador
    BDF-consistente (`run.acc`, issue 005): estoque(t) − estoque(0) − ∫(entrada − saída) dt, em kg, com o
    estoque somado sobre as células com a fração de gás sem corte (o modo conservativo admite um pequeno
    undershoot). Devolve dict com `T`, `L`, `G` (erros) e `mass_L`, `mass_G` (estoques) ou None sem `acc`."""
    if run.acc is None:
        return None
    model = run.model
    pb = model.pipes[pipe]
    sch = pb.scheme
    N = sch.N
    Y = run.pipe(pipe)["Y"][:, 1 : N + 1]
    A = np.pi * pb.pipe.D**2 / 4
    h = np.asarray(pb.mesh["h"])
    rhoL, rhoG, _, _ = model.fluid.props(Y[:, :, 1] * model.PR)
    mL = np.sum(rhoL * (1.0 - Y[:, :, 0]) * A * h, axis=1)
    mG = np.sum(rhoG * Y[:, :, 0] * A * h, axis=1)
    acc = np.asarray(run.acc, float)
    nt = min(acc.shape[1], mL.size)
    accL, accG = acc[2 * pipe, :nt] - acc[2 * pipe, 0], acc[2 * pipe + 1, :nt] - acc[2 * pipe + 1, 0]
    return dict(
        T=np.asarray(run.T)[:nt],
        L=(mL[:nt] - mL[0]) - accL,
        G=(mG[:nt] - mG[0]) - accG,
        mass_L=mL[:nt],
        mass_G=mG[:nt],
        fed_L=accL,
        fed_G=accG,
    )
