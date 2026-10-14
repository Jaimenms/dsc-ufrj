"""Gerador do experimento da coluna de líquido (issue 006, arrasto espúrio da fase virtual).

Coluna vertical de L m entre dois vértices: `B` = líquido monofásico; `Ap` = modelo a dois
fluidos cheia de líquido (αG = 0); `A` = tampão entre dois bolsões de gás (4 m com αG = 0.7 em
cada extremidade). Degrau de pressão inicial entre pressões fixas nas extremidades, ou
extremidades fechadas com o degrau no meio (modo acústico c/2L). Esquema escalonado
conservativo, `relax_target` "raw" (fração real) ou "noslip" (sem escorregamento).

    uv run python -m generators.column --out cases/studies     # escreve os seis casos
"""

from __future__ import annotations

import numpy as np

from mfsim.core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
from mfsim.fluid import Fluid
from mfsim.network.graph import BoundaryVertex, Network, Pipe, specs

FLUID = dict(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
CASES = (
    ("column_B_open", "B", False, "raw"),
    ("column_Ap_open_raw", "Ap", False, "raw"),
    ("column_Ap_open_noslip", "Ap", False, "noslip"),
    ("column_B_closed", "B", True, "raw"),
    ("column_Ap_closed_raw", "Ap", True, "raw"),
    ("column_Ap_closed_noslip", "Ap", True, "noslip"),
)


def column_case(which, closed=False, relax_target="raw", L=20.0, N=20, dP=0.2e5, tf=3.0, dt_out=1e-3, D=0.05):
    fluid = Fluid(**FLUID)
    P0 = 5e5
    xn = np.linspace(0, L, N + 1)
    ic = np.zeros((N + 1, 4))
    ic[:, 1] = (P0 + dP * (xn < L / 2)) / 1e6 if closed else (P0 + dP * (1 - xn / L)) / 1e6
    if closed:
        verts = [BoundaryVertex("1", spec=specs(qL=0.0, qG=0.0)), BoundaryVertex("2", spec=specs(qL=0.0, qG=0.0))]
    else:
        verts = [BoundaryVertex("1", spec=specs(PT=(P0 + dP) / 1e6)), BoundaryVertex("2", spec=specs(PT=P0 / 1e6))]
    if which == "B":
        model, extra = 3, {"q_switch": 1e-3}
        bci = (
            (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_ABSENT)
            if closed
            else (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
        )
        bco = bci
    else:
        model = 1
        if which == "A":
            ic[:, 0] = np.where((xn < 4.0) | (xn > L - 4.0), 0.7, 0.0)
        extra = {
            "regime_vertical": 0,
            "q_switch_L": 1e-3,
            "q_switch_G": 1e-5,
            "switch_frac": 0.5,
            "conservative": True,
            "relax_target": relax_target,
        }
        bci = (
            (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
            if closed
            else (BC_NEUMANN, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
        )
        bco = (
            (BC_ABSENT, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
            if closed
            else (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
        )
    pipe = Pipe.legacy(
        "1",
        "1",
        "2",
        [0.0, 0.0],
        [0.0, L],
        [0.0, 0.0],
        [N],
        D=D,
        roughness=1e-5,
        model=model,
        regime=4,
        method="fv_upwind",
        bc_in=bci,
        bc_out=bco,
        ic=ic,
        extra=extra,
    )
    net = Network.build(
        f"column_{which}_{'closed' if closed else 'open'}_{relax_target}",
        fluid,
        verts,
        [pipe],
        t0=0.0,
        tf=tf,
        dt_out=dt_out,
        PR=1e6,
        integrator="implicit",
        quadrature_order=2,
        max_step=np.inf,
        atol=1e-3,
        rtol=3e-3,
        backend="jax",
        first_step=1e-4,
        max_order=2,
        max_newton=20,
        newton_tol=1e-3,
        min_step=1e-9,
        slow_exit=False,
    )
    net.description = (
        f"coluna de {L:g} m: {which} ({'fechada' if closed else 'pressões fixas'}), relax_target={relax_target}"
    )
    return net


def main(argv=None):
    import argparse
    import os

    from mfsim.io.spec import case_to_json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cases/studies")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name, which, closed, rt in CASES:
        path = os.path.join(a.out, f"{name}.json")
        case_to_json(column_case(which, closed=closed, relax_target=rt), path)
        print("escrito", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
