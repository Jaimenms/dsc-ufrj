"""Estruturas comuns aos fechamentos."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

G = 9.80665
G_CLOSURE = 9.81  # valor usado nos termos de correção de pressão do código original


@dataclass
class ClosureInputs:
    """Variáveis locais (arrays 1-D de mesmo tamanho) usadas pelos fechamentos."""

    alphaG: np.ndarray
    alphaL: np.ndarray
    vL: np.ndarray      # velocidade média do líquido
    vLS: np.ndarray     # velocidade superficial do líquido
    vG: np.ndarray
    vGS: np.ndarray
    AreaL: np.ndarray
    AreaG: np.ndarray
    D: np.ndarray
    eps: np.ndarray     # rugosidade [m]
    theta: np.ndarray   # inclinação [rad]
    rhoL: np.ndarray
    rhoG: np.ndarray
    muL: np.ndarray
    muG: np.ndarray
    CL: np.ndarray
    CG: np.ndarray
    sigma: np.ndarray
    beta: np.ndarray    # ângulo molhado (estratificado)
    hD: np.ndarray      # h/D (estratificado)
    fi_mult: np.ndarray = None   # multiplicador do atrito interfacial do estratificado (None = 1)
    fi_vgt: np.ndarray = None    # v_GT de Andritsos & Hanratty (None = 5.5 m/s; inf = Taitel & Dukler)
    bubble_drag: np.ndarray = None   # 1 = arrasto físico de bolhas (PipeParams.bubble_drag = "fisico"); None/0 = fechamento da tese

    def subset(self, mask):
        return ClosureInputs(**{f.name: (getattr(self, f.name)[mask] if getattr(self, f.name) is not None else None) for f in fields(self)})

    def replace(self, **kw):
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(kw)
        return ClosureInputs(**d)

    @property
    def size(self):
        return self.alphaG.size


@dataclass
class ClosureTerms:
    """Termos de fechamento (por unidade de comprimento / Pa)."""

    GammaL: np.ndarray      # força dissipativa líquido [N/m] (parede + interface)
    GammaG: np.ndarray
    dPL: np.ndarray         # P_L - P  (correção de pressão da fase)
    dPG: np.ndarray
    ddPLdP: np.ndarray
    ddPGdP: np.ndarray
    ddPLdalphaG: np.ndarray
    ddPGdalphaG: np.ndarray
    dPI: np.ndarray         # P_interface - P
    hL: np.ndarray
    beta: np.ndarray
    tauG: np.ndarray        # tensão gás-parede
    tauL: np.ndarray        # tensão líquido-parede

    @classmethod
    def zeros(cls, n):
        return cls(**{f.name: np.zeros(n) for f in fields(cls)})

    def names(self):
        return [f.name for f in fields(self)]
