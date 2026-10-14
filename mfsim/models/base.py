"""Interface dos modelos de duto.

Todos os modelos trabalham sobre o vetor nodal de 4 variáveis por nó
(layout do MFSIM original):

    y[node, 0] = alphaG   (fração de área de gás)
    y[node, 1] = PT       (pressão adimensional P/PR)
    y[node, 2] = qL       (vazão mássica de líquido, kg/s)
    y[node, 3] = qG       (vazão mássica de gás, kg/s)

Cada modelo declara quais slots são variáveis de estado ("slots") e o
sistema quase-linear nesses slots:

    dy/dt + A(y) dy/dx + B(y) = 0

A(y) e B(y) são montados a partir de um vetor de coeficientes F (nF x n)
através dos padrões `A_pattern` (nv x nv, índice 1-based em F, 0 = nulo)
e `B_pattern` (nv). Esse formato é o usado pelas discretizações de Galerkin,
Petrov-Galerkin e diferenças finitas da tese.

Além disso, os modelos expõem `fluxes()` com os termos em forma
conservativa (usados pelo esquema de volumes finitos escalonados).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SLOT_ALPHA, SLOT_P, SLOT_QL, SLOT_QG = 0, 1, 2, 3
G = 9.80665


@dataclass
class PipeParams:
    """Parâmetros geométricos de um duto para avaliação do modelo."""

    D: float
    eps: float              # rugosidade [m]
    PR: float               # pressão de referência para PT
    regime: int = 4         # 1 anular, 2 bolhas, 3 intermitente, 4/5 estratificado, 6 automático
    k_switch: float = 1e6   # suavidade do chaveamento de regime
    ipe_smooth: bool = False   # identificação de padrão suave (sigmoides relativas, k ~ 20; equilíbrios com |v| e hD interpolado; sentido com tanh)
    ipe_local: bool = False    # critérios de padrão no estado local (nível/fração reais; fração de bolhas max(aG_bol, alphaG)) em vez dos equilíbrios
    alpha_min: float = 1e-3
    alpha_max: float = 1.0 - 1e-3
    legacy_convective: bool = True   # usa v|v| e 2|v| como no MFSIM original
    regime_vertical: int = 0         # se > 0, padrão usado onde |theta| > vertical_angle (ex.: riser)
    vertical_angle: float = 45.0     # graus
    vertical_blend: float = 0.0      # graus: meia-largura da transição suave entre o padrão da linha e o vertical (0 = chave dura)
    interfacial_pressure: float = 0.0   # delta: termo de pressão interfacial regularizante (CATHARE/Bestion), 0 = desligado
    low_void_regime: int = 0            # se > 0, padrão usado quando alphaG < low_void_alpha (ex.: 2 = bolhas; o estratificado é mal posto com camada de gás muito fina)
    low_void_alpha: float = 0.01        # fração de gás abaixo da qual o padrão de baixa fração entra (rampa C1 até 2x esse valor)
    high_void_regime: int = 0           # se > 0, padrão usado quando alphaG > high_void_alpha (ex.: 1 = anular: filme de parede num duto quase cheio de gás)
    high_void_alpha: float = 0.9        # fração de gás acima da qual o padrão de alta fração entra (rampa C1 de high_void_alpha - 0.05 a + 0.05)
    ip_slip_max: float = 10.0           # m/s: limite do escorregamento usado no termo de pressão interfacial
    bb_derivatives: str = "exact"       # Beggs & Brill: derivadas internas (momento, hold-up) "exact" (diferenciação
                                        # automática, só no caminho JAX) ou "fd" (diferenças finitas de 1e-3, como na tese;
                                        # o caminho numpy usa sempre "fd")
    fi_mult: float = 1.0     # multiplicador do fator de atrito interfacial do estratificado (sensibilidade)
    fi_vgt: float = 5.5      # v_GT (m/s) de Andritsos & Hanratty no atrito interfacial; inf -> Taitel & Dukler (f_I = f_G)
    virtual_mass: float = 0.0   # coeficiente de massa virtual C_vm (Drew & Lahey; 0.5 para bolhas esféricas), pesado pelo indicador de bolhas
    bubble_drag: str = "tese"   # arrasto interfacial do padrão bolhas: "tese" (modelo_bolhas.m: vb = sqrt(0.51 db + 2.14/db), sem g e
                                # sigma/rhoL; tensão interfacial f_I rhoG dv|dv|/2 com f_I = 1.5 alphaG rhoL CD/db, isto é, força ~ rhoG alphaG^2)
                                # ou "fisico" (vb de Mendelson/Fan-Tsuchiya sqrt(2.14 sigma/(rhoL db) + 0.505 g db), CD = 4/3 g db Δrho/(rhoL vb^2),
                                # força por volume 3/4 CD alphaG rhoL |vr| vr / db: na velocidade terminal o arrasto iguala o empuxo)
    vm_lambda: float = 1.0      # forma objetiva de Drew & Lahey: termos (lambda-2)(vG-vL) dvG/dx + (1-lambda)(vG-vL) dvL/dx (lambda = 2: só acelerações)
    extra: dict = field(default_factory=dict)


class PipeModel:
    """Classe base. Subclasses definem slots, padrões e `coefficients`.

    `bk` é o backend numérico (numpy por omissão, ou JAX): os métodos usam `self.xp`
    para a aritmética e as primitivas do backend onde os dois diferem."""

    name = "base"
    slots: tuple = (0, 1, 2, 3)
    A_pattern: np.ndarray = None
    B_pattern: np.ndarray = None
    nF: int = 12

    def __init__(self, fluid, params: PipeParams, bk=None):
        from ..backend import NUMPY
        self.fluid = fluid
        self.params = params
        self.bk = bk or NUMPY
        self.xp = self.bk.xp

    @property
    def nv(self):
        return len(self.slots)

    def coefficients(self, y, theta, outputs=False):
        """Coeficientes F (nF x n) em pontos com estados y (n x 4) e ângulos theta (n,).

        Se outputs=True retorna também um dict de saídas secundárias.
        """
        raise NotImplementedError

    def quasi_linear(self, y, theta):
        """Retorna (A, B): A (n, nv, nv), B (n, nv) a partir dos padrões."""
        xp = self.xp
        F = self.coefficients(y, theta)
        n = F.shape[1]
        z = xp.zeros(n)
        A = xp.stack([xp.stack([F[k - 1] if k > 0 else z for k in row], axis=1) for row in self.A_pattern], axis=1)
        B = xp.stack([F[k - 1] if k > 0 else z for k in self.B_pattern], axis=1)
        return A, B

    # --- utilidades comuns ---------------------------------------------------
    def unpack(self, y):
        y = self.xp.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 4)
        return y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    def default_state(self, alphaG, PT, qL, qG):
        """Estado nodal "cheio" (4 slots) a partir das variáveis físicas."""
        return np.stack([alphaG, PT, qL, qG], axis=1)
