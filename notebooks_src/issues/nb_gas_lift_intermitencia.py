# %% [markdown]
# # Poço com gas lift: intermitência forte (complemento ao caso 10.3.2)
#
# Cenário além da tese: choke de gás a 12 % e a 10 % por 16 000 s, com ciclos de *heading*. Corridas `runs/issues/gas_lift_12`, `gas_lift_10` (BDF próprio, rtol 1e-3) do lote `batches/issues.json`.
#
# Issues ilustradas (índice em [docs/issues/INDEX.md](../../docs/issues/INDEX.md)):
# - [033](../../docs/issues/033-slow-exit-regressao.md)
# - [034](../../docs/issues/034-tolerancia-por-faixa.md)
# - [036](../../docs/issues/036-lado-direito-jax.md)
#
# *Nenhuma simulação corre aqui: as corridas vêm de `notebooks/data` (dados históricos, 2026-10)
# e dos lotes `batches/*.json` (`mfsim batch`); o notebook só lê e desenha.*
# %% [markdown]
# No cenário 2 da tese (choke de gás 75 % → 15 %) o poço passa a operar de forma
# intermitente. Aqui a restrição é maior (choke de gás a **12 %** e a **10 %**) e
# a simulação é longa (16 000 s) para capturar vários ciclos de *heading*:
#
# 1. com pouco gás injetado, a coluna no tubing fica pesada e a produção cai;
# 2. a pressão no anular (vértice 8) sobe lentamente até vencer a pressão do
#    tubing na válvula de retenção;
# 3. o gás acumulado é injetado de uma vez, aliviando a coluna: a produção de
#    líquido dispara (pico de ~23 kg/s contra 0.2–0.6 kg/s no mínimo);
# 4. o anular se esvazia, a injeção cessa e o ciclo recomeça.
#
# Mesma rede do caso 10.3.2 (diferenças finitas centrais, modelo a 3 equações
# no tubing, monofásicos nas linhas de gás, válvula de retenção regularizada).
# Tolerância relativa 1e-3 (BDF2); o integrador atravessa os transientes rápidos
# em cerca de um minuto de CPU por simulação.

# %%
import numpy as np, matplotlib.pyplot as plt
from mfsim.runs import load_run

runs = {}
for ab in (0.12, 0.10):
    r = load_run(f"issues/gas_lift_{int(ab * 100)}")
    print(
        f"abertura {ab:.0%}: {r.info['n_steps']} passos, {r.info['n_rejected']} rejeitados, {r.info['cpu']:.1f} s de CPU"
    )
    runs[ab] = r.res
# %% [markdown]
# ## Produção de líquido, injeção de gás e pressões

# %%
for ab, res in runs.items():
    T = res.T
    choke = res.elements["valves"][4]  # choke de produção (vértice 4)
    gl = res.elements["check_valves"][8]  # válvula de gas lift (vértice 8)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(T, choke["qL"])
    axes[0].set_ylabel("Produção de líquido (kg/s)")
    axes[1].plot(T, gl["qG"], color="C1")
    axes[1].set_ylabel("Gás injetado no tubing (kg/s)")
    axes[2].plot(T, gl["ZP"], color="C2", label="anular (vértice 8)")
    axes[2].plot(T, res.pipe(1)["out"]["P_bar"][0], color="C3", label="fundo do tubing (entrada da aresta 2)")
    axes[2].set_ylabel("Pressão (bar)")
    axes[2].legend()
    axes[3].plot(T, choke["ZP"], color="C4")
    axes[3].set_ylabel("Pressão a montante da choke (bar)")
    axes[3].set_xlabel("Tempo (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Gas lift com choke de gás a {ab:.0%}", y=1.01)
    fig.tight_layout()
    plt.show()
    sel = T > 2000
    q = choke["qL"][sel]
    print(
        f"abertura {ab:.0%}: produção média {q.mean():.2f} kg/s, mín {q.min():.2f}, máx {q.max():.2f}, desvio padrão {q.std():.2f}"
    )

# %% [markdown]
# ## Fração de gás e vazão de líquido ao longo do tubing (choke a 12 %)

# %%
from mfsim.plotting import pipe_surfaces

res = runs[0.12]
fig = pipe_surfaces(res, 1, keys=("alphaG", "P_bar", "qL", "qG"), title="Aresta 2: tubing")
plt.show()

# %% [markdown]
# ## Período dos ciclos
# Estimativa do período pela autocorrelação da produção de líquido (após 2000 s).

# %%
for ab, res in runs.items():
    T = res.T
    q = res.elements["valves"][4]["qL"]
    sel = T > 2000
    s = q[sel] - q[sel].mean()
    ac = np.correlate(s, s, mode="full")[s.size - 1 :]
    ac /= ac[0]
    dt = T[1] - T[0]
    # primeiro máximo local após o primeiro mínimo
    i_min = np.argmin(ac[: ac.size // 2])
    i_max = i_min + np.argmax(ac[i_min : ac.size // 2])
    print(f"abertura {ab:.0%}: período estimado ≈ {i_max * dt:.0f} s (autocorrelação {ac[i_max]:.2f})")
