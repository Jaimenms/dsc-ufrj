# %% [markdown]
# # 10.1.1 Escoamento horizontal estratificado
#
# Reprodução, com o **mfsim** (Python), do primeiro caso do capítulo 10 da tese:
# duto horizontal de 1000 m e 0.1 m de diâmetro, 50 elementos finitos (Galerkin),
# modelo a dois fluidos (4 equações) com fechamento de padrão estratificado.
#
# Cenário: $q_L = 0.5$ kg/s constante; $q_G$ sobe de 0.1 para 0.13 kg/s em rampa
# entre 150 s e 180 s; pressão de saída 100 bar. Condições de contorno Dirichlet
# para as vazões na entrada e a pressão na saída.
#
# Os resultados originais do MATLAB (arquivo `CASO10-1-1.mf2`) são sobrepostos
# para verificação. Figuras reproduzidas: 10.2 a 10.8 e Tabela 10.1.

# %%
import copy, time
import numpy as np
import matplotlib.pyplot as plt
from mfsim.io_legacy import chapter10_case, chapter10_info, legacy_results
from mfsim.network.graph import ports
from mfsim.simulation import simulate, steady_state, build_model
from mfsim.plotting import (
    pipe_surfaces,
    pipe_profiles,
    pipe_timeseries,
    compare_runs,
    legacy_pipe_dict,
    eigenvalue_plot,
)
from mfsim.io_legacy import legacy_states
from mfsim.integrators.jacobian import colored_jacobian

info = chapter10_info("10.1.1")
case, raw = chapter10_case("10.1.1"), legacy_results("10.1.1")
legacy = legacy_pipe_dict(raw, 0)
print(info.title)
print(info.description)
print("Figuras:", info.figures)
print(case.time, case.reference, case.integration)

# %% [markdown]
# ## Estado estacionário (Figura 10.2)
#
# A condição inicial é o estado estacionário obtido anulando as derivadas temporais
# do modelo dinâmico. Partimos de uma estimativa baseada no nível estratificado de
# equilíbrio (modelo de Taitel & Dukler) e resolvemos o sistema algébrico por Newton
# com Jacobiana esparsa. O resultado é comparado com o estado inicial gravado pelo MATLAB.
#
# **Observação importante.** Com a fração de gás *livre* na entrada (condição de
# contorno "ausente", como no arquivo original) a Jacobiana do sistema discretizado
# tem um autovalor nulo (ver Figura 10.7 abaixo): o estado estacionário discreto não
# é único e o Newton pode convergir para perfis de fração de gás não uniformes que
# dependem da malha. Com gradiente nulo (von Neumann) na entrada o estado é único
# e coincide com o da tese para todas as malhas. Por isso, no estudo de malha
# usa-se von Neumann; na simulação principal parte-se do estado gravado pelo MATLAB.

# %%
case_ss = copy.deepcopy(case)
case_ss.pipes[0].ports_in = ports((3, 1, 2, 2))  # von Neumann para alphaG na entrada
case_ss.pipes[0].ic = None
t0 = time.time()
model_ss, X_ss, info_ss = steady_state(case_ss, X0="auto", verbose=True)
print(f"convergiu={info_ss['converged']} resíduo={info_ss['residual']:.2e} em {time.time()-t0:.2f} s")
Tl, Xl = legacy_states(raw)
E, Ys, _, _ = model_ss.assemble(0.0, X_ss)
print("fração de gás entrada/saída (Python):", Ys[0][0, 0], Ys[0][-1, 0], " | MATLAB:", Xl[0, 0], Xl[-3, 0])
print("pressão de entrada (bar): Python", Ys[0][0, 1] * 100, " | MATLAB", Xl[1, 0] * 100)

# %%
from mfsim.results import Results

res0 = Results(model_ss, [0.0], X_ss[:, None])
fig = pipe_profiles(res0, keys=("P_bar", "alphaG"), it=0, legacy=legacy)
fig.suptitle("Figura 10.2 - Condições iniciais (estado estacionário)", y=1.03)
plt.show()

# %% [markdown]
# ## Simulação dinâmica (Figura 10.3)
#
# Integração a partir do estado inicial gravado pelo MATLAB, com o BDF implícito
# próprio (ordem variável 1-2, passo adaptativo, Newton modificado com Jacobiana
# esparsa colorida). As condições de contorno são as do arquivo original.

# %%
t0 = time.time()
res = simulate(case, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf)
print(f"tempo de CPU: {time.time()-t0:.2f} s | {res.info}")
fig = pipe_surfaces(res, 0)
fig.suptitle("Figura 10.3 - Escoamento horizontal estratificado (Galerkin, 50 elementos)", y=1.02)
plt.show()

# %% [markdown]
# ### Comparação Python × MATLAB em três posições do duto
# (linhas cheias: mfsim; tracejadas: resultados originais gravados no `.mf2`)

# %%
fig = pipe_timeseries(res, 0, legacy=legacy)
plt.show()
slots = res.model.state_slots()
for s, name in enumerate(["alphaG", "PT", "qL", "qG"]):
    m = slots == s
    print(
        f"{name:7s}: máx |Python - MATLAB| = {np.max(np.abs(res.X[m] - Xl[m])):.2e}  (faixa MATLAB {np.ptp(Xl[m]):.2e})"
    )

# %% [markdown]
# ## Sensibilidade de malha - Galerkin (Figura 10.4) e diferenças finitas (Figura 10.5)
#
# Para cada malha o estado estacionário é recalculado e a simulação repetida.
# Os tempos de CPU reproduzem a Tabela 10.1 (na tese: 17 a 132 s em 2010; aqui
# com o integrador implícito próprio).


# %%
def run_mesh(nel, method):
    c = copy.deepcopy(case_ss)
    c.pipes[0].n_elements = np.array([nel])
    c.pipes[0].ic = None
    c.set_method(method)
    t0 = time.time()
    steady_state(c, X0="auto", verbose=False)
    r = simulate(c, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
    return r, time.time() - t0


meshes = [5, 10, 25, 50, 100]
runs = {}
table = []
for method in ["galerkin", "fd_central"]:
    for nel in meshes:
        r, cpu = run_mesh(nel, method)
        runs[(method, nel)] = r
        table.append((method, nel, cpu, r.info["n_steps"]))
        print(f"{method:11s} N={nel:4d}: {cpu:6.2f} s, {r.info['n_steps']} passos")

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, key in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
    compare_runs(
        {f"{n} elementos": runs[("galerkin", n)] for n in meshes},
        key=key,
        node=-1 if key != "P_bar" else 0,
        ax=ax,
        title=f"Galerkin - {'saída' if key != 'P_bar' else 'entrada'} do duto",
    )
fig.suptitle("Figura 10.4 - Sensibilidade de malha (Galerkin)", y=1.02)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, key in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
    compare_runs(
        {f"{n} nós": runs[("fd_central", n)] for n in meshes},
        key=key,
        node=-1 if key != "P_bar" else 0,
        ax=ax,
        title=f"Dif. finitas - {'saída' if key != 'P_bar' else 'entrada'} do duto",
    )
fig.suptitle("Figura 10.5 - Sensibilidade de malha (diferenças finitas centrais)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Comparação entre Galerkin e diferenças finitas (Figura 10.6) e Tabela 10.1

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, key in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
    compare_runs(
        {
            "Galerkin 50": runs[("galerkin", 50)],
            "Dif. finitas 50": runs[("fd_central", 50)],
            "Galerkin 100": runs[("galerkin", 100)],
            "Dif. finitas 100": runs[("fd_central", 100)],
        },
        key=key,
        node=-1 if key != "P_bar" else 0,
        ax=ax,
        legacy=legacy if key != "P_bar" else None,
    )
fig.suptitle("Figura 10.6 - Galerkin × diferenças finitas", y=1.02)
fig.tight_layout()
plt.show()

print("Tabela 10.1 - tempos computacionais (estado estacionário + simulação, Python)")
print(f"{'Método':12s} {'N':>5s} {'CPU (s)':>9s} {'passos':>7s}")
for method, nel, cpu, steps in table:
    print(f"{method:12s} {nel:5d} {cpu:8.2f}s {steps:7d}")

# %% [markdown]
# ## Autovalores da Jacobiana no estado final (Figuras 10.7 e 10.8)
#
# Formulação 1: condições de contorno do arquivo (vazões e pressão de saída Dirichlet,
# fração de gás livre na entrada: 201 estados). Formulação 2: fração de gás na
# entrada com gradiente nulo imposto (200 estados). Com a fração de gás livre na
# entrada surge um autovalor nulo (modo neutro: a fração de gás de entrada não é
# determinada pelas equações), que explica a não-unicidade do estado estacionário
# discutida acima; com gradiente nulo todos os autovalores ficam no semiplano esquerdo.


# %%
def jacobian_eigs(c, X):
    m = build_model(c)
    f0 = m.rhs(c.time.tf, X)
    n = m.n_states
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-7 * abs(X[j]) + 1e-9
        xp = X.copy()
        xp[j] += h
        J[:, j] = (m.rhs(c.time.tf, xp) - f0) / h
    return np.linalg.eigvals(J)


ev1 = jacobian_eigs(case, res.X[:, -1])
c2 = copy.deepcopy(case_ss)
res2 = simulate(c2, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
ev2 = jacobian_eigs(c2, res2.X[:, -1])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
eigenvalue_plot(ev1, axes[0], "Formulação 1 (Fig. 10.7)")
eigenvalue_plot(ev2, axes[1], "Formulação 2 (Fig. 10.8)")
for ax in axes:
    ax.set_xscale("symlog", linthresh=1e-3)
plt.show()
print("máx parte real: formulação 1 =", ev1.real.max(), "| formulação 2 =", ev2.real.max())


# %% [markdown]
# ## Lado direito compilado (JAX)
# A mesma simulação com `simulate(..., jax=True)`: lado direito compilado por
# `jax.jit` e Jacobiana exata por diferenciação automática (`mfsim/compiled.py`, o mesmo código numpy com o backend JAX),
# no lugar das diferenças finitas coloridas. O resultado deve coincidir com a versão
# numpy dentro da tolerância do integrador (as sequências de passos podem diferir
# porque a Jacobiana muda o Newton); o ganho de tempo cresce com o número de passos
# e de Jacobianas.

# %%
from mfsim.notebook_utils import jax_comparison

res_jax = jax_comparison(case, res, rtol=1e-4, atol=1e-6)
