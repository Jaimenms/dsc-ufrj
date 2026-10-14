# %% [markdown]
# # 10.1.1 Escoamento horizontal estratificado
#
# Duto horizontal de 1000 m e 0.1 m de diâmetro, 50 elementos finitos (Galerkin), modelo a
# dois fluidos (4 equações) com fechamento de padrão estratificado. $q_L = 0.5$ kg/s constante;
# $q_G$ sobe de 0.1 para 0.13 kg/s em rampa entre 150 s e 180 s; pressão de saída 100 bar;
# vazões impostas na entrada e pressão imposta na saída. Figuras 10.2 a 10.8 e Tabela 10.1.
#
# Os resultados originais do MATLAB (`CASO10-1-1.mf2`) são sobrepostos a tracejado.
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.1.1")
compare_states(res, raw)

# %% [markdown]
# ## Condições iniciais (Figura 10.2)
# O estado inicial é o gravado pelo MATLAB (`estados.X[:, 0]`, ver [issue 001](../../docs/issues/001-reproducao-capitulo-10.md)).

# %%
fig = pipe_profiles(res, keys=("P_bar", "alphaG"), it=0, legacy=legacy_pipe_dict(raw, 0))
fig.suptitle("Figura 10.2 - Condições iniciais (estado estacionário)", y=1.03)
plt.show()

# %% [markdown]
# ## Simulação dinâmica (Figura 10.3) e comparação com o MATLAB em três posições

# %%
show_pipe(res, raw, 0, title="Figura 10.3 - Escoamento horizontal estratificado (Galerkin, 50 elementos)")

# %% [markdown]
# ## Sensibilidade de malha (Figuras 10.4 e 10.5)
#
# Para cada malha o estado estacionário é recalculado e a simulação repetida, com o esquema
# de Galerkin e com diferenças finitas centrais (corridas `10_1_1_<esquema>_n<N>` do lote).

# %%
legacy = legacy_pipe_dict(raw, 0)
meshes = [5, 10, 25, 50, 100]
for scheme, fig_no in (("galerkin", "10.4"), ("fd_central", "10.5")):
    runs = {f"{n} elementos": load_run(f"chapter10/10_1_1_{scheme}_n{n}").res for n in meshes}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, k_ in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
        compare_runs(
            runs,
            key=k_,
            node=-1 if k_ != "P_bar" else 0,
            ax=ax,
            legacy=legacy if k_ != "P_bar" else None,
            title=f"{'saída' if k_ != 'P_bar' else 'entrada'} do duto",
        )
    fig.suptitle(f"Figura {fig_no} - Sensibilidade de malha ({scheme})", y=1.02)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Galerkin × diferenças finitas (Figura 10.6) e Tabela 10.1

# %%
sel = {
    "Galerkin 50": "galerkin_n50",
    "Dif. finitas 50": "fd_central_n50",
    "Galerkin 100": "galerkin_n100",
    "Dif. finitas 100": "fd_central_n100",
}
runs6 = {lab: load_run(f"chapter10/10_1_1_{tag}") for lab, tag in sel.items()}
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, k_ in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
    compare_runs(
        {lab: r.res for lab, r in runs6.items()},
        key=k_,
        node=-1 if k_ != "P_bar" else 0,
        ax=ax,
        legacy=legacy if k_ != "P_bar" else None,
    )
fig.suptitle("Figura 10.6 - Galerkin × diferenças finitas", y=1.02)
fig.tight_layout()
plt.show()

print("Tabela 10.1 - custo (estado estacionário + simulação, BDF do SciPy)")
print(f"{'Método':12s} {'N':>5s} {'CPU':>8s}")
for scheme in ("galerkin", "fd_central"):
    for n in meshes:
        r = load_run(f"chapter10/10_1_1_{scheme}_n{n}")
        print(f"{scheme:12s} {n:5d} {r.info.get('cpu_total', float('nan')):7.2f}s")

# %% [markdown]
# ## Autovalores da Jacobiana no estado final (Figuras 10.7 e 10.8)
# Formulação 1: condições de contorno do ficheiro (vazões e pressão de saída impostas, fração de gás
# livre na entrada; 201 estados). Formulação 2: fração de gás na entrada com gradiente nulo imposto
# (200 estados; corrida `10_1_1_neumann` do lote, com o estacionário recalculado). Com a fração de gás
# livre surge um autovalor nulo (modo neutro: a fração de gás de entrada não é determinada pelas
# equações); com gradiente nulo todos os autovalores ficam no semiplano esquerdo.

# %%
from mfsim.analysis import jacobian_eigenvalues
from mfsim.plotting import eigenvalue_plot

run2 = load_run("chapter10/10_1_1_neumann")
ev1 = jacobian_eigenvalues(res.model, res.X[:, -1], t=run.T[-1])
ev2 = jacobian_eigenvalues(run2.res.model, run2.X[:, -1], t=run2.T[-1])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
eigenvalue_plot(ev1, axes[0], "Formulação 1 (Fig. 10.7)")
eigenvalue_plot(ev2, axes[1], "Formulação 2 (Fig. 10.8)")
for ax in axes:
    ax.set_xscale("symlog", linthresh=1e-3)
plt.show()
print(f"máx parte real: formulação 1 = {ev1.real.max():.2e} | formulação 2 = {ev2.real.max():.2e}")

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
