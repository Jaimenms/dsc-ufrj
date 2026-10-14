# Arquivo da pasta scratch (2026-10)

Conteúdo da antiga `scratch/` movido na fase 3 da reorganização (nada apagado; os ficheiros
versionados foram movidos com `git mv`, os dados com `mv`). Os scripts úteis passaram a `tools/`
(caminhos atualizados); as experiências ficam em `py/` (referidas no README/notebooks ou com lições
registadas) e `obsolete/` (versões iniciais superadas). Os dados (`npz/`, `logs/`, `nblogs/`, `cmp*/`)
estão fora do git e fixados pelo `MANIFEST.sha256`; `runs_index.json` lista as opções de cada npz.

Equivalentes atuais: `scratch/severe_D.py '{opts}' tag` -> `tools/severe_run.py` (mesmas opções) ou
`mfsim run caso.json` (casos em `cases/severe_slug/*.json`); `scratch/cycle_stats.py` -> `mfsim stats`;
`val_launch*.sh` -> `mfsim run cases/severe_slug/lab_*.json --backend jax`.

| origem | destino | propósito |
|---|---|---|
| `scratch/severe_D.py` | `tools/severe_run.py` | (sem docstring) |
| `scratch/save_run_data.py` | `tools/save_run_data.py` | Converte scratch/severe_D_<tag>.npz (+ log) em notebooks/data/<nome>.npz com meta. |
| `scratch/cycle_stats.py` | `tools/cycle_stats.py` | Estatísticas de ciclo de trajetórias gravadas do severe slug (partida bloqueada, duto único). |
| `scratch/scan_summary.py` | `tools/scan_summary.py` | Resumo da varredura da fronteira de intermitência severa: por trajetória gravada, vazões, |
| `scratch/val_summary.py` | `tools/val_summary.py` | Resumo das corridas de validação com dados de laboratório (item 8): período, pressões e picos |
| `scratch/val_plot.py` | `tools/val_plot.py` | Figura rápida de uma corrida de validação: P na entrada (kPa), vazão de líquido na saída, fração de gás na base e no topo do riser. |
| `scratch/riser_cc.py` | `tools/riser_cc.py` | Diagnóstico do filme no riser em trajetórias gravadas (partida bloqueada, duto único): |
| `scratch/analyze_run.py` | `tools/analyze_run.py` | (sem docstring) |
| `scratch/compare_network.py` | `tools/compare_network.py` | Compara a rede de 4 dutos (vértices dinâmicos) com o duto único curvado no severe slug. |
| `scratch/run_all_cases.py` | `tools/run_all_cases.py` | (sem docstring) |
| `scratch/verify_jax.py` | `tools/verify_jax.py` | Verifica rhs JAX vs numpy e a Jacobiana em todos os casos dos notebooks. |
| `scratch/compare_jax.py` | `tools/compare_jax.py` | Compara numpy e JAX num caso dos notebooks: estado estacionário (opcional) e simulação. |
| `scratch/diag_lab.py` | `tools/diag_lab.py` | Saída densa do estado de um caso de laboratório (opções JSON como severe_D.py) entre t0 e t0+dur: variação máxima |
| `scratch/diag_state.py` | `tools/diag_state.py` | Saída densa do estado do severe slug (opções JSON como severe_D.py) entre t0 e t0+dur: |
| `scratch/diag_ipe.py` | `tools/diag_ipe.py` | Diagnóstico do IPE em dinâmica: integra o severe slug (opções JSON como severe_D.py) até t0, |
| `scratch/restart_ring.py` | `archive/scratch_2026-10/py/restart_ring.py` | Reinicia a trajetória M1 num instante com tampão de líquido (toque acústico) e integra |
| `scratch/restart_probe.py` | `archive/scratch_2026-10/py/restart_probe.py` | Reinício em t0 com estatísticas do Newton/controlador (fase cara). |
| `scratch/reversal_demo.py` | `archive/scratch_2026-10/py/reversal_demo.py` | Inversão do sentido do escoamento num duto entre dois vértices de pressão (esquema |
| `scratch/plug_ring.py` | `archive/scratch_2026-10/py/plug_ring.py` | Excitação do modo acústico do tampão: três colunas de 20 m (D = 0.05 m, 20 células de 1 m) |
| `scratch/well_fv.py` | `archive/scratch_2026-10/py/well_fv.py` | Poço 10.3.1 (B&B): transiente de fechamento/abertura do choke com o esquema nodal da tese |
| `scratch/lessons_gaslift.py` | `archive/scratch_2026-10/py/lessons_gaslift.py` | Configuração rápida do integrador (atol nas vazões, Newton 20) no gas lift intermitente (JAX). |
| `scratch/lessons_massbal.py` | `archive/scratch_2026-10/py/lessons_massbal.py` | Balanço de massa (acumulador BDF-consistente) nos casos da tese com os esquemas legados e com fv_upwind. |
| `scratch/probe_ctrl.py` | `archive/scratch_2026-10/py/probe_ctrl.py` | (sem docstring) |
| `scratch/probe_gaslift_jax.py` | `archive/scratch_2026-10/py/probe_gaslift_jax.py` | (sem docstring) |
| `scratch/probe_jump.py` | `archive/scratch_2026-10/py/probe_jump.py` | Procura descontinuidades de f ao longo da direção de Newton num passo que estagna. |
| `scratch/probe_newton.py` | `archive/scratch_2026-10/py/probe_newton.py` | Traça a convergência do Newton (modo conservativo vs primitivo) até t=tend. |
| `scratch/newton_probe.py` | `archive/scratch_2026-10/py/newton_probe.py` | (sem docstring) |
| `scratch/newton_stats.py` | `archive/scratch_2026-10/py/newton_stats.py` | (sem docstring) |
| `scratch/gl_strong.py` | `archive/scratch_2026-10/py/gl_strong.py` | Gas lift com intermitência mais forte: choke de gás mais fechada, simulação longa. |
| `scratch/bb_exact_effect.py` | `archive/scratch_2026-10/py/bb_exact_effect.py` | (sem docstring) |
| `scratch/diag_K7.py` | `archive/scratch_2026-10/py/diag_K7.py` | Diagnóstico da malha fina graduada (K7): o que oscila em 60-80 s? |
| `scratch/diagStall.py` | `archive/scratch_2026-10/py/diagStall.py` | (sem docstring) |
| `scratch/diagM.py` | `archive/scratch_2026-10/py/diagM.py` | (sem docstring) |
| `scratch/diag2600.py` | `archive/scratch_2026-10/py/diag2600.py` | (sem docstring) |
| `scratch/cont2600.py` | `archive/scratch_2026-10/py/cont2600.py` | (sem docstring) |
| `scratch/severe_C.py` | `archive/scratch_2026-10/py/severe_C.py` | Caso 'de laboratório': ar-água, linha descendente curta, riser 15 m, separador 1.5 bar. |
| `scratch/severe_B.py` | `archive/scratch_2026-10/py/severe_B.py` | (sem docstring) |
| `scratch/severe_B_diag.py` | `archive/scratch_2026-10/py/severe_B_diag.py` | (sem docstring) |
| `scratch/severe_B_diag2.py` | `archive/scratch_2026-10/py/severe_B_diag2.py` | (sem docstring) |
| `scratch/severe_diag.py` | `archive/scratch_2026-10/py/severe_diag.py` | (sem docstring) |
| `scratch/severe_diag2.py` | `archive/scratch_2026-10/py/severe_diag2.py` | (sem docstring) |
| `scratch/severe1.py` | `archive/scratch_2026-10/obsolete/severe1.py` | (sem docstring) |
| `scratch/severe2.py` | `archive/scratch_2026-10/obsolete/severe2.py` | (sem docstring) |
| `scratch/severe3.py` | `archive/scratch_2026-10/obsolete/severe3.py` | (sem docstring) |
| `scratch/severe_single.py` | `archive/scratch_2026-10/obsolete/severe_single.py` | (sem docstring) |
| `scratch/gl_diag.py` | `archive/scratch_2026-10/obsolete/gl_diag.py` | (sem docstring) |
| `scratch/gl_run.py` | `archive/scratch_2026-10/obsolete/gl_run.py` | (sem docstring) |
| `scratch/gl_verbose.py` | `archive/scratch_2026-10/obsolete/gl_verbose.py` | (sem docstring) |
| `scratch/cmpA.py` | `archive/scratch_2026-10/obsolete/cmpA.py` | (sem docstring) |
| `scratch/cmp_10_1_1.py` | `archive/scratch_2026-10/obsolete/cmp_10_1_1.py` | (sem docstring) |
| `scratch/cmp_scipy.py` | `archive/scratch_2026-10/obsolete/cmp_scipy.py` | (sem docstring) |
| `scratch/run_10_1_1.py` | `archive/scratch_2026-10/obsolete/run_10_1_1.py` | (sem docstring) |

| dados | ficheiros |
|---|---|
| `cmp/` | 12 |
| `cmp2/` | 2 |
| `logs/` | 394 |
| `nblogs/` | 8 |
| `npz/` | 182 |
| `sh/` | 3 |
