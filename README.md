# mfsim — simulador dinâmico de escoamento bifásico em redes de dutos

Reimplementação em Python do MFSIM/SDMPFLOW (MATLAB, 2010) desenvolvido na tese
*Modelagem Dinâmica de Redes de Escoamento Bifásico* (Jaime Souza), com foco no
capítulo 10 (simulação dinâmica) e na exploração de intermitência severa
(*severe slugging*) em geometrias linha–riser.

## Instalação e uso

```bash
uv sync                      # cria o ambiente e instala o pacote (editável)
uv run pytest                # testes (fechamentos x MATLAB, regressão dos casos)
uv run jupyter lab notebooks # notebooks de reprodução das figuras
```

Os notebooks são gerados a partir de `notebooks_src/*.py` (marcadores `# %%`):

```bash
uv run python tools/make_nb.py notebooks_src/nb_10_1_1_estratificado.py --execute   # ou: uv run mfsim nb ...
```

Linha de comando (`mfsim/cli.py`; os casos são ficheiros JSON `mfsim-case/1`, ver
[docs/data_formats.md](docs/data_formats.md)):

```bash
uv run mfsim run cases/severe_slug/reference.json --backend jax --tf 150 --out runs/ref150.npz
uv run mfsim run cases/severe_slug/lab_jansen.json --backend jax        # validação de laboratório (shut_in, slow)
uv run mfsim steady cases/chapter10/10_1_1.json --out /tmp/10_1_1_ss.json
uv run mfsim stats severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys   # ciclos de uma corrida gravada
uv run mfsim export-case --chapter10            # .mf2 -> cases/chapter10/*.json
uv run mfsim export-case --severe --out novo.json --param D=0.05 --param start=blocked
```

`mfsim run` grava a corrida em npz v2 (`T`, `X`, `acc`, `meta`, `info_json`,
`provenance_json` e o caso completo em `case_json`; `mfsim.runs.load_run` lê qualquer
corrida — `runs/`, `notebooks/data/`, arquivo — e reconstrói caso, modelo e resultados);
uma falha do integrador grava a trajetória parcial. As ferramentas de análise e
diagnóstico ficam em `tools/` (`severe_run.py` é o antigo driver `scratch/severe_D.py`
com as mesmas opções; `cycle_stats`, `scan_summary`, `val_summary`, `riser_cc`,
`verify_jax`, `diag_*`); a antiga `scratch/` está arquivada em
`archive/scratch_2026-10/` (índice em `INDEX.md`, nada apagado; os dados fora do git
são fixados por `MANIFEST.sha256`, `uv run python tools/manifest.py --check`).

## Estrutura

| módulo | conteúdo |
|---|---|
| `mfsim/fluid.py` | equação de estado politrópica (ρ, C = dρ/dP) |
| `mfsim/geometry.py` | nível estratificado em seção circular, malha do duto |
| `mfsim/friction.py` | fator de atrito de Fanning (Churchill) |
| `mfsim/closures/` | fechamentos por padrão (estratificado, anular, bolhas, intermitente) e identificação do padrão (IPE: Taitel, Pots, critérios de estabilidade) |
| `mfsim/models/` | modelos de duto: dois fluidos (4 eq.), mistura Beggs & Brill (3 eq.), monofásicos |
| `mfsim/discretization/` | Galerkin, Petrov-Galerkin, diferenças centrais, diferenças regressivas (métodos da tese) e **volumes finitos escalonados com upwind pela velocidade de fase** (`fv_upwind`, novo; classes concretas por modelo: dois fluidos, deriva, monofásico, e `fv_upwind_cons` com massas por volume) |
| `mfsim/network/` | dados do caso, especificações temporais, relações das arestas de comprimento zero (válvula/reservatório/válvula de retenção/conexão), grafo `Vertex`/`Edge` (`graph.py`, `schedules.py`) e `NetworkModel` (layout de estados, `time_inputs(t)`, núcleo puro `rhs_core(x, E_t, opt_t)`, mapa conservativo, escalas, esparsidade) |
| `mfsim/backend/` | backend numérico injetado (`NUMPY`, `JAX`): as primitivas (`set/add` funcionais, `while_loop`, `sigmoid`, `masked_apply`, `robust_inv`, `jvp`) sobre as quais fluido, fechamentos, modelos, esquemas e rede são escritos **uma só vez** |
| `mfsim/compiled.py` | `CompiledNetwork(model)`: o mesmo `NetworkModel` reconstruído com o backend JAX, `jit` do `rhs_core` e Jacobiana exata por `jacfwd` (`build_jax_rhs`/`get_jax`) |
| `mfsim/core/` | slots do estado, `PortKind` (livre/Dirichlet/Neumann), dataclasses de opções (modelo, esquema, tempo, referência, integração) |
| `mfsim/io/`, `mfsim/initial/` | casos em JSON (`mfsim-case/1`, `$ref`/`$include`), conversão dos `.mf2`; receitas de condição inicial (bloqueada, linha fechada) |
| `mfsim/runs.py`, `mfsim/analysis.py` | carregador único das corridas gravadas (`load_run`, `case_from_meta`, `integrate_run`), assinaturas de ciclo |
| `mfsim/cli.py` | `mfsim run\|steady\|stats\|export-case\|nb` |
| `cases/`, `tools/`, `tests/golden/`, `archive/` | casos JSON (capítulo 10, severe slug), ferramentas (`make_nb`, `golden_generate`, `manifest`, `severe_run`, análises), referências de regressão (nível A = bit a bit), arquivo da antiga `scratch/` |
| `mfsim/integrators/` | BDF implícito próprio (ordem 1–2, passo adaptativo, Newton modificado, Jacobiana colorida) e invólucro do `solve_ivp` |
| `mfsim/steady.py`, `mfsim/initial_guess.py` | estado estacionário (Newton + pseudo-transiente) e estimativa inicial |
| `mfsim/io_legacy.py` | leitura dos `.mf2` originais (entrada e resultados do MATLAB) |
| `mfsim/cases/` | casos do capítulo 10 e gerador do caso de severe slug |
| `mfsim/plotting.py`, `mfsim/notebook_utils.py` | figuras no estilo da tese |

## Documentação

- [docs/thesis_reproduction.md](docs/thesis_reproduction.md) — modelo e layout de estados, resultados da reprodução do capítulo 10, gas lift com intermitência forte, observações sobre os ficheiros originais.
- [docs/data_formats.md](docs/data_formats.md) — casos em JSON (`mfsim-case/1`) e grafo da rede, corridas gravadas (npz v1/v2), referências de regressão e manifesto.
- [docs/jax_backend.md](docs/jax_backend.md) — backend único numpy/JAX, `CompiledNetwork`, paridade e desempenho.
- [docs/severe_slug.md](docs/severe_slug.md) — intermitência severa: esquema escalonado conservativo, rede com vértices dinâmicos, identificação automática de padrão, itens 1–7 do modelo físico, fronteira de Bøe/Taitel.
- [docs/lab_validation.md](docs/lab_validation.md) — validação com dados de laboratório (Jansen, Fabre, Tengesdal; item 8).
- [docs/findings_for_paper.md](docs/findings_for_paper.md) — achados para o artigo (resumo).
- `archive/scratch_2026-10/INDEX.md` — índice do arquivo da antiga `scratch/`; `tests/golden/CHANGELOG.md` — registo das referências de regressão.

## Reorganização do código (2026-10, status)

Plano aprovado (OO + injeção de dependências, grafo `Vertex`/`Edge`, casos em JSON, backend
único numpy/JAX, `scratch/` arquivada, camada de compatibilidade temporária) e executado
por fases, cada uma fechada com a suíte de regressão bit a bit verde:

| fase | conteúdo | estado |
|---|---|---|
| 0 | git, manifesto dos dados, referências de regressão (níveis A/B/C), `mfsim.runs` | feito (tags `v0-baseline`, `v0-results-snapshot`) |
| 1 | `core/options`, grafo `network/graph` + `schedules`, `io/spec` (JSON `mfsim-case/1`) + `legacy_mf2`, receitas de condição inicial, casos JSON do capítulo 10 e do severe slug; round-trip bit-idêntico | feito |
| 2 | backend único (`mfsim/backend`): fluido, fechamentos, identificação de padrão, modelos, discretizações, rede numa só implementação; `mfsim/compiled.py`; `jax_physics`/`jax_schemes`/`jaxrhs` apagados; numpy bit-idêntico, JAX paridade 1e−14, desempenho ≤ 1.1× | feito (tag `v2-backend`) |
| 3 | `mfsim run\|steady\|stats\|export-case\|nb`, corridas npz v2, `tools/` e `archive/scratch_2026-10` (nada apagado) | feito |
| 4 | notebooks pelo carregador único e API pública, re-executados; outputs de texto iguais (numpy) ou explicados (células JAX) | feito (tag `v4-notebooks`) |
| noturna | referência 3000 s JAX, gas lift 2000 s, validação de laboratório (Fabre 2, Jansen), 9 notebooks re-executados contra os outputs de texto | verde 14/14 em 2026-10-14 |
| 5 | remoção da compatibilidade: `jaxrhs` e módulos JAX duplicados apagados, documentação dividida em `docs/` | feito em parte: o `NetworkModel` ainda consome o `Case` legado (`network/data.py`), convertido bit a bit do grafo por `Network.to_legacy_case`; a montagem da rede diretamente sobre o grafo (resolução de portas, `NetworkSystem`) fica como trabalho seguinte |

## Pendências do modelo físico (status)

Lacunas do modelo a dois fluidos levantadas em 2026-10-13, por ordem do que
mais pode mudar o severe slug; o status é atualizado aqui e na lista de
pendências do documento-resumo.

| item | o que é | status |
|---|---|---|
| 1 | Escoamento contracorrente no riser: fechamentos co-correntes, sem inundação nem filme descendente (reenchimento do riser) | feito (2026-10-13): o problema era o padrão de bolhas a 95 % de gás; anular acima de 90 % (`high_void_regime`) deixa o filme drenar e a inundação emerge do balanço de forças; intervalo +10 % |
| 2 | Sensibilidade ao atrito interfacial do estratificado: multiplicador ×0.5/×2; Taitel–Dukler (f_I = f_G) contra Andritsos–Hanratty da tese | feito (2026-10-13): ciclo insensível, ±2 % no intervalo, ±8 % nos picos |
| 3 | Massa virtual (Drew–Lahey, C_vm ≈ 0.5 em bolhas) como regularização física no lugar da pressão interfacial de Bestion: mapa de hiperbolicidade e ciclo | feito (2026-10-13): forma objetiva hiperboliza todo o mapa; intervalo +7–10 %, mesmo custo |
| 4 | Lei do gás: γG adiabático (1.3) contra quase isotérmico (1.05) no pico do blowdown | feito (2026-10-13): γ 1.3 encurta o intervalo 9–18 % e baixa os picos 10 %; γ 1.0 alonga 4 % |
| 5 | Fechamento do riser no blowdown: intermitente (`regime_vertical=3`) contra bolhas, isolando o efeito do fechamento do efeito do IPE | feito (2026-10-13): no caso base (fora de Bøe) só bolhas dão ciclo; no caso Bøe-consistente (L = 1000 m, 2 bar) os dois fechamentos dão o ciclo clássico com período ±10 % |
| 6 | Contorno de entrada com volume de gás a montante (gasoduto compressível, parâmetro do critério de Bøe) em vez de vazões constantes | feito (2026-10-13): o vértice de volume é um tanque de mistura (fração de gás → 0.05) e não acumula gás; o gasoduto a montante representa-se prolongando `L_h1` (resultados na tabela da fronteira); o vértice separador fica pendente |
| 7 | Arraste e deposição de gotas no anular (forma do pico de produção) | avaliado (2026-10-13): fração arrastada de Ishii–Mishima desprezível (E > 0.1 em ≤ 1 % do tempo com riser cheio de gás; 1–5 % do líquido produzido); sem modelo de gotas |
| 8 | Validação com dados experimentais (Jansen–Shoham–Taitel 1996, Fabre 1990, Tengesdal 2002) | feito (2026-10-13): [docs/lab_validation.md](docs/lab_validation.md). Com o fechamento de bolhas da tese nenhum caso cicla (gás a 4 m/s com 3 % de fração a 1 atm); com `bubble_drag="fisico"` (sem efeito no caso de campo) Fabre 2 dá 135.8 s contra 137 medidos, Tengesdal 64–68 s contra 79, Jansen 56–63 s contra 46.6 (ensaio com choke), Fabre 1 o tipo certo (penetração contínua, +50 %). Critério de Bøe corrigido. Numérica a 1 atm: banda de undershoot ligada a γG/γL e recálculo da Jacobiana no Newton (`slow`). Por fazer: choke no topo (item 9) para fechar a comparação com Jansen |
| 9 | Mitigação com os elementos existentes: choke no topo e gas lift na base | não iniciado |
