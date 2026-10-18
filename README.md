# Modelagem e Simulação de Escoamento Multifásico em Dutos de Produção de Óleo e Gás Natural

Tese de doutorado de **Jaime Neiva Miranda de Souza** (UFRJ, 2010) e a sua reimplementação
moderna em Python: o simulador **mfsim**, a reprodução bit a bit dos casos da tese contra os
resultados originais em MATLAB, e a resolução do problema que a tese deixou em aberto, a
intermitência severa (*severe slugging*), validada com dados de laboratório.

## A tese

| | |
|---|---|
| título | Modelagem e Simulação de Escoamento Multifásico em Dutos de Produção de Óleo e Gás Natural |
| autor | Jaime Neiva Miranda de Souza |
| grau | Doutor em Ciências, Programa de Pós-Graduação em Tecnologia de Processos Químicos e Bioquímicos (TPQB), Escola de Química, Universidade Federal do Rio de Janeiro |
| orientadores | Prof. José Luiz de Medeiros, D.Sc. (presidente); Prof. André Luiz Hemerly Costa, D.Sc.; Prof. Giovani Cavalcanti Nunes, Ph.D. |
| banca | Prof. Alexandre Rodrigues Tôrres, Prof. Carlos Eduardo Fontes da Costa e Silva, Prof. Heloísa Lajas Sanches, Prof. Luiz Fernando Lopes Rodrigues Silva, Prof. Su Jian |
| defesa | Rio de Janeiro, fevereiro de 2010 |
| apoio | PETROBRAS (incentivo e apoio financeiro); Chemtech Serviços de Engenharia e Software |
| texto | [docs/thesis.pdf](docs/thesis.pdf) (xxvi + 265 p.) |

A tese propõe modelos dinâmicos de escoamento monofásico e bifásico aplicados a **redes de dutos**
de produção de óleo e gás, com qualquer topologia e com elementos modificadores de pressão
(válvulas, reservatórios, válvulas de retenção, acoplamentos): modelo a dois fluidos a quatro
equações com fechamentos por padrão de escoamento e identificação automática de padrão, modelo de
mistura a três equações (Beggs & Brill) e modelos monofásicos, discretizados pelo método das
linhas (Galerkin, Petrov-Galerkin, diferenças finitas) e integrados por BDF implícito. Sobre esse
modelo dinâmico constrói um modelo estacionário de redes e um arcabouço de otimização de sistemas
de elevação por gas lift contínuo. O capítulo 10 apresenta os casos de simulação dinâmica (duto
horizontal nos padrões estratificado, anular e bolhas; redes em série e com ciclo; poço simples,
poço com gas lift e poço offshore), todos reproduzidos aqui. A motivação é a falta, na literatura
aberta, de simuladores de redes bifásicas capazes de descrever o petróleo brasileiro e a
profundidade dos poços nacionais sem a arquitetura fechada das ferramentas comerciais.

## Este repositório

O código original (MFSIM/SDMPFLOW, MATLAB, 2010) e os seus resultados gravados foram a referência
para o **mfsim**, reimplementação em Python com o solver próprio da tese: modelo a dois fluidos
(4 equações), modelo de mistura (3 equações) e monofásicos; Galerkin, diferenças finitas e volumes
finitos escalonados; BDF implícito próprio com lado direito em numpy ou compilado com JAX. Os
casos do capítulo 10 reproduzem-se contra os ficheiros MATLAB; as 61 alterações em relação à tese
(correções, melhorias e estudos) estão em [docs/issues/INDEX.md](docs/issues/INDEX.md), e a
resposta à pergunta que a tese não conseguiu fechar, a intermitência severa em geometria
linha–riser, em [docs/why_no_severe_slug_in_thesis.md](docs/why_no_severe_slug_in_thesis.md).

## Começar

```bash
uv sync                                                        # ambiente e pacote (editável)
uv run mfsim validate cases/examples/hello_world.json          # o caso obedece ao schema?
uv run mfsim run cases/examples/hello_world.json --out runs/hello_world.npz
```

Um caso é **um ficheiro JSON** (`mfsim-case/1`) com tudo o que a simulação precisa: fluido,
rede, tempo, integrador e estado inicial. [docs/hello_world.md](docs/hello_world.md) explica
esse primeiro caso bloco a bloco, como ler os resultados e como construir o mesmo caso em
Python; [docs/data_formats.md](docs/data_formats.md) tem o vocabulário completo.

```bash
uv run pytest                                  # rotina (≈ 3 min): regressão bit a bit, paridade JAX, CLI
uv run pytest -m nightly                       # noturna (≈ 25 min): corridas longas, laboratório, notebooks
uv run black --check mfsim tests tools notebooks_src   # formatação (120 colunas)
uv run mfsim batch batches/thesis.json                         # lote da tese -> runs/chapter10 (os notebooks só leem)
uv run mfsim batch batches/issues.json                         # lote dos notebooks de issues -> runs/issues (~6 min)
uv run python tools/slug_map.py build && uv run mfsim batch runs/maps/well/steady/batch.json --jobs 12   # mapa (043)
uv run mfsim nb notebooks_src/thesis/nb_10_1_1.py --execute    # notebooks a partir de notebooks_src/**/*.py
```

## Linha de comando

```bash
# simular (grava a corrida em npz v2; --set altera qualquer campo do caso)
uv run mfsim run cases/severe_slug/reference.json --backend jax --tf 150 --out runs/ref150.npz
uv run mfsim run cases/severe_slug/lab_jansen.json --backend jax           # laboratório (shut_in, slow)
uv run mfsim run cases/severe_slug/reference.json --set integration.integrator=radau --set integration.rtol=1e-5
uv run mfsim steady cases/chapter10/10_1_1.json --out /tmp/10_1_1_ss.json  # estacionário -> caso com a condição inicial

# analisar corridas gravadas (nomes em notebooks/data ou caminhos .npz)
uv run mfsim stats severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys   # ciclos
uv run mfsim scan  severe_slug_blocked_conservative_arc_scan_P2_g005           # varredura da fronteira (Bøe, padrão, ciclos)
uv run mfsim riser severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys   # filme no riser (contracorrente)
uv run mfsim val   severe_slug_lab_fabre2_fisico --plot                        # validação de laboratório + figura

# casos e corridas
uv run mfsim validate cases/chapter10/*.json cases/severe_slug/*.json      # schema mfsim-case/1
uv run python -m thesis.export --chapter10                                 # .mf2 da tese -> cases/chapter10/*.json
uv run python -m generators.severe_slug --out novo.json --param D=0.05 --param start=blocked   # gerador (fora do simulador)
uv run python tools/convert_runs.py antiga.npz --out-dir runs/            # corrida v1 (meta) -> v2 (case_json)
```

- **Um só caminho de execução.** `simulate(case)` lê tudo do caso, em numpy ou JAX; CLI,
  notebooks e testes usam-no. Uma corrida gravada (npz v2: `T`, `X`, `acc`, `info_json`,
  `provenance_json`, `case_json`) reconstrói caso, modelo e resultados sem outra fonte
  (`mfsim.runs.load_run`); uma falha do integrador grava a trajetória parcial.
- **Casos sem repetição.** O schema `mfsim/io/schema/mfsim-case-1.schema.json` é verificado ao
  carregar, ao exportar e por `mfsim validate`. Uma variante herda o caso base com
  `"$extends"` e traz só o que muda (as de `cases/severe_slug` têm poucas linhas);
  `--set caminho=valor` varia um caso sem ficheiro novo.
- **Dados.** As 76 corridas históricas de `notebooks/data` estão em v2; os dados fora do git
  (`notebooks/data`) e os `.mf2` da tese (`thesis/data`) são fixados por `MANIFEST.sha256`
  (`uv run python tools/manifest.py --check`). A antiga `scratch/`, os diagnósticos, as cópias v1 e
  os notebooks anteriores foram removidos em 2026-10-14 (os ficheiros versionados ficam no
  histórico do git até ao commit 22ecbc9).

## Estrutura

| módulo | conteúdo |
|---|---|
| `mfsim/core/` | slots do estado, `PortKind` (livre/Dirichlet/Neumann), dataclasses de opções (modelo, esquema, tempo, referência, integração) |
| `mfsim/fluid.py`, `geometry.py`, `friction.py` | equação de estado politrópica (ρ, C = dρ/dP); nível estratificado em seção circular e malha do duto; fator de atrito de Fanning (Churchill) |
| `mfsim/closures/` | fechamentos por padrão (estratificado, anular, bolhas, intermitente) e identificação do padrão (Taitel, Pots, critérios de estabilidade) |
| `mfsim/models/` | modelos de duto: dois fluidos (4 eq.), mistura Beggs & Brill (3 eq.), monofásicos |
| `mfsim/discretization/` | Galerkin, Petrov-Galerkin, diferenças centrais e regressivas (métodos da tese) e volumes finitos escalonados com upwind pela velocidade de fase (`fv_upwind`, `fv_upwind_cons` com massas por volume) |
| `mfsim/network/` | grafo `Vertex`/`Edge` (`graph.py`: fronteira, junção, tanque; duto, conector, choke, reservatório, retenção; `Network.build`, `set_method`/`set_time`/`set_options`), especificações no tempo (`schedules.py`), relações das arestas de comprimento zero (`elements.py`), resolução de portas e plano estático de avaliação (`ports.py`), `NetworkSystem` (`system.py`: layout de estados, `time_inputs(t)`, núcleo puro `rhs_core`, mapa conservativo, escalas, esparsidade) |
| `mfsim/backend/`, `mfsim/compiled.py` | backend numérico injetado (`NUMPY`, `JAX`): fluido, fechamentos, modelos, esquemas e rede escritos **uma só vez**; `CompiledNetwork` reconstrói o mesmo `NetworkSystem` em JAX com `jit` e Jacobiana exata por `jacfwd` |
| `mfsim/integrators/`, `steady.py`, `initial_guess.py` | BDF implícito próprio (ordem 1–2, passo adaptativo, Newton modificado, Jacobiana colorida ou externa) e invólucro do `solve_ivp`; estado estacionário (Newton + pseudo-transiente); estimativa inicial |
| `mfsim/simulation.py`, `results.py` | `build_model`, `simulate`, `steady_state`; `Results` com as saídas da tese por duto e por vértice |
| `mfsim/io/`, `mfsim/initial/` | casos em JSON (`spec.py`; `$extends`/`$ref`/`$include` em `extends.py`; schema e `validate.py`); receitas de condição inicial (bloqueada, linha fechada) |
| `mfsim/runs.py`, `analysis.py`, `cli.py` | corridas gravadas v2 (`load_run`, `save_run`, `Run.case`/`network`/`res`); assinaturas de ciclo e análises (varredura, riser, laboratório); `mfsim run\|steady\|stats\|scan\|riser\|val\|validate\|nb` |
| `thesis/` | ponte com o MFSIM da tese, **fora do simulador**: os `.mf2` originais em `thesis/data/` e a sua leitura (`mf2.py`: `load_mf2`, `chapter10_case`, `legacy_results`), registo legado e conversão para o grafo (`record.py`), exportação para JSON (`python -m thesis.export --chapter10`), apoio aos notebooks de reprodução (`notebook_utils.py`) |
| `mfsim/plotting.py`, `notebook_utils.py` | figuras no estilo da tese e apoio aos notebooks |
| `cases/`, `generators/` | JSON: `examples/hello_world`, `chapter10/` (11 casos da tese), `severe_slug/` (referência, variantes como `$extends`, laboratório); geradores fora do simulador (`generators/severe_slug.py`: parâmetros físicos → JSON; `generators/column.py`: coluna de líquido da issue 006), `cases/studies/` com os seus casos |
| `batches/` | lotes de corridas em JSON executados por `mfsim batch` para `runs/`: `thesis.json` (casos da tese tal e qual, sensibilidades de malha, formulação 2 do 10.1.1) e `issues.json` (coluna de líquido, capítulo 10 em numpy e JAX, estacionários, poço de deriva, gas lift 16 000 s) |
| `tools/`, `tests/`, `notebooks_src/` | `golden_generate` (referências de regressão), `manifest`, `make_nb`, `slug_map` (mapa de intermitência severa, issue 043), migração do dialeto antigo (`legacy_runs`, `convert_runs`, `severe_run`, `save_run_data`, `export_severe_cases`); testes com `tests/golden/` (nível A = bit a bit, B = tolerância, C = assinaturas de ciclo; `CHANGELOG.md`); fontes dos notebooks: `thesis/` reproduz os casos da tese tal e qual (um por secção), `issues/` ilustra as correções e melhorias (um por grupo de issues, cada um a apontar para `docs/issues/`); nenhum simula, todos leem `runs/` e `notebooks/data` |
| `docs/` | documentação (índice abaixo), índice de issues e o texto da tese (`docs/thesis.pdf`) |

## Documentação

- [docs/issues/INDEX.md](docs/issues/INDEX.md) — o que mudou em relação ao modelo e ao solver da tese: 61 alterações atómicas (feature, bugfix, estudo), cada uma com antes, depois e benefício medido.
- [docs/method_thesis_to_modern_code.md](docs/method_thesis_to_modern_code.md) — o método que nos trouxe até aqui (tese + código legado → versão moderna e melhorada) e um prompt consolidado para reutilizar; [CLAUDE.md](CLAUDE.md) é o guia do projeto para sessões do Claude.
- [docs/why_no_severe_slug_in_thesis.md](docs/why_no_severe_slug_in_thesis.md) — por que a tese não conseguiu simular a intermitência severa, em quatro pontos ligados às issues.
- [docs/hello_world.md](docs/hello_world.md) — primeiro caso: um duto e duas fronteiras em JSON, correr, ler os resultados, o mesmo caso em Python, variar com `--set`/`$extends`.
- [docs/data_formats.md](docs/data_formats.md) — casos em JSON (`mfsim-case/1`, schema, herança), grafo e montagem da rede, corridas gravadas (v2, conversão de v1), referências de regressão e manifesto.
- [docs/thesis_reproduction.md](docs/thesis_reproduction.md) — modelo e layout de estados, reprodução do capítulo 10, gas lift com intermitência forte, observações sobre os ficheiros originais.
- [docs/jax_backend.md](docs/jax_backend.md) — backend único numpy/JAX: como usar, garantias de paridade (detalhe nas issues 036–038).
- [docs/severe_slug.md](docs/severe_slug.md) — intermitência severa: o caso, a configuração recomendada e o mapa dos temas para as issues.
- [docs/lab_validation.md](docs/lab_validation.md) — validação com dados de laboratório (Jansen, Fabre, Tengesdal): casos, comandos e resultado em uma linha.
- [docs/findings_for_paper.md](docs/findings_for_paper.md) — achados para o artigo (resumo).
- [papers/README.md](papers/README.md) — artigos para revistas: mecanismo de construção (LaTeX Elsevier CAS, figuras e tabelas geradas por código a partir das corridas), escolha da revista e o artigo 01 (intermitência severa com padrão de escoamento dinâmico: boa colocação, discretização, integrador, JAX e mapa de estabilidade com choke).
- [docs/future/README.md](docs/future/README.md) — planos para temas futuros, não iniciados: chill-down criogénico (hidrogénio líquido), despressurização de dutos de CO₂, composição e transferência de massa, água como terceira fase, catálogo de assinaturas de golfada e alocação de poços a separadores (linha de controle anti-slug e gêmeo digital).
- `tests/golden/CHANGELOG.md` — registo das referências de regressão; `docs/thesis.pdf` — a tese.

## Reorganização do código (2026-10)

O plano aprovado (OO com injeção de dependências, grafo `Vertex`/`Edge`, casos em JSON,
backend único numpy/JAX, `scratch/` arquivada, compatibilidade temporária depois removida)
está executado. Cada fase fechou com a suíte de regressão bit a bit verde; a noturna
(referência 3000 s em JAX, gas lift 2000 s, Fabre 2 e Jansen, 9 notebooks contra os
outputs de texto) está verde (14/14) em 2026-10-14.

| fase | conteúdo | tag |
|---|---|---|
| 0 | git, manifesto dos dados, referências de regressão (níveis A/B/C), `mfsim.runs` | `v0-baseline`, `v0-results-snapshot` |
| 1 | opções tipadas, grafo + especificações no tempo, JSON `mfsim-case/1` e conversão dos `.mf2`, receitas de condição inicial; round-trip bit-idêntico | — |
| 2 | backend único: fluido, fechamentos, padrão, modelos, discretizações e rede numa só implementação; `compiled.py`; módulos JAX duplicados apagados; numpy bit-idêntico, JAX paridade 1e−14, desempenho ≤ 1.1× | `v2-backend` |
| 3 | CLI, corridas npz v2, `tools/` revistas e `scratch/` arquivada (arquivo removido em 2026-10-14; histórico do git até ao commit 22ecbc9) | — |
| 4 | notebooks pelo carregador único e API pública, re-executados; outputs de texto iguais ou explicados (células JAX) | `v4-notebooks` |
| 5 | compatibilidade removida; documentação dividida em `docs/`; rede montada diretamente sobre o grafo (`system.py` + `ports.py`, junções explícitas) com o golden `network` bit-idêntico; `Network` de ponta a ponta e registo legado só como formato de entrada | `v5-refactored`, `v5-network-native` |
| 6 | caminho único de execução (`simulate` só lê o caso), corridas só em v2 com as 76 históricas convertidas, schema com validação, herança `$extends`, `--set`, primeiro caso documentado | `v6-single-driver` |

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
| 9 | Mitigação com os elementos existentes: choke no topo e gas lift na base | choke feito (2026-10-16): mapa abertura × pressão do reservatório ([043](docs/issues/043-mapa-de-intermitencia-severa.md)) com a lei de válvula homogénea ([039](docs/issues/039-lei-de-valvula-homogenea.md)); a fronteira segue o critério de Jansen et al. (1996); gas lift não iniciado |

## Licença

MIT ([LICENSE](LICENSE)).
