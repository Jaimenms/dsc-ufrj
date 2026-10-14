# Formatos de dados: casos em JSON, corridas gravadas e referências de regressão

*Documentação do mfsim (índice no [README](../README.md)).*

## Casos em JSON (`mfsim-case/1`) e grafo da rede

Reorganização em curso (plano em `~/.claude/plans/keen-splashing-mango.md`; fases 0 e 1
feitas em 2026-10-13). Todas as entradas de um caso cabem num ficheiro JSON:
`fluid`, `reference` (PR e faixas `state_ref`), `time`, `library` (blocos `models` e
`schemes` reutilizáveis por `"$ref"`), `network` (`vertices` de tipo `boundary`,
`junction` ou `tank`; `edges` de tipo `pipe`, `connector`, `choke`, `reservoir`,
`check_valve`, todas com `from`/`to`), `integration`. Unidades SI; especificações no
tempo como escalar, `{"t": [...], "v": [...]}` ou `{"ramp": [t1, t2], "from": a, "to": b}`;
pressões como `"P"` (Pa) ou `"PT"` (P/PR). A condição inicial de um duto é uma receita
(`{"recipe": "blocked" | "shut_in", ...}`, `mfsim/initial/recipes.py`) ou arrays num
npz ao lado (`{"arrays": {"file": ..., "key": ...}}`).

- `cases/chapter10/*.json`: os 11 casos da tese exportados dos .mf2 por
  `uv run python -m mfsim.io.legacy_mf2` (com a composição do 10.3.2a e as correções de
  `load_chapter10_case`; `meta.provenance`); `load_chapter10_json(chave)` lê-os.
- `cases/severe_slug/*.json`: referência (config recomendada com arrasto físico de
  bolhas), variantes e os casos de laboratório, exportados das corridas gravadas
  (`uv run python -m mfsim.cases.severe_slug --export-runs`); casos novos por
  `generate(**parâmetros de severe_slug_case)` ou pela linha de comando
  (`python -m mfsim.cases.severe_slug --out x.json --param D=0.05 ...`).
- `mfsim.io.case_from_json(caminho)` devolve o `Case` que `simulate`/`build_model`
  consomem; `case_to_json(case_ou_grafo, caminho)` grava. O grafo (`mfsim/network/graph.py`:
  `Vertex` → `BoundaryVertex`/`JunctionVertex`/`TankVertex`; `Edge` → `Pipe`/`Connector`/
  `Choke`/`Reservoir`/`CheckValve`) converte-se sem perda para o `Case` legado: o legado
  deixava implícitas as junções de choke (`Jc<v>`) e de reservatório (`Jr<v>`), que o
  grafo explicita. `tests/test_json_roundtrip.py` verifica, em todos os casos, que grafo e
  JSON reconstroem o mesmo modelo (rhs bit-idêntico).
- Rede de segurança (fase 0): `tests/golden/` gerados por `tools/golden_generate.py`
  (nível A = hash bit a bit, `policy.json`, `CHANGELOG.md`), `tools/manifest.py`
  (`MANIFEST.sha256` dos dados fora do git), `mfsim/runs.py` (carregador único das corridas
  gravadas; `gammaL = 100` por omissão, como o notebook), `uv run pytest -m nightly`.

## Corridas gravadas (`notebooks/data/*.npz`, `runs/*.npz`)

Formato v1 (corridas históricas): arrays `T` (nt,), `X` (n_states, nt), opcional `acc`
(integral BDF-consistente dos fluxos de contorno) e `meta` (JSON com as opções da
corrida nos nomes do antigo driver; defaults do notebook, em particular γL = 100).
Formato v2 (`mfsim run`, `mfsim.runs.save_run`): os mesmos arrays mais `info_json`
(passos, rejeições, CPU, sucesso/erro), `provenance_json` (commit, versões, comando) e
`case_json` — o caso completo em `mfsim-case/1` (condições iniciais explícitas em
`ic/<duto>`), de modo que `mfsim.runs.load_run` reconstrói caso, modelo e resultados
sem o `meta`. `load_run(nome)` procura em `runs/`, `notebooks/data/` e no arquivo da
antiga `scratch/` (ficheiros do driver com `opts`, traduzidos por
`meta_from_severe_opts`). As 80 trajetórias históricas **não são reproduzíveis bit a
bit** pelo código atual (dinâmica caótica: 15 % de desvio pontual a 150 s) e são
tratadas como dados; a identidade do código garante-se pelo lado direito avaliado nos
estados guardados e por assinaturas de ciclo (`mfsim.analysis.cycle_signature`).

## Referências de regressão (`tests/golden/`)

Geradas só por `tools/golden_generate.py` (`--rebaseline --note "..."` para regenerar;
cada regeneração fica registada em `CHANGELOG.md` com o desvio máximo). `HASHES.json`
guarda o sha256 dos bytes de cada array e `policy.json` o nível de comparação: **A** =
bit a bit (padrão), **B** = tolerância relativa (`{"tier": "B", "rtol": ...}`; usada para o
rhs/Jacobiana do caminho JAX, 2e−15 do anterior), **C** = assinaturas de ciclo para as
corridas longas (1.º blowdown ±2 s, P_in mín/máx ±0.01 bar, pico ±2 %). Grupos:
`chapter10` (12 trajetórias + variantes), `severe` (referência 150 s numpy e JAX,
rhs/transformação/escalas em 20 estados históricos, Jansen 10 s), `results`, `runs_meta`
(76 corridas gravadas reconstruídas do `meta`), `notebook_text` (outputs de texto dos
notebooks, tempos normalizados), `nightly` (referência 3000 s, gas lift 2000 s). Rotina:
`uv run pytest` (≈ 3 min); noturna: `uv run pytest -m nightly`. Os dados fora do git
(`notebooks/data`, `.mf2`, arquivo) são fixados por `MANIFEST.sha256`
(`uv run python tools/manifest.py --check`).

## Montagem da rede sobre o grafo (`network/system.py`, `network/ports.py`)

O sistema de EDOs é montado diretamente sobre o `Network`: o vetor de estados tem os
nós livres de cada duto (na ordem do grafo) e, no fim, (HG, PT) por tanque — o layout
legado, logo as comparações com `estados.X` do MATLAB mantêm-se. As portas (alphaG, PT,
qL, qG) de cada vértice formam o vetor de especificações E, preenchido por: tabelas dos
vértices (`Schedule`; ausente = UNSPECIFIED), vazões externas dos tanques não
especificadas = 0, os passos das arestas de comprimento zero em ordem topológica
(`ConnectorPressure`, `ChokeStep`, `ReservoirStep`, `CheckValveStep`, `ConnectorFlows`;
desempate pela ordem legada das classes e das arestas) e, por fim, os estados dos
tanques. A pressão de um vértice lida por uma aresta é o estado do primeiro nó de um
duto que parte dele (diretamente ou pela junção de um reservatório), senão a porta PT;
as vazões que chegam a um vértice são as do último nó dos dutos que lá chegam
diretamente ou pela junção de um choke. Cada porta tem um só escritor (conflitos dão
`ValueError` na construção). Uma `JunctionVertex` pode trazer `spec` com os valores das
portas que a aresta não fornece — nos casos convertidos do legado são as especificações
do vértice que a junção desdobra (`Jc<v>`, `Jr<v>`), e é assim que o estado estacionário
inicial e a estimativa inicial ficam iguais aos de antes. Nas saídas por
vértice (`Results.vertex`), um vértice desdobrado em junção é representado pela linha da
sua junção (pressão a montante do choke, vazões do reservatório), como o vértice legado e a
tese reportavam (`NetworkSystem.report_row`).

## O grafo de ponta a ponta e o registo legado

`load_chapter10_case`, `load_case` (`.mf2`), `case_from_json`, `severe_slug_case` e
`mfsim.runs.load_run(...).case` devolvem um `Network`; `build_model`, `simulate`,
`steady_state`, `integrate_run`, `save_run` e a linha de comando consomem-no diretamente.
Edição conveniente no dialeto legado (o que os notebooks e testes faziam no `Case`):
`net.set_method("fv_upwind")`, `net.set_time(tf=...)`, `net.set_integration(rtol=...)`,
`net.set_options(conservative=True, q_switch=0.05, bb_derivatives="fd", regime=6)` (reconstrói
`ModelSpec`/`SchemeSpec` com as chaves alteradas; `from_legacy(to_legacy())` é a identidade),
`net.pipes[0].ports_in = ports((3, 1, 2, 2))`, `net.vertices[0].spec["qG"] = Constant(0.13)`,
`net.edges_of(Choke)[i].opening = PiecewiseLinear(t, v)`; construção direta com
`Network.build(...)`, `Pipe.legacy(...)`, `BoundaryVertex(id, spec=specs(PT=...))`. O registo
legado (`mfsim.io.legacy_case`: `Case`, `Vertex`, `Pipe`, `Valve`, ...; o struct `dados`) é só o
formato intermédio do leitor dos `.mf2` e do exportador `Network.to_legacy_case` (round-trip
testado); nada no simulador o consome.
