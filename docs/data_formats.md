# Formatos de dados: casos em JSON, corridas gravadas e referências de regressão

*Documentação do mfsim (índice no [README](../README.md)).*

## Casos em JSON (`mfsim-case/1`) e grafo da rede

Reorganização em curso (plano em `~/.claude/plans/keen-splashing-mango.md`; fases 0 e 1
feitas em 2026-10-13). Todas as entradas de um caso cabem num ficheiro JSON:
`fluid`, `reference` (PR e faixas `state_ref`), `time`, `library` (blocos `models` e
`schemes` reutilizáveis por `"$ref"`), `network` (`vertices` de tipo `boundary`,
`junction` ou `tank`; `edges` de tipo `pipe`, `connector`, `choke`, `reservoir`,
`check_valve`, todas com `from`/`to`; o `choke` leva `CV`, `opening` e `law`, `"homogeneous"` por
omissão ou `"thesis"` nos casos da tese, issue 039), `integration`. Unidades SI; especificações no
tempo como escalar, `{"t": [...], "v": [...]}` ou `{"ramp": [t1, t2], "from": a, "to": b}`;
pressões como `"P"` (Pa) ou `"PT"` (P/PR). A condição inicial de um duto é uma receita
(`{"recipe": "blocked" | "shut_in", ...}`, `mfsim/initial/recipes.py`) ou arrays num
npz ao lado (`{"arrays": {"file": ..., "key": ...}}`), nos nós da malha ou, com `"at": "scheme"`, nos nós do
esquema (estado exato; issue 040).
Com `time.settle` (`window`, `t_min`, `amp_tol`, `rel_tol`, `signal`) a corrida integra em janelas e
pára quando o sinal fica permanente ou o ciclo se repete; `tf` passa a ser o tempo máximo (issue 041).
O estado inicial da corrida é o dos dutos (`initial_state.source = "edges"`), a estimativa
automática (`"guess"`) ou o estacionário por Newton a partir de um deles (`"steady"`, com
`from`, `perturb`, `max_iter` e `on_failure`; issue 040).

A secção `integration` é completa: `integrator` (`bdf`/`implicit` = BDF próprio, `radau`,
`LSODA`), `quadrature_order`, `max_step` (`null` = sem limite), `atol`, `rtol`, `backend`
(`numpy`/`jax`) e os campos opcionais do BDF (`tol_range`, `atol_P`, `atol_q`, `first_step`,
`max_order`, `max_newton`, `newton_tol`, `min_step`, `slow_exit`, `predict_space`,
`jt_refresh`, `accumulate`, `tight_bounds`); um campo ausente usa o default do integrador e
`simulate(case)` não lê mais nada (`mfsim/simulation.py`). `initial_state.source` escolhe
entre as condições iniciais dos dutos/tanques (`edges`) e a estimativa de escoamento
estabelecido (`guess`).

O estilo dos ficheiros é amarrado pelo schema `mfsim/io/schema/mfsim-case-1.schema.json`
(JSON Schema 2020-12): vocabulário dos modelos, padrões, esquemas e respetivas opções,
tipos de porta, tipos de vértice e de aresta com os campos permitidos em cada um, receitas
de condição inicial, campos de integração e os seus limites. `case_from_json` valida ao
carregar (`validate=False` desliga), `case_to_json` antes de escrever, `mfsim validate
<ficheiros>` imprime as violações com o caminho (`network/edges/0/scheme/options: ...`) e
`tests/test_schema.py` valida todos os ficheiros de `cases/`, um caso gerado e casos
deliberadamente errados. A coerência estrutural (ids referidos, tabelas) fica com
`Network.validate` e os construtores.

Variantes sem repetição: um ficheiro com `"$extends": "reference.json"` herda o caso base
(da mesma pasta ou por caminho) e traz só o que muda (`mfsim/io/extends.py`): dicionários
fundem-se chave a chave, as listas de vértices e arestas por `id`, o resto substitui-se;
`"$replace": true` num dicionário (ou `{"$replace": [...]}` numa lista) substitui o bloco
inteiro, que é como um remendo apaga chaves (`null` é valor legítimo, p. ex. em `max_step`).
Cadeias (`a` herda de `b` que herda de `c`) e várias bases são permitidas; a cadeia fica em
`meta.extends`, a validação é sobre o caso expandido e as corridas gravam o caso expandido.
`case_to_json(net, path, extends="reference.json")` escreve o remendo mínimo (`make_patch`,
verificado por fusão antes de gravar); as seis variantes de `cases/severe_slug` (`EXTENDS` em
`tools/export_severe_cases.py`) são remendos de 0.4–3 kB sobre `reference`, `lab_fabre2` e
`lab_jansen`, p. ex. `lab_fabre2_thesis_bubbles.json` = nome, `meta` e
`closures.bubble.drag`. Para variar um caso sem ficheiro novo: `mfsim run caso.json --set
integration.integrator=radau --set network.edges.1.D=0.06` (qualquer campo, por caminho; um
segmento sobre uma lista é o `id` do elemento ou, se nenhum coincidir, o índice), também em
`mfsim steady` e em `case_from_json(..., overrides=[...])`; `tests/test_extends.py`.

- `cases/chapter10/*.json`: os 11 casos da tese, exportados dos .mf2 por
  `uv run python -m thesis.export --chapter10` (`thesis.export.network_from_mf2`: condição
  inicial de `estados.X[:, 0]`, `LEGACY_OVERRIDES`, composição do 10.3.2a; registado em
  `meta.provenance`). São a **fonte** dos casos: a descrição (secção, título, figuras,
  descrição) vive em `meta` e a exportação preserva-a. `thesis.mf2.chapter10_keys()`,
  `chapter10_case(chave)` e `chapter10_info(chave)` leem-nos; `legacy_results(chave)` carrega
  o .mf2 indicado em `meta.provenance.source` só para comparar com os resultados do MATLAB.
- `cases/severe_slug/*.json`: referência (configuração recomendada), variantes (`$extends`) e
  casos de laboratório. Foram gerados pelo gerador paramétrico `generators/severe_slug.py`
  (fora do simulador: `severe_slug_case(**parâmetros)` constrói o grafo, `generate` acrescenta
  a receita de condição inicial e regista os parâmetros em `meta.generator_params`); casos
  novos por `uv run python -m generators.severe_slug --out x.json --param D=0.05 ...`. A
  corrida de origem de cada ficheiro está em `meta.run` (reexportação em
  `tools/export_severe_cases.py`).
- `mfsim.io.case_from_json(caminho)` devolve o `Network` que `simulate`/`build_model`
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

Formato v2 (`mfsim run`, `mfsim.runs.save_run`): arrays `T` (nt,), `X` (n_states, nt),
opcional `acc` (integral BDF-consistente dos fluxos de contorno), `info_json` (passos,
rejeições, CPU, sucesso/erro), `provenance_json` (commit, versões, comando) e `case_json` —
o caso completo em `mfsim-case/1` com a secção `integration` que o produziu e as condições
iniciais explícitas em `ic/<duto>`. `mfsim.runs.load_run` só lê este formato e reconstrói
caso (`Run.case`, com `tf = T[-1]`), rede para continuar/variar (`Run.network(tf=...)`),
modelo e resultados; `Run.meta` devolve os parâmetros do gerador (`meta.generator_params`)
para as análises. `load_run(nome)` procura em `runs/` e `notebooks/data/`.

Formato v1 (corridas históricas até 2026-10-14): os mesmos arrays e `meta` (JSON com as
opções da corrida nos nomes do antigo driver; defaults do notebook, em particular
γL = 100). O dialeto está isolado em `tools/legacy_runs.py` (`case_from_meta`,
`case_from_severe_opts` para os ficheiros do driver com `opts`, `apply_driver_integration`
com os defaults do driver) e só o conversor o usa: `uv run python tools/convert_runs.py <ficheiros>
[--out-dir]` (`convert_run`) reescreve em v2 com `case_json`, `info_json` e proveniência da
conversão. As 76 corridas de `notebooks/data` foram convertidas no lugar (T/X/acc
bit-idênticos; modelos reconstruídos iguais ao golden `runs_meta`), com cópias v1 em
cópias v1 removidas em 2026-10-14); os quatro
`severe_slug_run*.npz` sem `meta` completo ficam em v1 e o notebook lê-os diretamente.
As 80 trajetórias históricas **não são reproduzíveis bit a bit** pelo código atual
(dinâmica caótica: 15 % de desvio pontual a 150 s) e são tratadas como dados; a
identidade do código garante-se pelo lado direito avaliado nos estados guardados e por
assinaturas de ciclo (`mfsim.analysis.cycle_signature`).

## Lotes de corridas (`batches/*.json`, `mfsim batch`)

Um lote declara corridas a partir de casos JSON, para que os notebooks só leiam resultados:

```json
{"out_dir": "runs/chapter10",
 "runs": [{"name": "10_1_1", "case": "cases/chapter10/10_1_1.json"},
          {"name": "10_1_1_galerkin_n5", "case": "cases/chapter10/10_1_1.json", "steady": true,
           "set": ["network.edges.1.geometry.segments.0.n=5", "network.edges.1.ic=null"]}]}
```

`set` são overrides por caminho (como `mfsim run --set`; `ic=null` retira a condição inicial
gravada quando a malha muda), `steady` resolve o estado estacionário antes de integrar (como
`mfsim steady`), `tf`/`dt_out` alteram o tempo. Os caminhos são relativos à raiz do projeto;
uma entrada cuja saída já existe é saltada (`--force` reexecuta, `--only nome` restringe, `--jobs N`
corre N entradas em paralelo, cada uma igual à sequencial). Cada
saída é uma corrida v2 normal (`case_json` com os overrides aplicados, `info_json` com
`cpu_total`). `batches/thesis.json` é o lote dos casos da tese e das sensibilidades de malha; `batches/issues.json` o dos
notebooks de issues (coluna de líquido de `generators/column.py` em `cases/studies/`, capítulo 10 com o
BDF próprio em numpy e JAX, estacionários do 10.1.1, poço de deriva, gas lift de 16 000 s). Uma entrada
que rebenta é registada como falha e não trava o lote; uma que para a meio fica gravada como parcial.

## Referências de regressão (`tests/golden/`)

Geradas só por `tools/golden_generate.py` (`--rebaseline --note "..."` para regenerar;
cada regeneração fica registada em `CHANGELOG.md` com o desvio máximo). `HASHES.json`
guarda o sha256 dos bytes de cada array e `policy.json` o nível de comparação: **A** =
bit a bit (padrão), **B** = tolerância relativa (`{"tier": "B", "rtol": ...}`; usada para o
rhs/Jacobiana do caminho JAX, 2e−15 do anterior), **C** = assinaturas de ciclo para as
corridas longas (1.º blowdown ±2 s, P_in mín/máx ±0.01 bar, pico ±2 %). Grupos:
`chapter10` (12 trajetórias + variantes), `severe` (referência 150 s numpy e JAX,
rhs/transformação/escalas em 20 estados históricos, Jansen 10 s), `results`, `runs_meta`
(76 corridas gravadas reconstruídas do `case_json`), `notebook_text` (outputs de texto dos
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

`chapter10_case`, `load_case` (`.mf2`), `case_from_json`, `severe_slug_case` e
`mfsim.runs.load_run(...).case` devolvem um `Network`; `build_model`, `simulate`,
`steady_state`, `save_run` e a linha de comando consomem-no diretamente.
Edição conveniente no dialeto legado (o que os notebooks e testes faziam no `Case`):
`net.set_method("fv_upwind")`, `net.set_time(tf=...)`, `net.set_integration(rtol=...)`,
`net.set_options(conservative=True, q_switch=0.05, bb_derivatives="fd", regime=6)` (reconstrói
`ModelSpec`/`SchemeSpec` com as chaves alteradas; `from_legacy(to_legacy())` é a identidade),
`net.pipes[0].ports_in = ports((3, 1, 2, 2))`, `net.vertices[0].spec["qG"] = Constant(0.13)`,
`net.edges_of(Choke)[i].opening = PiecewiseLinear(t, v)`; construção direta com
`Network.build(...)`, `Pipe.legacy(...)`, `BoundaryVertex(id, spec=specs(PT=...))`. O registo
legado (`thesis.record`: `Case`, `Vertex`, `Pipe`, `Valve`, ...; o struct `dados`) é só o
formato intermédio do leitor dos `.mf2` (`thesis.record.network_from_record` e a inversa
`record_from_network`, round-trip testado); nada no simulador o consome.
