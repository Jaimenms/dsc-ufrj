# Intermitência severa: visão geral

*Documentação do mfsim (índice no [README](../README.md)). O detalhe de cada tema — como era
na tese, o que mudou e o que se mediu — está em [issues/INDEX.md](issues/INDEX.md); este
ficheiro é o mapa.*

## O caso

Geometria linha–riser (`cases/severe_slug/reference.json`, gerada por
`mfsim.cases.severe_slug.severe_slug_case`): 20 m horizontais, 300 m descendentes a −3°,
joelho em arco (R = 5 m, 4 elementos), riser de 20 m e 5 m até o separador a 4 bar;
D = 50 mm, vSL = 0.3 m/s, vSG ≈ 0.1 m/s; malha de 15 + 5 células; partida de linha
bloqueada. Modelo a dois fluidos no esquema de volumes finitos escalonados, conservativo,
integrado pelo BDF próprio com o lado direito compilado em JAX.

```bash
uv run mfsim run cases/severe_slug/reference.json --backend jax --out runs/ref.npz
uv run mfsim stats runs/ref.npz          # blowdowns, intervalo, picos, pressão, balanço
uv run mfsim riser runs/ref.npz          # filme no riser (contracorrente)
uv run mfsim scan  <corridas da varredura>   # fronteira: Bøe, padrão da linha, ciclos
```

## Configuração recomendada

| o quê | opção no caso JSON | entrada |
|---|---|---|
| esquema escalonado conservativo | `scheme.type = "fv_upwind"`, `options.conservative = true` | [002](issues/002-fv-upwind.md), [004](issues/004-formulacao-conservativa.md) |
| fase que desaparece sem arrasto espúrio | `relax_target = "noslip"`, `relax_time`, `switch_frac`, `q_switch_*` | [003](issues/003-fase-que-desaparece.md), [006](issues/006-arrasto-espurio-fase-virtual.md) |
| viscosidade artificial acústica | `art_visc = 0.05` | [007](issues/007-viscosidade-artificial.md) |
| joelho em arco e mistura por ângulo | geometria com arco; `regime.vertical = {bubble, 45°, blend 20°}` | [008](issues/008-joelho-em-arco.md) |
| regras físicas de fração | `regime.low_void = {bubble, 0.01}`, `regime.high_void = {annular, 0.9}` | [017](issues/017-fracao-baixa-bolhas-bestion.md), [016](issues/016-fracao-alta-anular.md) |
| massa virtual | `options.virtual_mass = {C_vm 0.5, lambda 1}` | [018](issues/018-massa-virtual.md) |
| arrasto físico de bolhas | `closures.bubble.drag = "physical"` | [015](issues/015-fechamento-de-bolhas.md) |
| líquido com γL = 100 | `fluid.gammaL = 100` | [011](issues/011-compressibilidade-artificial.md) |
| tolerância por faixa e balanço | `integration.tol_range = 0.01`, `accumulate = true`, `reference.state_ref` | [034](issues/034-tolerancia-por-faixa.md), [005](issues/005-acumulador-bdf.md) |
| lado direito compilado | `integration.backend = "jax"`, `integrator = "implicit"` | [036](issues/036-lado-direito-jax.md), [032](issues/032-bdf-implicito-proprio.md) |
| partida | `ic = {"recipe": "blocked", ...}` (campo) ou `"shut_in"` (laboratório) | [027](issues/027-condicoes-iniciais-por-receita.md) |

Com esta configuração o caso de referência dá o ciclo clássico em 3000 s: 4 blowdowns,
intervalo ≈ 800 s, pressão de entrada 4.00–5.85 bar, picos de líquido ≈ 6 kg/s (12× a
alimentação), balanço de massa exato, 16.8 mil passos em ≈ 40 s.

## Mapa dos temas

| tema | entradas |
|---|---|
| por que os esquemas da tese não davam o ciclo; o esquema novo e os seus ingredientes | [001](issues/001-reproducao-capitulo-10.md), [002](issues/002-fv-upwind.md), [003](issues/003-fase-que-desaparece.md) |
| conservação de massa e a sua verificação | [004](issues/004-formulacao-conservativa.md), [005](issues/005-acumulador-bdf.md), [025](issues/025-vertices-conservativos.md) |
| joelho, malha, acústica do tampão, arrasto espúrio | [008](issues/008-joelho-em-arco.md), [009](issues/009-malha-graduada.md), [007](issues/007-viscosidade-artificial.md), [006](issues/006-arrasto-espurio-fase-virtual.md), [010](issues/010-reconstrucao-muscl.md) |
| hiperbolicidade e fechamentos | [017](issues/017-fracao-baixa-bolhas-bestion.md), [018](issues/018-massa-virtual.md), [016](issues/016-fracao-alta-anular.md), [019](issues/019-atrito-interfacial-estratificado.md), [015](issues/015-fechamento-de-bolhas.md) |
| identificação automática de padrão | [020](issues/020-ipe-da-tese-defeitos.md), [021](issues/021-ipe-suave-e-local.md) |
| o que sustenta o ciclo: riser, lei do gás, gotas, volume a montante | [022](issues/022-riser-intermitente-vs-bolhas.md), [023](issues/023-lei-do-gas.md), [024](issues/024-arraste-de-gotas.md), [026](issues/026-volume-de-gas-a-montante.md) |
| fronteira (Bøe, Taitel) | [028](issues/028-criterio-de-boe.md), [029](issues/029-fronteira-da-intermitencia-severa.md) |
| rede de dutos, refluxo, modelo de deriva | [025](issues/025-vertices-conservativos.md), [013](issues/013-filtro-anti-refluxo.md), [014](issues/014-inversao-de-sentido.md), [012](issues/012-vf-escalonado-deriva.md) |
| laboratório | [030](issues/030-validacao-laboratorio.md), [031](issues/031-numerica-a-1-atm.md) |
| custo: tolerâncias, Newton, JAX | [035](issues/035-config-rapida.md), [034](issues/034-tolerancia-por-faixa.md), [036](issues/036-lado-direito-jax.md) |

## Dados e notebook

As trajetórias estão em `notebooks/data/severe_slug_*.npz` (formato v2, caso completo em
`case_json`; o nome codifica a variante: `blocked_conservative_arc_tol_*` = partida
bloqueada, conservativo, arco, `tol_range`; `lab_<caso>_<tese|fisico>`; `scan_P<bar>_v<vSL>`;
`regime6_*`; `network_v*`). O notebook `notebooks/nb_severe_slug.ipynb` (fonte
`notebooks_src/nb_severe_slug.py`) lê-as com `mfsim.runs.load_run` e tem uma secção por
tema, na mesma ordem do mapa acima.
