# 043 · Mapa de intermitência severa: abertura do choke × pressão do reservatório

| | |
|---|---|
| tipo | feature |
| área | análise |
| estado | feito (2026-10-16) |
| onde | `tools/slug_map.py` (build, plot); `cases/severe_slug/well_choke.json`; `mfsim.analysis.cycle_class`, `run_cycle_class`; `mfsim batch --jobs`; `generators/severe_slug.py` (`choke_opening`, `choke_cv`, `choke_law`, `reservoir`) |
| dados | `runs/maps/well/` (fora do git, regenerável): casos de cada ponto, lotes, corridas, `summary.json`, `slug_map.png` |

## Antes

A fronteira da intermitência severa só tinha a varredura de 029 (vazões e pressão do
separador, dez pontos, partida bloqueada, `tf` fixo) e o critério de Bøe, que é cego ao choke
e, com razão gás–líquido fixa, à IPR. Não havia choke no caso de referência, nem poço, nem
execução em paralelo, nem classificação das corridas. O primeiro protótipo (fora do repo,
lei de válvula da tese, 8000 s fixos) teve 7 falhas do integrador e 15 células que ainda
derivavam no fim.

## Depois

- **Caso base**: `well_choke.json`, remendo sobre o caso de referência com IPR linear na
  entrada (0,056 kg/s/bar, a razão gás–líquido da referência) e choke homogéneo no topo
  (CV 15) a descarregar no separador a 4 bar. O gerador aceita os mesmos elementos como
  parâmetros.
- **Grelha**: 14 aberturas (0,03 a 1) × 9 pressões do reservatório (8 a 48 bar).
- **Duas partidas por ponto**: do estacionário (issue 040; continuação de Newton pela grelha a
  partir de uma corrida que assenta, gravada como condição inicial exata nos nós do esquema,
  perturbação de 0,1 % nas pressões) e bloqueada (linha cheia, riser bloqueado). A diferença
  entre as duas é a histerese.
- **Paragem adaptativa** (041) com janelas de 2000 s e 8000 s de máximo.
- **Classe** (`cycle_class`): estável se a amplitude da pressão de entrada é menor do que
  0,1 bar ou 1 % da média, ou se não há ciclo até aos 8000 s; golfada severa se o pico da vazão
  de saída é pelo menos 2× a média; ciclo de pressão com saída suave no resto. Cada célula da
  figura mostra a pressão de entrada (estável) ou a pressão máxima e o período (com ciclo).
- **Execução**: os 252 casos são JSON corridos por `mfsim batch --jobs 13`.

## Benefício

*Refeito em 2026-10-16 com passo máximo de 1 s em todas as corridas (issue 047), períodos com média móvel de 20 s
e a classe linear do estacionário (issue 048). A primeira versão, sem passo máximo, dava 8 células de «histerese»
que eram, em 7 delas, partidas do estacionário presas ao equilíbrio pelo amortecimento do BDF; essas corridas
ficaram em `runs/maps/well_nomaxstep`.*

| | partida do estacionário | partida bloqueada | classe linear do estacionário |
|---|---|---|---|
| estáveis | 81 | 77 | 75 |
| ciclo de pressão (Hopf no linear) | 20 | 24 | 27 |
| golfada severa (modo real no linear) | 25 | 25 | 24 |
| concordância com a classe linear | 117 / 126 | 117 / 126 | — |
| falhas do integrador | 0 | 0 | — |

- **O espetro prevê o mapa.** A classe linear do estacionário (modo global lento dominante, com a matriz de massa)
  bate com a dinâmica em 117 de 126 pontos: o ciclo de pressão nasce de uma bifurcação de Hopf, e onde o modo
  dominante passa a real começa a golfada severa. A 15 bar o par atravessa o eixo entre 0,17 e 0,22 (período
  linear 473 s; ciclo simulado na fronteira 489 s).
- **Fronteira com a forma de Jansen, limiar mais baixo.** No cruzamento linear a queda no choke (lei monofásica do
  líquido) é 0,37–0,48 bar de 10 a 30 bar, contra 0,61–0,72 bar do critério de Jansen, Shoham e Taitel (1996); as
  aberturas críticas são 1,2 a 1,5 vezes as de Jansen. Na resolução da grelha a fronteira dinâmica cabe entre as
  duas, e por isso parecia seguir Jansen.
- **Histerese.** 4 células têm classes diferentes entre as partidas. Duas são histerese real, estacionário
  linearmente estável e partida bloqueada a ciclar (8 bar / 0,06 e 30 bar / 0,6: bifurcação subcrítica); as outras
  duas (15 bar / 0,22 e 24 bar / 0,46) são pares de Hopf no limiar (crescimento ~10⁻⁴ /s) que a partida do
  estacionário perturbada em 0,1 % não vê em 2000 s. A paragem adaptativa não é um teste de estabilidade.
- **Produção**: a 15 bar a abertura estável mais produtiva é 0,22 (0,49 kg/s); segurar o estacionário instável à
  abertura total daria +7 %.
- **Ciclos longos**: a classe decide-se no registo depois do transitório (todas as janelas menos a primeira,
  `run_cycle_class`); com só a última janela, golfadas de uma hora a 8 bar passavam por estáveis.
- A lei de válvula homogénea (039) é o que faz o choke estabilizar; com a lei da tese o protótipo tinha 11 células
  estáveis em 126.

## Referências

Correção feita (2026-10-16): o mapa foi refeito com `integration.max_step` = 1 s em todas as corridas (issue [047](047-passo-maximo-perto-do-equilibrio.md)); ver «Benefício».

Notebook: [nb_estabilidade_mapa](../../notebooks/issues/nb_estabilidade_mapa.ipynb) (fonte em `notebooks_src/issues/`).

Issues 029 (fronteira por varredura), 039–042. Jansen, Shoham e Taitel (1996), *The
elimination of severe slugging — experiments and modeling*, IJMF 22 (eqs. 5 e 10). Plano em
`docs/future/slug_signatures_routing.md` (degrau 1).
