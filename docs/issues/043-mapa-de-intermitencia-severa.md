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

| | partida do estacionário | partida bloqueada |
|---|---|---|
| estáveis | 87 | 79 |
| ciclo de pressão, saída suave | 14 | 22 |
| golfada severa | 25 | 25 |
| falhas do integrador | 0 | 0 |
| paragem: permanente / ciclo repetido / tempo máximo | 84 / 18 / 24 | 76 / 23 / 27 |
| CPU total, passos (mediana) | 5,1 h, 9 349 | 6,0 h, 12 744 |

- **Concorda com a teoria**: a fronteira estável do lado aberto segue, linha a linha, o
  critério de choke de Jansen, Shoham e Taitel (1996), queda no choke de cerca de 0,67 bar.
- **Histerese**: 8 células estáveis a partir do estacionário ciclam a partir da partida
  bloqueada, sete delas logo acima da fronteira entre 15 e 30 bar. Um poço que escoa estável
  aguenta o choke mais aberto do que um poço a arrancar depois de uma paragem.
- **Produção**: a abertura estável mais produtiva sobe com a pressão do reservatório (0,10 a
  10 bar; 0,22 a 15 bar a partir do estacionário, 0,17 bloqueada; aberto acima de 30 bar).
- **Ciclos longos**: a linha de 8 bar tem ciclos de cerca de uma hora, mais longos do que as
  janelas de 2000 s. A classe decide-se por isso no registo depois do transitório (todas as janelas
  menos a primeira, `run_cycle_class`), e não só na última janela; com a última janela apenas,
  5 golfadas dessa linha passavam por estáveis.
- A lei de válvula homogénea (039) é o que faz o choke estabilizar; com a lei da tese o
  protótipo tinha 11 células estáveis em 126.

## Referências

Notebook: [nb_estabilidade_mapa](../../notebooks/issues/nb_estabilidade_mapa.ipynb) (fonte em `notebooks_src/issues/`).

Issues 029 (fronteira por varredura), 039–042. Jansen, Shoham e Taitel (1996), *The
elimination of severe slugging — experiments and modeling*, IJMF 22 (eqs. 5 e 10). Plano em
`docs/future/slug_signatures_routing.md` (degrau 1).
