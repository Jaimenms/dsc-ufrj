# 061 · Paragem adaptativa com vários sinais, ciclos alternados e captura de N ciclos (massa de dados do artigo 02)

| | |
|---|---|
| tipo | feature |
| área | integração (paragem adaptativa, 041), ferramentas (massa de dados do artigo 02) |
| estado | feito (2026-10-18); chaves novas de `time.settle` (`signals`, `match`, `capture_cycles`, `capture_window`, `amp_tol` por sinal), todas opcionais: sem elas nada muda (goldens intactos) |
| onde | `mfsim/simulation.py` (`window_signature` com mínimo e máximo, `settled(match=…)`, `_integrate_settled`), `mfsim/core/options.py` (`TimeSpec` com a lista `signals`), `mfsim/io/schema/mfsim-case-1.schema.json`; `tools/paper02_dataset.py` (desenho de Sobol em cadeias com arranque quente, corridas retomáveis, marcadores → `dataset.csv`); `tests/test_settle.py`, `tests/test_paper02_dataset.py` |
| dados | caso 10.1.1 da tese (testes); geometria com gas lift do artigo 01 (massa de dados do artigo 02, `runs/paper02/`) |

## Antes

A paragem adaptativa (041) seguia um só sinal, aceitava um ciclo como repetido só quando a amplitude e o período
entre cruzamentos da média coincidiam em duas janelas, e parava no instante em que o sinal assentava: a corrida
acabava sem uma janela limpa do regime estabelecido, e um ciclo com sopros alternados (cujo período entre
cruzamentos muda de janela para janela, como o choke aberto do artigo 01) nunca era dado por assentado. A geração
de uma massa de dados (centenas de corridas sorteadas, a retomar depois de paragens) não tinha ferramenta.

## Depois

**No simulador** (`time.settle`): `signals` acrescenta sinais ao critério (cada um com `amp_tol` próprio, nas suas
unidades; todos têm de assentar, e a razão é «cycle» se algum cicla); `match = "envelope"` aceita o ciclo quando
amplitude, média, mínimo e máximo de duas janelas coincidem a `rel_tol` da amplitude (em vez de amplitude e
período); `capture_cycles = N` continua, depois de assentar, N vezes o maior período dos sinais que ciclam
(`capture_window` s num estacionário) e regista a janela em `info["settle"]["capture"]`; `windows_all` e
`signals` guardam as assinaturas de todos os sinais. O comportamento sem estas chaves é o de antes, bit a bit.

**Ferramenta** `tools/paper02_dataset.py` (plano do artigo 02): `design` sorteia o hipercubo das cinco entradas
(P_res, PI, RGO, P_gas, abertura) por Sobol embaralhado e agrupa os pontos em cadeias de vizinhos (k-means nas
entradas normalizadas; em cada cadeia o primeiro ponto é o mais perto do caso base do artigo 01 e a cadeia arranca a quente
do estado final desse caso (`gl_P20.npz`; `--start random|centre` são as alternativas), e dos outros segue o
caminho mais curto com as entradas reescalonadas a [−1, 1] — exato até 10 pontos, 2-opt acima —, a sequência
pedida pelo dono; rondas sucessivas continuam a sequência de Sobol; `--resequence` reordena uma ronda sem corridas), gravando `design.json`; `run` corre as cadeias em
processos paralelos, o primeiro ponto de cada cadeia a quente do estado final do caso base (ou da partida bloqueada
do gerador, nos conjuntos de teste) e os seguintes com arranque quente a partir do estado final do anterior
(`state_to_case_ic`), cada corrida com a paragem adaptativa (P na
cabeça do poço e q_L no topo do riser, envelope, N_c = 3 ciclos de captura, 600 s num estacionário, tf máximo
24 000 s, orçamento de relógio 045); `markers` extrai de cada corrida feita, na janela de captura (ou nos últimos
4000 s sem ela), média, mínimo, máximo, f_min e f_max de P, α_G, q_L e q_G em oito pontos característicos (fundo
do poço, injeção, cabeça, ponto baixo, topo do riser, linha de gás na plataforma e na cabeça, base do anular),
mais o gás injetado, a classe (`cycle_class`), o período e a razão pico/média, e escreve `dataset.csv` com o
dicionário das colunas em `dataset.json`; `status` conta. Rondas de 25 pontos em pastas datadas
(`runs/paper02/rounds/YYYYMMDD/`, raiz `rounds.json`), abertas pela própria ferramenta ao acabar a anterior
(`run --rounds -1`), com a tabela de cada ronda e a de todas na raiz. As frequências vêm dos intervalos entre máximos do sinal
suavizado (20 s, proeminência de 5 % da faixa): iguais num ciclo periódico, diferentes com sopros alternados,
nulas num sinal permanente. **Retomável**: o estado é o conjunto de ficheiros (`<id>.npz` feito, `<id>.failed.json`
falhado), uma cadeia retoma do último ponto concluído, e um ficheiro `STOP` na pasta pára as cadeias depois da
corrida em curso; matar os processos perde só a corrida em curso.

## Benefício medido

- Testes: `settled` com `match="envelope"` aceita duas janelas de um ciclo com dois harmónicos (período entre
  cruzamentos instável) que o critério do período recusa; no caso 10.1.1 com `capture_window = 30` a corrida
  assenta aos 40 s e pára aos 70 s com `capture = [40, 70]` e as assinaturas dos dois sinais; o desenho de Sobol
  é determinista com a semente, as cadeias cobrem todos os pontos e a ordem é uma permutação; os marcadores dão
  f_min = f_max = 1/500 s numa sinusoide, 1/800 e 1/400 s em máximos alternados a 400 e 800 s, zero num sinal
  permanente.
- Fumo de ponta a ponta (3 pontos numa cadeia, tf 1500 s) e comparação com a partida bloqueada nos mesmos pontos:
  o vizinho próximo em P_gas (117 → 155 bar) custou quente 583 s contra 1534 s bloqueado (que esgotou o orçamento);
  o salto grande (155 → 36 bar) custou quente 32 082 passos contra 6808 bloqueado, porque a linha de gás se esvazia
  para a plataforma. O arranque quente paga com passos pequenos: daí a sequência de distância mínima.

## Referências

- 041 (paragem adaptativa), 040/`state_to_case_ic` (estado gravado como condição inicial), 045 (orçamento de
  relógio), 060 (geometria com gas lift); `papers/02-gas-lift-surrogate-optimization/PLAN.md` (entradas, pontos
  característicos, marcadores, cadeias com arranque quente).
- Sobol, I. M., 1967; Owen, A. B., 1998 (embaralhamento); `scipy.stats.qmc.Sobol`.
