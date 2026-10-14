# 014 · Inversão do sentido do escoamento: fração imposta nas extremidades de entrada

| | |
|---|---|
| tipo | feature |
| área | esquema numérico / rede |
| estado | feito (2026-10-13) |
| onde | célula fantasma do esquema escalonado; `ports.in/out.alphaG = "dirichlet"` lê a fração do vértice |
| dados | `tests/test_reversal.py` (demonstração `reversal_demo.py` no histórico do git até ao commit 22ecbc9) |

## Antes

A onda de fração de gás viaja com o escoamento, logo a fração tem de ser informada em toda
extremidade por onde a massa entra. Nos esquemas nodais da tese a fração fica livre nos
nós de extremidade (gradiente nulo): o que volta tem a composição do que saía, hipótese que
só serve quando não se sabe o que há do outro lado. Era a dificuldade relatada na tese para
refluxo.

## Depois

A fração da célula fantasma é imposta pelo vértice (hold-up do vértice dinâmico ou
`alphaG` especificado num vértice de fronteira) quando a porta é Dirichlet, ou copiada da
célula vizinha quando é livre; o fluxo de massa na face de contorno usa o fantasma como
doador quando ele é imposto, e as faces de contorno com fração imposta usam o estado do
lado doador (célula em saída, vértice em entrada) para os fechamentos, em vez da média
(a média fazia o fantasma alterar a última célula mesmo em escoamento direto: αN 0.68 em vez
de 0.53).

## Benefício

Duto entre dois vértices de pressão, diferença de pressão que troca de sinal em 60–100 s,
separador com αG = 0: com a fração imposta a inversão enche o duto com o líquido do
separador (frente nítida, balanço fechado). Nos esquemas nodais da tese a fração continua
livre, como na tese. Pendente: vértice do tipo separador (fases separadas, devolve a fase
certa em refluxo), anotado como trabalho futuro.

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Inversão do sentido do escoamento») em 2026-10-14.*

A onda de fração de gás viaja com o escoamento, então a fração tem de ser
informada em toda extremidade por onde a massa entra (a dificuldade relatada na
tese). No esquema escalonado isso é feito pela célula fantasma: a fração do
fantasma é imposta pelo vértice (hold-up do vértice dinâmico, ou `alphaG`
especificado num vértice estático, quando a condição de contorno de αG é
Dirichlet) ou copiada da célula vizinha quando é "ausente"; o fluxo de massa na
face de contorno usa o fantasma como doador quando ele é imposto, e as faces de
contorno com fração imposta usam o estado do lado doador (célula em saída,
vértice em entrada) para os fechamentos, em vez da média. Demonstração em
`tests/test_reversal.py` (`reversal_demo.py`, histórico do git até ao commit 22ecbc9): duto entre dois vértices
de pressão, diferença de pressão que troca de sinal em 60–100 s, separador com
αG = 0 — com a fração imposta, a inversão enche o duto com o líquido do separador
(frente nítida, balanço fechado); com a fração livre, o que volta tem a
composição do que saía (gradiente nulo), hipótese que só serve quando não se
sabe o que há do outro lado. Nos esquemas nodais da tese a fração continua livre
nos nós de extremidade, como na tese.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
