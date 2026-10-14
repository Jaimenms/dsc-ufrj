# 032 · Integrador BDF implícito próprio (em vez do `ode15s`)

| | |
|---|---|
| tipo | feature |
| área | integrador |
| estado | feito (2026-10-12/13); integrador de todos os casos de severe slug e de laboratório |
| onde | `integration.integrator = "implicit"` com `max_order`, `max_newton`, `newton_tol`, `first_step`, `min_step`, `slow_exit`, `tight_bounds`, `accumulate`; `mfsim/integrators/implicit.py` |
| dados | todas as corridas gravadas (`info_json`: passos, rejeições, CPU) |

## Antes

A tese integrava com o `ode15s` do MATLAB, uma caixa fechada: sem controlo sobre a
Jacobiana (esparsidade, reutilização, recálculo), sem projeção em limites físicos, sem
tratamento de eventos das tabelas no tempo, sem forma de aplicar o BDF a variáveis
transformadas (004) nem de acumular balanços consistentes (005). O `solve_ivp` do SciPy
(BDF/Radau/LSODA) continua disponível como `integrator = "bdf"` e reproduz o capítulo 10.

## Depois

BDF de ordem 1–2 com passo adaptativo, Newton modificado com Jacobiana reutilizada
(diferenças finitas coloridas pelo padrão de esparsidade da rede, ou Jacobiana externa
exata do backend JAX, 036), pontos de quebra nos instantes das tabelas de especificação,
limites físicos dos estados com projeção e recuo BDF2 → BDF1 quando há projeção (003),
cooldown após rejeição, orçamento de iterações de Newton (035), recálculo da Jacobiana
dentro do Newton quando a convergência é lenta (`slow_exit`, 033), saída parcial em falha
e diagnóstico do passo rastejante (031). Todas as opções ficam na secção `integration` do
caso, com os valores efetivos gravados na corrida.

## Benefício

- Caso 10.1.1 reproduzido em menos de 1 s; os casos do capítulo 10 em segundos.
- Foi o que permitiu a formulação conservativa (004), o acumulador (005), a tolerância por
  faixa (034) e a Jacobiana exata (036): nada disto cabe num integrador de caixa fechada.
- Lições medidas: BDF1 (Euler implícito) é catastrófico na fase rápida do severe slug
  (92k passos na janela de teste, contra 3k com BDF2); Jacobiana fresca a cada passo e passos
  diferentes nas diferenças finitas **não** mudam nada (Newton ≈ 6 iterações por
  não-suavidade dos fechamentos, não por Jacobiana desatualizada).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Ingredientes numéricos" e "Aceleração"; [docs/data_formats.md](../data_formats.md).
