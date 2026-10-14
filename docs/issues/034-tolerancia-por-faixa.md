# 034 · Tolerância por faixa de variável (`tol_range` e `state_ref`)

| | |
|---|---|
| tipo | feature |
| área | integrador |
| estado | feito (2026-10-13); `tol_range = 0.01` na configuração recomendada |
| onde | `integration.tol_range`; `reference.state_ref = {"PT", "qL", "qG"}`; `NetworkSystem.state_scales` |
| dados | `severe_slug_blocked_conservative_arc_tol*.npz`; gas lift 16 000 s |

## Antes

Uma tolerância absoluta única (`atol`) para estados de escalas tão diferentes como fração
(0–1), pressão adimensional e vazões (1e−3–10 kg/s); ou, depois de 035, tolerâncias
absolutas por tipo de variável ajustadas à mão caso a caso.

## Depois

Cada caso declara as **faixas esperadas** dos estados e o integrador usa uma única
tolerância como fração dessas faixas: atol_i = tol_range · faixa_i (frações: 1; pressão e
vazões: referências do caso). Sem declaração, a estimativa automática usa as especificações,
a IPR e o estado inicial — que subestima vazões de pico (blowdown) ou superestima (IPR a
pressão nula), por isso os casos devem declará-las (`severe_slug_case` declara qL = 20× e
qG = 50× a alimentação e a pressão máxima da linha).

## Benefício

- Severe slug (JAX) com `tol_range` 0.01: 25.9 mil passos, 50 s, mesmo ciclo e balanço exato —
  equivalente à configuração ajustada à mão (035), sem ajuste.
- Gas lift intermitente 16 000 s com `tol_range` 0.005: 1.5 mil passos, 6 s contra 13.7 s,
  ciclo intacto (0.01 já distorce o mínimo de produção).
- Lição transferida a todos os casos, junto com `slow_exit`, o orçamento de Newton e o
  acumulador; as alavancas específicas do dois fluidos escalonado (viscosidade artificial,
  limitador, arco) não se aplicam ao tubing em Beggs & Brill nem às linhas monofásicas.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Tolerâncias por faixa de variável".
