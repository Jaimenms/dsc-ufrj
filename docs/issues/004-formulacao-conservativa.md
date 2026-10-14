# 004 · Formulação conservativa: BDF aplicado às massas por volume

| | |
|---|---|
| tipo | feature |
| área | esquema numérico / integrador |
| estado | feito (2026-10-13); configuração recomendada |
| onde | `scheme.options.conservative = true`, `flux_band`, `flux_eps`; `NetworkSystem.transform`/`transform_jacobian`; `ImplicitBDF(transform=...)` |
| dados | `severe_slug_blocked_{primitive,conservative}.npz`, `..._arc.npz` |

## Antes

Com os estados primitivos (αG, P) nas células, a integração temporal da forma
não-conservativa (dαG/dt, dP/dt) mais o limitador/projeção em αG = 0 e 1 fazem o estoque
de líquido do duto descolar da integral (entrada − saída) **em degraus, um por blowdown**:
≈ 5 kg por slug, 2.1 % do líquido alimentado em 8000 s na malha fina, 5.6 % em 3000 s na
partida bloqueada; 8–12 % do gás. É erro numérico, não propriedade do modelo, e cada balanço
por célula no espaço estava exato.

## Depois

O BDF é aplicado às massas por volume T(x) = (ρG·αG, ρL·αL) — resíduo
T(x_{n+1}) − Σ a_j T(x_{n−j}) − γh f(x_{n+1}) = 0 — resolvido por Newton **nas mesmas variáveis
primitivas**, com dT/dx analítica. Os fluxos de massa nas faces são multiplicados por um
limitador C1 da célula doadora (`flux_band`): fluxo nulo quando a célula a montante não tem a
fase, o que garante positividade sem limitador de fração e mantém o balanço exato por célula
em cada passo. Como T é singular em αG = −γG/γL, os limites de αG são [−0.005, 1.005]
(um *undershoot* pequeno evita que o Newton projete αG em 0 a cada iteração; ver 031 para a
relação com γL). Usar as massas como estados (`fv_upwind_cons`) foi implementado e
descartado: P ∝ mL^γL é uma não-linearidade extrema para o Newton.

## Benefício

| partida bloqueada, malha grossa, 3000 s | erro de massa (líquido / gás) | passos | CPU |
|---|---|---|---|
| primitiva | −43 kg (2.7 %) / +0.28 kg (8.5 %) | 91.8 mil | 18 min |
| conservativa | 0 a menos de arredondamento (< 1e−9) | 132.7 mil | 25 min |

Mesma dinâmica nas duas (primeiro slug ≈ 12 s, 12–13 blowdowns, picos 8.5–8.9 kg/s). Com o
joelho em arco (008) a conservativa passa a ser **mais barata** que a primitiva (112 contra
128 mil passos). Verificado por `tests/test_conservative_residual.py` e, nos vértices
dinâmicos, no caso 10.2.2 (025).

## Limitações

A pressão passa a ser escrava da massa de líquido pela compressibilidade (δP/P = γL·δmL/mL):
o erro de truncamento da massa aparece como ruído de pressão e o controlador de passo
responde com passos menores (custo 1.45× sem o arco). Erro em espaço T, preditor em T e
Jacobiana reavaliada por iteração não resolvem e ficaram como opções (`predict_space`,
`jt_refresh`).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Intermitência severa — Conservação de massa») em 2026-10-14.*

**Conservação de massa e escolha das variáveis de estado.** Com os estados
primitivos (αG, P) nas células, o estoque de líquido no duto descola da integral
(entrada − saída) em degraus, um por blowdown (≈ 5 kg por slug; 2.1 % do líquido
alimentado em 8000 s na malha fina; 5.6 % em 3000 s com a partida bloqueada, que
tem muito mais slugs). É erro numérico da integração temporal da forma
não-conservativa (dαG/dt, dP/dt) somado ao limitador/projeção nos limites αG = 0
e 1, não uma propriedade do modelo. Solução implementada (`{"conservative": True}`
em `pipe.extra`, esquema `fv_upwind`): o BDF é aplicado às massas por volume
T(x) = (ρG·αG, ρL·αL) — resíduo T(x_{n+1}) − Σ a_j T(x_{n−j}) − γh f(x_{n+1}) = 0 —
resolvido por Newton nas mesmas variáveis primitivas, com dT/dx analítica
(`NetworkModel.transform_jacobian`); os fluxos de massa nas faces são
multiplicados por um limitador da célula doadora (rampa C1 em αfase/`flux_band`,
fluxo nulo quando a célula a montante não tem a fase), que garante positividade
sem o limitador de fração de gás e mantém o balanço exato por célula a cada passo
(`tests/test_conservative_residual.py`). Como T é singular em αG = −γG/γL ≈ −0.01,
os limites admitidos para αG são [−0.005, 1.005] (um pequeno *undershoot* evita que
o Newton projete αG em 0 a cada iteração). Custo: a pressão passa a ser escrava da
massa de líquido pela compressibilidade (δP/P = γL·δmL/mL), de modo que o erro de
truncamento da massa aparece como "ruído" de pressão e o controle de passo
responde com passos menores. Usar as massas como *estados* (`fv_upwind_cons`)
foi descartado: P ∝ mL^γL é uma não-linearidade extrema para o Newton.

Resultado da comparação (partida bloqueada, malha grossa 15 + 5, 3000 s, rtol
3e-3; `notebooks/data/severe_slug_blocked_{primitive,conservative}.npz`): mesma
dinâmica (primeiro slug em ~12 s, 12–13 blowdowns, intervalo ≈ 240–250 s, picos de 8.5–8.9 kg/s);
erro de massa medido com a integral dos fluxos de contorno **consistente com o
BDF** (`ImplicitBDF.accum`, mesma recorrência dos estados — a regra dos
trapézios nas saídas de 2 s não é exata): primitivas −43 kg de líquido (2.7 % do
alimentado) e +0.28 kg de gás (8.5 %); conservativa 0 a menos de arredondamento
(< 1e-9 relativo, também em `tests/test_conservative_residual.py`). Custo:
132.7 mil passos contra 91.8 mil (1.45×; 25 min contra 18 min de CPU).

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
