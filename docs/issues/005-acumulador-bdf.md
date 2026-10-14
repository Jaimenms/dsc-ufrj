# 005 · Acumulador de balanço de massa consistente com o BDF

| | |
|---|---|
| tipo | feature |
| área | integrador |
| estado | feito (2026-10-13) |
| onde | `integration.accumulate = true`; `ImplicitBDF.accum`, `NetworkSystem.boundary_net_fluxes`; array `acc` das corridas gravadas |
| dados | todas as corridas `severe_slug_*` posteriores (`acc` 2 × nt) |

## Antes

O balanço de massa era verificado integrando os fluxos de contorno pela regra dos
trapézios nas saídas de 2 s. Isso **não fecha** nem no modo conservativo (−1.7 kg na
referência): é erro de quadratura nos picos do blowdown, não erro do esquema, e tornava
impossível distinguir perda de massa real de erro de verificação.

## Depois

O integrador integra os fluxos de contorno (e as vazões líquidas dos vértices dinâmicos)
com **a mesma recorrência BDF dos estados**, I_{n+1} = w1·I_n + w2·I_{n−1} + γh·g, e grava o
resultado em `info["accum"]` / `acc`. `Run.balance()` e `mfsim stats` usam-no.

## Benefício

O erro de massa passa a ser uma medida exata do esquema: zero a menos de arredondamento no
modo conservativo (004), e os −43 kg da forma primitiva são atribuíveis ao esquema, não à
verificação. Todos os resultados posteriores reportam o balanço com esta medida.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Conservação de massa"; `tests/test_conservative_residual.py`.
