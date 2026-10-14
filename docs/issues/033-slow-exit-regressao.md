# 033 · `slow_exit` desligado quebrava o gas lift (regressão corrigida)

| | |
|---|---|
| tipo | bugfix |
| área | integrador |
| estado | feito (2026-10-13); padrão `slow_exit = true`; os casos de severe slug de campo desligam-no explicitamente |
| onde | `integration.slow_exit` |
| dados | `notebooks/nb_gas_lift_intermitencia.ipynb` (16 000 s); golden noturno de gas lift (2000 s) |

## Antes

Durante o trabalho do severe slug o recálculo da Jacobiana dentro do Newton
(`slow_exit`) foi desligado por omissão, porque no caso de campo não ajuda. Isso quebrava o
caso de gas lift com intermitência forte em t ≈ 2100–2400 s, no evento da válvula de
retenção (numpy e JAX falhavam).

## Depois

Padrão de volta a ligado; os casos de severe slug de campo gravam `slow_exit = false` no
JSON. À escala de laboratório ele voltou a ser indispensável (031).

## Benefício

Gas lift intermitente (choke a 12 % e 10 %, 16 000 s) volta a correr: ciclos de *heading* de
≈ 4000 s, produção entre 0.2 e 23 kg/s; é um dos casos da suíte noturna.

## Detalhe

*Texto movido de `docs/jax_backend.md` (secção «Lado direito em JAX para toda a tese — Regressão encontrada») em 2026-10-14.*

Regressão encontrada e corrigida nessa verificação: o padrão `slow_exit`
(recálculo da Jacobiana dentro do Newton quando a convergência é lenta) tinha
sido desligado durante o trabalho do severe slug e isso quebrava o caso de gas
lift (evento da válvula de retenção): voltou a ser ligado por padrão; os scripts
do severe slug o desligam explicitamente.

## Referências

[docs/jax_backend.md](../jax_backend.md) (visão geral). O texto integral está na secção «Detalhe» acima.
