# 023 · Lei do gás: expoente politrópico γG (item 4)

| | |
|---|---|
| tipo | estudo |
| área | modelo |
| estado | feito (2026-10-13) |
| onde | `fluid.gammaG` (1.05 nos casos de severe slug) |
| dados | `..._tol_gammaG{10,13}.npz`, `..._scan_P2_v03_gammaG13.npz` |

## Pergunta

Quanto do período depende de o gás ser quase isotérmico (1.05) ou adiabático (1.3)?

## O que se encontrou

| caso | γG | passos | intervalo | pico | ΔP_in |
|---|---|---|---|---|---|
| base | 1.05 | 16.8k | 833 s | 6.1 kg/s | 1.85 bar |
| base | 1.0 | 17.4k | 868 s (+4 %) | 6.3 kg/s | 1.85 bar |
| base | 1.3 | 18.0k | 680 s (−18 %) | 5.4 kg/s | 1.85 bar |
| linha de 1000 m, 2 bar | 1.05 | 59.1k | 3602 s | 8.6 kg/s | 1.62 bar |
| linha de 1000 m, 2 bar | 1.3 | 80.7k | 3295 s (−9 %) | 8.3 kg/s | 1.77 bar |

Gás mais rígido pressuriza o bolsão mais depressa (dP/dt ∝ γ para a mesma massa) e é menos
denso à mesma pressão, logo o ciclo encurta; a amplitude de pressão, fixada pela carga do
riser, não muda. É uma sensibilidade de primeira ordem no período, da ordem da massa
virtual (018) e do filme a drenar (016); o expoente deve ser justificado caso a caso (o
blowdown é rápido, o enchimento lento: um γ efetivo entre os dois é defensável).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Lei do gás (item 4)») em 2026-10-14.*

O gás é politrópico, ρG = ρG0·(P/P0)^(1/γG), com γG = 1.05 (quase isotérmico)
nos casos do severe slug. Sensibilidade a γG = 1.0 (isotérmico) e 1.3
(adiabático), configuração recomendada:

| caso | γG | passos | blowdowns (s) | intervalo | pico | ΔP_in |
|---|---|---|---|---|---|---|
| base | 1.05 | 16.8k | 786, 1618, 2452 | 833 s | 6.1 kg/s | 1.85 bar |
| base | 1.0 | 17.4k | 826, 1704, 2562 | 868 s (+4 %) | 6.3 kg/s | 1.85 bar |
| base | 1.3 | 18.0k | 634, 1310, 1994, 2674 | 680 s (−18 %) | 5.4 kg/s | 1.85 bar |
| linha de 1000 m, 2 bar | 1.05 | 59.1k | 302, 3640, 7506 | 3602 s | 8.6 kg/s | 1.62 bar |
| linha de 1000 m, 2 bar | 1.3 | 80.7k | 302, 3452, 6892 | 3295 s (−9 %) | 8.3 kg/s | 1.77 bar |

Gás mais rígido pressuriza o bolsão mais depressa (dP/dt ∝ γ para a mesma
massa acumulada) e é menos denso à mesma pressão, logo o ciclo encurta; a
amplitude de pressão, fixada pela carga do riser, não muda. É uma sensibilidade
de primeira ordem no período (da ordem da massa virtual e do filme a drenar),
e a escolha do expoente deve ser justificada caso a caso (o blowdown é rápido,
o enchimento lento: um γ efetivo entre os dois é defensável). Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_gammaG{10,13}.npz` e
`..._scan_P2_v03_gammaG13.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
