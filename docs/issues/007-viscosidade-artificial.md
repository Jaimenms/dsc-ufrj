# 007 · Viscosidade artificial acústica nas equações de momento

| | |
|---|---|
| tipo | feature |
| área | esquema numérico |
| estado | feito (2026-10-13); `art_visc = 0.05` na configuração rápida |
| onde | `scheme.options.art_visc` (ν = C·c·h) |
| dados | `severe_slug_blocked_conservative_arc{_av,_graded_av,_fine_av,_fine_graded_av}.npz`, `..._av05.npz` |

## Antes

Na malha fina graduada (células de 0.8 m na base do riser) a corrida rastejava desde
t ≈ 65 s com passos de 1 ms: *chattering* da vazão de líquido nas faces do joelho, amplitude
0.6 kg/s (maior que a alimentação) e período < 0.1 s. O diagnóstico denso (saída a cada 5 ms)
mostrou um **modo acústico do tampão de líquido** que se forma na base do riser após cada
blowdown (≈ 8 m de líquido entre o bolsão e o gás do riser): vazão e pressão a 16–20 Hz =
c/2L com c = 258 m/s. O esquema escalonado com gradiente de pressão centrado não amortece
acústica, o BDF2 quase não dissipa e o controlador de passo resolve a oscilação.

## Depois

Viscosidade artificial linear ν = C·c·h nas equações de momento (some com h → 0; a taxa
ν·k² é desprezível para ondas longas). BDF1 (Euler implícito) foi testado como alternativa e
é igual ou pior (catastrófico na fase cara: 92k passos).

## Benefício

| malha (conservativa, arco, 3000 s) | passos sem | passos com C = 0.01 | CPU |
|---|---|---|---|
| grossa uniforme 15 + 5 | 112 mil | 85.7 mil | 20 min |
| grossa graduada | 166 mil | 145 mil | 35 min |
| fina uniforme 30 + 10 | 262 mil | 202 mil | 58 min |
| fina graduada | inviável | 303 mil | 83 min |

Mesmo ciclo e balanço exato em todas: 5 blowdowns, intervalo 680–730 s, P_in 4.15–5.86 bar;
os picos de líquido crescem com a resolução no pescoço (6.7 / 8.6 / 9.1 / 10.2 kg/s). Com
C = 0.05 mais o orçamento de Newton (035) a grossa cai para 59.2 mil passos.

## Limitações

A viscosidade de malha só atrasa o início do modo do tampão nas fases de reenchimento (o
comprimento de onda é o do tampão, não o da malha); a causa residual foi resolvida pelo alvo
`noslip` (006).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Modo acústico do tampão de líquido e viscosidade artificial".
