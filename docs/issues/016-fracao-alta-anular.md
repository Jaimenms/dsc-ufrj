# 016 · Regra de fração alta → anular: escoamento contracorrente no riser (item 1)

| | |
|---|---|
| tipo | feature |
| área | fechamentos |
| estado | feito (2026-10-13); na configuração recomendada |
| onde | `model.regime.high_void = {"regime": "annular", "alpha": 0.9}` (rampa C1 de ±0.05) |
| dados | `severe_slug_blocked_conservative_arc_tol_{hv,hv_vm_lv}.npz`; diagnóstico `mfsim riser` |

## Antes

Com bolhas prescritas no riser (padrão fixo da tese acima de 45°), o riser está com mais
de 90 % de gás em 19 % do tempo (depois de cada blowdown) e nessas células o filme de
líquido residual **nunca drena**: desce em 5 % do tempo e sobe a mais de 0.5 m/s em 28 %,
com velocidade média +0.9 m/s contra +0.4 m/s do gás. O fechamento de bolhas a 95–100 % de
gás exerce sobre o líquido um arrasto 3–4 ordens de grandeza acima da gravidade (GammaL de
−300 a −20000 N/m contra 0.87 N/m de peso do filme) e o líquido é expulso pelo topo. Os
fechamentos da tese são co-correntes; não há inundação nem filme descendente.

## Depois

Acima de `alpha` = 0.9 o fechamento passa a anular (regra simétrica à de fração baixa, 017).
O anular dá um balanço de forças sensato no filme (Kowalski na parede, Andritsos &
Hanratty na interface): o filme drena até vG ≈ 6 m/s e passa a ser arrastado perto de
10 m/s, coerente com a inundação de Wallis para D = 5 cm (vSG ≈ 8 m/s). Não há critério de
inundação explícito: emerge do balanço de momento do filme.

## Benefício

| riser quase cheio de gás | passos | CPU | intervalo | pico | P_in (bar) | filme descendo | líquido subindo > 0.5 m/s |
|---|---|---|---|---|---|---|---|
| bolhas (referência) | 25.2k | 47 s | 749 s | 7.1 kg/s | 4.14–5.92 | 5 % | 28 % |
| anular acima de 90 % | 23.5k | 52 s | 821 s | 8.3 kg/s | 4.01–5.98 | 56 % | 6 % |
| + massa virtual 0.5 + bolhas < 1 % | 16.8k | 39 s | 800 s | 6.1 kg/s | 4.00–5.85 | 66 % | 6 % |

O filme a drenar alonga o intervalo entre blowdowns em ≈ 10 %; a configuração com as três
regras físicas é a mais barata de todas e passou a ser a recomendada. Os fechamentos
prescritos "estratificado na linha, bolhas no riser" subestimam o intervalo em ≈ 7 %.

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Escoamento contracorrente no riser (item 1)») em 2026-10-14.*

Diagnóstico na corrida de referência (padrão fixo: bolhas no riser): o riser está
com mais de 90 % de gás em 19 % do tempo (depois de cada blowdown), e nessas
células o filme de líquido residual **nunca drena**: desce em 5 % do tempo e sobe
a mais de 0.5 m/s em 28 %, com velocidade média +0.9 m/s contra +0.4 m/s do gás.
O fechamento de bolhas a 95–100 % de gás exerce sobre o líquido um arrasto de 3 a 4
ordens de grandeza acima da gravidade (GammaL de −300 a −20000 N/m contra 0.87
N/m de peso do filme, tabela no `nb_severe_slug`), e o líquido é expulso pelo topo em
vez de escorrer para a base. O fechamento anular, ao contrário, dá um balanço de
forças sensato no filme (atrito de Kowalski na parede, Andritsos & Hanratty na
interface): o filme drena até vG ≈ 6 m/s e passa a ser arrastado perto de 10 m/s,
coerente com a inundação de Wallis para D = 5 cm (vSG ≈ 8 m/s). Não há critério
de inundação explícito: ele emerge do balanço de momento do filme, que é o que um
modelo a dois fluidos deve fazer; os fechamentos co-correntes da tese só falham
onde o padrão prescrito é o errado.

Regra acrescentada (numpy e JAX, simétrica à de fração baixa):
`PipeParams.high_void_regime` (1 = anular) acima de `high_void_alpha` (0.9), rampa
C1 de ±0.05. Severe slug de referência (`noslip`, JAX, 3000 s):

| riser quase cheio de gás | passos | CPU | intervalo | pico | P_in (bar) | filme descendo | líquido subindo > 0.5 m/s |
|---|---|---|---|---|---|---|---|
| bolhas (referência) | 25.2k | 47 s | 749 s | 7.1 kg/s | 4.14–5.92 | 5 % | 28 % |
| anular acima de 90 % | 23.5k | 52 s | 821 s | 8.3 kg/s | 4.01–5.98 | 56 % | 6 % |
| anular acima de 90 % + massa virtual 0.5 + bolhas < 1 % | 16.8k | 39 s | 800 s | 6.1 kg/s | 4.00–5.85 | 66 % | 6 % |
| IPE local (anular acima de 90 % por regra) | 35.3k | 207 s | 851 s | 4.1 kg/s | 4.01–5.86 | 10 % | 1 % |

O filme a drenar alonga o intervalo entre blowdowns em ~10 % (o líquido que
volta à base do riser antecipa o bloqueio, mas o riser esvazia mais e precisa de
mais líquido para reencher). A configuração com as três regras físicas (anular
acima de 90 %, massa virtual, bolhas abaixo de 1 %) é a mais barata de todas e
passa a ser a recomendada; os fechamentos prescritos "estratificado na linha,
bolhas no riser" subestimam o intervalo em ~7 %. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{hv,hv_vm_lv}.npz`;
diagnóstico `mfsim riser` (`mfsim.analysis.riser_countercurrent`).

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
