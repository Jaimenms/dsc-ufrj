# 019 · Atrito interfacial do estratificado parametrizado (item 2)

| | |
|---|---|
| tipo | feature (estudo de sensibilidade) |
| área | fechamentos |
| estado | feito (2026-10-13) |
| onde | `model.closures.stratified = {"fi_mult": 1.0, "v_gt": 5.5}` (`v_gt = inf` recupera Taitel & Dukler, f_I = f_G) |
| dados | `..._tol_fi{05,20,taitel}.npz` |

## Antes

Fator de atrito interfacial de Andritsos & Hanratty (v_GT = 5.5 m/s) fixo no código, sem
forma de testar a sensibilidade do ciclo a esta escolha nem de usar Taitel & Dukler.

## Depois

Multiplicador e v_GT como opções do fechamento, em numpy e JAX.

## Benefício (resultado do estudo)

| atrito interfacial | passos | intervalo | pico | P_in (bar) |
|---|---|---|---|---|
| tese (Andritsos & Hanratty) | 25.2k | 749 s | 7.1 kg/s | 4.14–5.92 |
| × 0.5 | 30.6k | 759 s | 7.5 kg/s | 4.17–5.91 |
| × 2 | 20.2k | 746 s | 6.6 kg/s | 4.15–5.92 |
| Taitel & Dukler | 27.3k | 735 s | 6.5 kg/s | 4.13–5.92 |

O ciclo é pouco sensível: intervalo dentro de ±2 %, pressão igual, picos dentro de ±8 %.
Taitel & Dukler quase coincide com a tese porque a velocidade superficial do gás na linha
raramente passa de v_GT. O período é fixado pela compressão do bolsão e pela carga
hidrostática do riser, não pelo atrito da interface.

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Sensibilidade ao atrito interfacial do estratificado (item 2)») em 2026-10-14.*

`PipeParams.fi_mult` multiplica o fator de atrito interfacial do estratificado e
`fi_vgt` é o v_GT de Andritsos & Hanratty (`inf` recupera Taitel & Dukler,
f_I = f_G); ambos entram nos fechamentos numpy e JAX (`ClosureInputs.fi_mult`,
`fi_vgt`). Severe slug de referência (padrão fixo, `relax_target="noslip"`,
JAX, 3000 s):

| atrito interfacial | passos | blowdowns (s) | intervalo | pico de líquido | P_in (bar) |
|---|---|---|---|---|---|
| tese (Andritsos & Hanratty, v_GT = 5.5 m/s) | 25.2k | 38, 704, 1484, 2284 | 749 s | 7.1 kg/s | 4.14–5.92 |
| × 0.5 | 30.6k | 54, 766, 1546, 2330 | 759 s | 7.5 kg/s | 4.17–5.91 |
| × 2 | 20.2k | 34, 726, 1506, 2272 | 746 s | 6.6 kg/s | 4.15–5.92 |
| Taitel & Dukler (f_I = f_G) | 27.3k | 38, 698, 1470, 2244 | 735 s | 6.5 kg/s | 4.13–5.92 |

O ciclo é pouco sensível: intervalo entre blowdowns dentro de ±2 %, pressão de
entrada igual, picos de líquido dentro de ±8 %, produção média igual. A forma de
Taitel & Dukler quase coincide com a da tese porque a velocidade superficial do
gás na linha raramente passa de v_GT = 5.5 m/s, onde o fator de Andritsos &
Hanratty se afasta de f_G. O período é fixado pela compressão do bolsão e pela
carga hidrostática do riser, não pelo atrito da interface. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_fi{05,20,taitel}.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
