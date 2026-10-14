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

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Sensibilidade ao atrito interfacial do estratificado (item 2)".
