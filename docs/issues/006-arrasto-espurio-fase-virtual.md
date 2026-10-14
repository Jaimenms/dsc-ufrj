# 006 · Arrasto espúrio da fase virtual nas células cheias de líquido

| | |
|---|---|
| tipo | bugfix |
| área | esquema numérico |
| estado | feito (2026-10-13); `relax_target = "noslip"` na configuração recomendada |
| onde | `scheme.options.relax_target = "noslip"` (só no modo conservativo) |
| dados | `severe_slug_blocked_conservative_arc_tol_noslip.npz`; experimento `archive/scratch_2026-10/py/plug_ring.py` |

## Antes

Uma coluna de 20 m do modelo a dois fluidos **cheia de líquido**, com fechamento
estratificado e um degrau de 0.2 bar entre pressões fixas, travava em 0.49 kg/s, enquanto o
líquido monofásico acelerava a 3.6 kg/s em 3 s. O balanço de momento mostrava GammaL =
1.96 N/m contra 0.05 N/m de atrito de parede: era a força interfacial com a camada de gás
virtual (αG cortado em `alpha_min` = 1e−3) que a relaxação da fase ausente (003) mantinha
**parada**, porque o alvo da relaxação usava a fração real (nula → vazão de gás nula → vG = 0
com vL = 0.27 m/s). O estratificado exercia 40–58× o atrito de parede; anular 8–10×;
bolhas e intermitente nada. Todos os resultados anteriores do severe slug tinham este
arrasto.

## Depois

Alvo da relaxação sem escorregamento com a fração da célula **doadora** com piso em
`alpha_min`, nas duas fases: qG_ns = ρG·max(αG, αmin)·A·vL, qL_ns = ρL·max(αL, αmin)·A·vG.
Só no modo conservativo, onde o limitador de fluxo impede que a vazão virtual drene massa
de uma célula sem a fase. Uma primeira versão com a fração **média da face** fazia o severe
slug rastejar (h ≈ 1e−7 s): numa face entre uma célula cheia de gás e outra com 13 % o filme
de 1 % era arrastado como 44 % do tubo à velocidade do gás.

## Benefício

- Coluna de teste: 3.65 kg/s (= monofásico); o decaimento do modo acústico da coluna
  fechada passa de 1.33 para 0.63 1/s (monofásico 0.32).
- Severe slug de referência: mesmo custo (25.2k passos, 47 s), ciclo mantido (intervalo
  749 s contra 765), pressão máxima 5.92 em vez de 6.00 bar e picos 7.1 em vez de 7.9 kg/s —
  o arrasto espúrio segurava um pouco o líquido.
- Com `noslip` a malha fina graduada no pescoço, antes inviável, passou a custar 77.9k
  passos / 240 s (antes 302.6k / 83 min).
- Resposta a uma pendência: as células cheias de líquido não excitam o modo acústico do
  tampão; o tampão fechado entre dois bolsões oscila no modo "mola de gás" (1.1 Hz medido,
  0.94 previsto).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Reconstrução de α nas faces e o arrasto espúrio".
