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

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Arrasto espúrio da fase virtual») em 2026-10-14.*

**Arrasto espúrio nas células cheias de líquido** (experimento
`archive/scratch_2026-10/py/plug_ring.py`, seção correspondente no `nb_severe_slug`): uma coluna de
20 m do modelo a dois fluidos cheia de líquido, com fechamento estratificado e um
degrau de 0.2 bar entre pressões fixas, trava em 0.49 kg/s, enquanto o líquido
monofásico acelera a 3.6 kg/s em 3 s. O balanço de momento mostra GammaL =
1.96 N/m contra 0.05 N/m de atrito de parede: é a força interfacial com a camada
de gás virtual (αG cortado em `alpha_min` = 1e−3) que a relaxação da fase ausente
mantém parada, porque o alvo da relaxação usa a fração real (nula → vazão de gás
nula → vG = 0 enquanto vL = 0.27 m/s). O estratificado exerce então 40× o atrito
de parede a 0.27 m/s e 58× a 1 m/s (anular 8–10×; bolhas e intermitente nada);
com vG = vL todos os fechamentos reproduzem o monofásico.
`relax_target="noslip"` (alvo sem escorregamento com a fração da célula doadora
com piso em `alpha_min`, nas duas fases; só no modo conservativo, onde o
limitador de fluxo impede que a vazão virtual drene massa de uma célula sem a
fase) elimina o arrasto: a coluna volta a 3.65 kg/s e o decaimento do modo
acústico da coluna fechada (c/2L = 5.6 Hz) passa de 1.33 para 0.63 1/s
(monofásico: 0.32). Uma primeira versão usou a fração **média da face** em vez
da doadora e fazia o severe slug rastejar (h ≈ 1e−7 s): numa face entre uma
célula cheia de gás e outra com 13 % de gás o filme de 1 % era arrastado como se
fosse 44 % do tubo, à velocidade do gás do blowdown. Com o alvo correto, o
severe slug de referência custa o mesmo (25.2k passos, 47 s) e mantém o ciclo
(blowdowns em 38, 704, 1484, 2284 s, intervalo 749 s contra 765 s, produção média
igual), com pressão máxima de entrada 5.92 bar em vez de 6.00 e picos de líquido
7.1 em vez de 7.9 kg/s: o arrasto espúrio nas células cheias de líquido segurava
um pouco o líquido na linha e no riser. É a variante fisicamente consistente; os
resultados anteriores foram obtidos com o alvo de fração real e devem ser lidos
com essa ressalva. Resposta à pergunta da pendência: as células cheias de
líquido do modelo a dois fluidos não excitam o modo do tampão; são mais
amortecidas do que o monofásico, por um arrasto espúrio que liga e desliga onde a
fração cruza a banda da relaxação. O tampão entre dois bolsões de gás, fechado,
oscila no modo "mola de gás" (1.1 Hz medido, 0.94 Hz previsto por
k = 2A²γP/V_bolsão e m = ρL A L_tampão), não no acústico.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
