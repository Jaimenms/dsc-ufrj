# 010 · Reconstrução MUSCL da fração nas faces e velocidade doadora de 2.ª ordem

| | |
|---|---|
| tipo | feature (sem ganho medido) |
| área | esquema numérico |
| estado | feito (2026-10-13); disponível, fora da configuração recomendada |
| onde | `scheme.options.alpha_face = "muscl"`, `fv_order = 2` (numpy e JAX) |
| dados | `severe_slug_blocked_conservative_arc_tol_muscl.npz` |

## Antes

Fração de área e velocidade doadora de 1.ª ordem nas faces; hipótese de que a frente de
gás no riser era difundida pela interpolação de α.

## Depois

Declive minmod a partir da célula doadora para α (`muscl`) e para a velocidade doadora
(`fv_order = 2`).

## Benefício (negativo, mas informativo)

Não dá nitidez à frente: na malha grossa o ciclo é igual (intervalo 775 s contra 765 s;
24.8k passos) e a largura mediana da transição 0.1–0.9 de αG no riser fica em 2.9 m com e
sem reconstrução (1.7 m na malha fina). No esquema escalonado os fluxos de massa **são** as
vazões das faces (estados); a fração nas faces só entra no momento. A difusão da frente vem
da dinâmica do momento (velocidade doadora, relaxação da fase ausente, limitador), não da
interpolação de α. Fica documentado para não ser repetido.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Reconstrução de α nas faces".
