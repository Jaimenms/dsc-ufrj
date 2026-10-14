# 009 · Malha não uniforme (graduada) por trecho

| | |
|---|---|
| tipo | feature |
| área | geometria / malha |
| estado | feito (2026-10-13) |
| onde | `geometry.segments[].grading` (razão h_último/h_primeiro, progressão geométrica); `severe_slug_case(grading_down=, grading_riser=)` |
| dados | `severe_slug_blocked_conservative_arc{,_graded,_fine}.npz`; `tests/test_graded_mesh.py` |

## Antes

Malha uniforme por trecho: refinar a base do riser (onde o bolsão entra e a frente de
líquido se forma) obrigava a refinar o trecho inteiro.

## Depois

Progressão geométrica por trecho (< 1 refina para o fim do trecho, > 1 para o início). O
esquema de volumes finitos já usa as distâncias entre centros de células, logo não muda.

## Benefício

| malha (conservativa, arco, 3000 s) | passos | CPU | perfis junto ao joelho |
|---|---|---|---|
| uniforme 15 + 5 | 112 mil | 22 min | bolsão ruidoso, entrada de gás difusa |
| graduada 15 + 5 (0.25 / 3; células 36 → 9 m e 1.6 → 4.7 m) | 166 mil | 35 min | **coincidem com a fina** |
| uniforme 30 + 10 | 262 mil | 72 min | referência |

As três dão o mesmo ciclo (blowdowns a cada ≈ 700 s, 4.2–5.85 bar): a graduação compra a
resolução da malha fina no pescoço por metade do custo.

## Limitações

A fina graduada (0.8 m na base do riser) só ficou utilizável com a viscosidade artificial
(007) e o alvo `noslip` (006): 68.9k passos / 189 s na configuração recomendada.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Malha não uniforme".
