# 008 · Joelho em arco e transição suave dos fechamentos por ângulo

| | |
|---|---|
| tipo | feature |
| área | geometria / fechamentos |
| estado | feito (2026-10-13); R = 5 m, 4 elementos, Δ = 20° na configuração recomendada |
| onde | `severe_slug_case(elbow_radius=, n_elbow=)`; `model.regime.vertical = {regime, angle_deg, blend_deg}` |
| dados | `severe_slug_blocked_{primitive,conservative}_arc.npz`; `tests/test_vertical_blend.py` |

## Antes

Com o traçado poligonal a inclinação salta de −3° para 90° entre duas células e o
fechamento muda de forma abrupta (estratificado → bolhas, chave dura em |θ| > 45°, como na
tese). O ciclo obtido era **amortecido**: pressão de entrada 5.7–6.4 bar (nunca caía até a do
separador) e blowdowns a cada ≈ 240–250 s com picos de 8.5–8.9 kg/s.

## Depois

Dois ingredientes removem a descontinuidade espacial: um arco tangente à linha e ao riser
com elementos de ângulos intermediários, e o peso do padrão vertical como rampa C1 entre
45° ± Δ (os fechamentos são somados com pesos (1 − w) e w). Misturar sem o arco (só na face
do joelho) **piora** a convergência (1661 contra 877 passos até 7 s): metade de cada fechamento
entre duas células "puras".

## Benefício

O ciclo muda qualitativamente para o blowdown "de livro": a pressão na entrada cai quase
até a do separador (4.2–6.1 bar), o intervalo entre blowdowns quase triplica (≈ 660–700 s) e
os picos caem para ≈ 6.8 kg/s. As formulações primitiva e conservativa concordam entre si,
ou seja, a sensibilidade é do tratamento do joelho e não das variáveis de estado. Custo:
128 mil passos (primitiva, 24 min) e 112 mil (conservativa, 22 min).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Intermitência severa — Joelho curvo») em 2026-10-14.*

**Joelho curvo e transição suave dos fechamentos por ângulo.** Com o traçado
poligonal a inclinação salta de −3° para 90° entre duas células e o fechamento
muda de forma abrupta (estratificado → bolhas, chave dura em |θ| > 45°). Dois
ingredientes removem a descontinuidade espacial: `elbow_radius`/`n_elbow` em
`severe_slug_case` (arco tangente à linha e ao riser, elementos com ângulos
intermediários) e `vertical_blend` (graus) em `pipe.extra` (peso do padrão
vertical como rampa C1 entre 45° ± Δ; os fechamentos são somados com pesos
(1 − w) e w, `TwoFluidModel.vertical_weight`). A mistura sem o arco só atua na
face do joelho e piora a convergência (metade de cada fechamento entre duas
células "puras"); com o arco ela ajuda. Resultado (R = 5 m, 4 elementos, Δ = 20°,
partida bloqueada, 3000 s, `notebooks/data/severe_slug_blocked_*_arc.npz`): o
ciclo muda qualitativamente para o *blowdown* "de livro" — a pressão na entrada
cai quase até a do separador (4.2–6.1 bar contra 5.7–6.4 bar com o joelho
anguloso), o intervalo entre *blowdowns* quase triplica (≈ 660–700 s contra ≈ 240–250 s;
ciclos contados pelos mínimos da pressão na entrada) e os picos caem para
~6.8 kg/s (contra 8.5–8.9). As duas formulações concordam entre si (mesma faixa
de pressão e de período), ou seja, a sensibilidade é do tratamento do joelho, não
das variáveis de estado. Custo: 128 mil passos (primitivas, 24 min) e 112 mil
(conservativa, 22 min) — com o joelho suave a formulação conservativa fica mais
barata que a primitiva.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
