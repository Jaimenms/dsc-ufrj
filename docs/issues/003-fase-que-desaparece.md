# 003 · Tratamento da fase que desaparece no esquema escalonado

| | |
|---|---|
| tipo | feature |
| área | esquema numérico |
| estado | feito (2026-10-13) |
| onde | opções do esquema `relax_time`, `switch_frac`, `q_switch_L`/`q_switch_G`, `limit_alpha`, `limit_band`; `mfsim/discretization/staggered.py` |
| dados | corridas `severe_slug_blocked_*`; `nb_severe_slug` |

## Antes

Com uma célula cheia de líquido (αG → 0) as equações de momento da fase gasosa ficam sem
massa, a vazão de gás "virtual" fica indeterminada e qualquer limitador de fração de gás
com banda larga (0.02) **destruía massa de líquido** e gerava ciclos falsos de intermitência
severa (slugs sem balanço). A primeira versão do `fv_upwind` sofria disso.

## Depois

Ingredientes que se mostraram necessários, todos no esquema escalonado:

- coeficientes de massa com a fração de gás **real** da célula: a compressibilidade do gás
  desaparece com a fase, e a acumulação de líquido vai para a pressão;
- relaxação da vazão da fase ausente para a condição sem escorregamento com base na célula
  **doadora** e chaveamento abrupto (`relax_time`, fração de corte 0.02): a mistura suave
  "vaza" fração da célula vizinha e sustenta fluxo de uma fase que não existe;
- chaveamento *upwind* com largura adaptativa às vazões locais (`switch_frac` × |q| mais um
  piso por fase, `q_switch_L`/`q_switch_G`);
- limitador de fração só como guarda fina (1e−4) com limites físicos [0, 1] (no modo
  conservativo, 004, uma banda de *undershoot* controlada);
- no integrador, recuo do BDF2 para BDF1 quando estados são projetados nos limites (o
  histórico extrapolado é inconsistente com a projeção) e cooldown de três passos sem
  crescimento após rejeição (evita o ciclo h/2h).

## Benefício

O ciclo de intermitência severa passa a ter balanço de massa verificável (≈ 2 % na forma
primitiva, zero na conservativa, 004/005) e a fase rápida do *blowdown* converge. Com a
condição inicial bloqueada (027) o teste de regressão da referência (150 s) corre em
dezenas de segundos.

## Limitações

A relaxação da fase ausente com alvo na fração real tinha um efeito colateral (arrasto
espúrio da fase virtual, 006), corrigido com `relax_target = "noslip"`.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Ingredientes numéricos";
[docs/findings_for_paper.md](../findings_for_paper.md) ponto 3.
