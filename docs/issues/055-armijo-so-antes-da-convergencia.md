# 055 · Busca em linha de Armijo só enquanto o Newton não convergiu

| | |
|---|---|
| tipo | bugfix |
| área | integrador (BDF próprio) |
| estado | feito (2026-10-17); valor novo `integration.newton_damping = "armijo_tol"`; `armijo` (053) fica igual, para a tabela do 053 continuar reprodutível |
| onde | `mfsim/integrators/implicit.py` (`_newton`), `mfsim/core/options.py`, schema |
| dados | caso offshore de 2 km do artigo 01 com `regime.local_bubble` (054), 8000 s; estados exatos de Newton falhados gravados no estudo de 2026-10-17 (`runs/ab_study/L2000/lb054/`) |

## Antes

Com `newton_damping = "armijo"` (053), a busca em linha corre antes do critério de convergência: em cada iteração,
o passo de Newton só é aceite se o resíduo escalado desce. Quando o iterado inicial já está convergido (correção
abaixo de `newton_tol`), o resíduo pode não descer com o passo completo nem com nenhum λ. É o que se vê nas frentes,
onde os fechamentos têm quinas e o resíduo fica num patamar da ordem da própria correção. Nesse caso a tentativa
falhava como «diverge», o passo era cortado e a história repetia-se até ao passo mínimo.

Nos estados gravados das duas corridas que falhavam com a 054 (limiar de vazio baixo a 0,025, em 2082 s; pressão
interfacial δ = 1,2, em 4256 s), a correção do Newton na 1.ª iteração valia 1e-5 (tolerância 1e-3), o resíduo
1,1e-5, e o passo completo levava-o a 2,2e-5. Os λ de 1/2 a 1/16 também o deixavam acima de 1,1e-5. Aceitando a
correção convergida, o mesmo Newton terminava em 1 iteração.

Há outro defeito, secundário, no mesmo critério: o resíduo está nas variáveis conservadas (massas por volume) e a
escala é a das primitivas. As linhas de massa pesam ~ρG (≈ 20×) e ~ρL·0,01/0,034 (≈ 250×) mais do que as de momento.
Um critério de monotonia natural (Deuflhard), que mede ‖J⁻¹F‖ nas primitivas, resolveu um caso e piorou outro; não
foi adotado.

## Depois

`newton_damping = "armijo_tol"`: se a correção já está abaixo de `newton_tol`, o passo completo é aceite sem
exigir descida do resíduo (o critério de convergência normal do Newton). Acima da tolerância, a busca em linha
é a do 053.

## Benefício

Caso offshore de 2 km com 054, 8000 s:

| variante | `armijo` (053) | `armijo_tol` |
|---|---|---|
| referência | 10 033 passos | 10 400 passos |
| limiar de vazio baixo 0,025 | falha em 2082 s | 9307 passos |
| pressão interfacial δ = 1,2 | falha em 4256 s | 12 778 passos |
| massa virtual também no estratificado | 9953 passos | 10 690 passos |

As quatro variantes chegam ao fim e o custo da referência quase não muda.

## Referências

Issue 053 (busca em linha de Armijo); Deuflhard (2004), *Newton Methods for Nonlinear Problems*, Springer (teste
de monotonia natural); issue 054.
