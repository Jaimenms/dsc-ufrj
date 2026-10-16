# 048 · Espetro do modelo conservativo: problema generalizado com a matriz de massa

| | |
|---|---|
| tipo | bugfix |
| área | análise |
| estado | feito (2026-10-16); `mfsim.analysis.linear_spectrum`; notas de correção em 042 e 044; o notebook nb_estabilidade_mapa e o 043 refazem a secção do espetro com ela |
| onde | `mfsim/analysis.py` (`mass_matrix`, `linear_spectrum`; `jacobian_eigenvalues` e `stability_report` passam a usá-las); `tests/test_linear_spectrum.py` |
| dados | estacionários exatos do mapa com padrão dinâmico a 15 bar (`runs/maps/well_auto/steady/cases`) e corridas `runs/paper01/dynamics` (fora do git); estacionários do mapa 043 (padrão fixo), verificação da sessão que refaz o 043 |

## Antes

No modo conservativo do esquema escalonado (005) o lado direito devolve, nas células, dT/dt com
T(x) = (ρG αG, ρL αL), e o integrador aplica a fórmula BDF a T. A linearização num estado é então
M·δẋ = J·δx, com M = ∂T/∂x (`transform_jacobian`), e não δẋ = J·δx. As análises de estabilidade linear
calculavam os autovalores de J sozinha (`jacobian_eigenvalues`, o espetro de 042, do notebook
nb_estabilidade_mapa e do rascunho do artigo 01). As linhas das massas vêm escaladas por ρL ≈ 900 e
ρG ≈ 5–30 kg/m³ e o espetro é outro: num riser em bolhas, módulos até 15× diferentes dos certos.

Consequências nas conclusões:

- modos reais rápidos e instáveis no ponto baixo (+27 a +41 /s; 042, 044) que nenhum transitório mostrava
  (explicados como «saturam com amplitude desprezável»);
- nenhum par lento a atravessar o eixo imaginário: 042 (item 4) e o notebook concluíram que a passagem a
  ciclos não é uma bifurcação de Hopf do estacionário e que os autovalores não desenham o mapa.

## Depois

`mass_matrix(model, X)` devolve ∂T/∂x nos modelos conservativos e a identidade nos outros;
`linear_spectrum(model, X, jax=..., vectors=...)` resolve o problema generalizado (autovalores de M⁻¹J, com J
exata em JAX ou por diferenças finitas); `jacobian_eigenvalues` e `stability_report` usam-na. Nos modelos
não conservativos (os casos da tese) nada muda.

## Benefício

- Validação: num riser em bolhas o espetro generalizado do modelo conservativo coincide com o do mesmo esquema
  na forma primitiva (o mesmo estacionário e a mesma dinâmica semi-discreta) a 1e−9; o de J sozinha difere até
  15× (teste).
- Padrão dinâmico (exemplo base do artigo 01), 15 bar, par lento do estacionário: −0,0033 ± 0,0067j /s a
  0,13 e −0,00067 ± 0,0103j a 0,22 (estáveis, como os transitórios); +0,00125 ± 0,0121j (518 s) a 0,28, onde o
  transitório oscila com ~495 s e cresce devagar; +0,0036 ± 0,0138j (455 s) a 0,36, onde chega a um
  ciclo-limite de ~450 s. A passagem a ciclos é uma bifurcação de Hopf do estacionário, com o período do ciclo.
- Padrão fixo (mapa 043): o par atravessa entre 0,17 (−0,0025 ± 0,0102j) e 0,22 (+0,00013 ± 0,0133j, 473 s),
  a fronteira do mapa dinâmico; a 0,6 e 1,0 colide no eixo real e dá dois autovalores reais positivos globais
  (+0,040 e +0,015 /s a 0,6; +0,076 e +0,011 /s a 1,0): ciclo de pressão = par complexo lento; golfada severa =
  instabilidade real (relaxação).
- Os modos rápidos que sobram são locais e menores: no padrão fixo, reais de 6 a 9 /s na célula do ponto baixo
  entre 0,10 e 0,22, e nenhum a partir de 0,46; no padrão dinâmico, um ou dois reais (0,6 a 120 /s) na face da
  frente do troço de entrada (0,28 e 0,36) ou no ponto baixo (0,6 e 1,0). Nos transitórios a vazão de líquido
  nessa face oscila 0,5 a 3 % sem efeito nas frações (±3e−4) nem na pressão.

## Referências

Issues 005 (resíduo conservativo), 042 (estudo dos modos locais, revisto aqui), 043 (mapa), 044 (fração das
faces) e o artigo 01. Storkaas e Skogestad (2007), golfada no riser como bifurcação de Hopf com o choke como
parâmetro.
