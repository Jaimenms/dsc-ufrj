# 012 · Volumes finitos escalonados para o modelo de deriva (Beggs & Brill)

| | |
|---|---|
| tipo | feature |
| área | esquema numérico |
| estado | feito (2026-10-13); verificado no poço 10.3.1 |
| onde | `model.type = "drift_flux_bb"` com `scheme.type = "fv_upwind"`; opções `drift_relax`, `drift_den_min`, `ic_holdup`; `model.options.derivatives` |
| dados | `tests/test_fv_drift.py`; `notebooks/figs_well_fv.png` |

## Antes

O modelo de deriva da tese tem 3 equações (massa de cada fase, momento da mistura) com o
hold-up algébrico da correlação de Beggs & Brill, diferenciado no tempo pela matriz de
massa MA no esquema nodal. No arranjo escalonado a contagem não fecha: duas equações de
massa por célula pedem dois estados de célula e a divisão da vazão da mistura numa face é
uma relação algébrica. Não existia versão escalonada deste modelo.

## Depois

Células com hold-up HG e pressão (mesmas equações de massa do modelo a dois fluidos, modo
conservativo, limitador e acumulador reutilizados); faces com qL e qG, cuja soma obedece à
equação de momento da mistura da tese; a divisão entre fases impõe a relação de
escorregamento HG_BB(P, qL, qG) = HG_f na forma diferenciada no tempo com estabilização de
Baumgarte (`drift_relax` = 0.05 s), o que dá uma EDO explícita para qG com o coeficiente
∂HG/∂qG − ∂HG/∂qL (> 0; piso `drift_den_min`). Sem DAE, sem matriz de massa, sem inverter a
correlação. As derivadas do hold-up são as diferenças finitas da tese (`derivatives = "fd"`)
ou jvp exatas avaliadas a 1e−6 kg/s do lado do escoamento (038).

## Benefício

Poço 10.3.1: estacionário com resíduo 1e−11 e todos os autovalores da Jacobiana negativos
(a primeira versão, com o sinal do coeficiente trocado, tinha autovalores +4400 e parava em
3 ms); transiente de fechamento e reabertura do choke em 232 passos / 1.4 s, líquido no topo
5.8–25.8 kg/s contra 5.9–23.4 no esquema nodal, pressão de fundo 143.8–158.1 contra
146.1–158.4 bar, balanço de massa exato; paridade numpy/JAX a 1e−13.

## Limitação física encontrada

Severe slug com Beggs & Brill **não é simulável**, e o motivo é físico: um bolsão de gás
parado sobre líquido em movimento (escorregamento infinito) não é um estado de uma
correlação de hold-up ajustada em escoamento co-corrente permanente. Da condição bloqueada
a restrição exige vazões de gás de vários m/s e o integrador para em 2.3 s; do hold-up da
própria correlação o caso vai a um estacionário sem ciclo em 157 passos. Seria preciso uma
lei de escorregamento em velocidade (vG = C0·vM + vd) que admita gás parado.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Volumes finitos escalonados para o modelo de deriva".
