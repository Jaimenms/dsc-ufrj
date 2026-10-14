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

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Volumes finitos escalonados para o modelo de deriva (Beggs & Brill)») em 2026-10-14.*

O modelo de deriva da tese tem 3 equações (massa de cada fase e momento da
mistura) com o hold-up algébrico, da correlação de Beggs & Brill em função de
(P, qL, qG), e no esquema nodal as vazões são estados e o hold-up é diferenciado
no tempo pela matriz MA. No arranjo escalonado isso não fecha a contagem: duas
equações de massa por célula pedem dois estados de célula, e a divisão da vazão
da mistura entre as fases numa face é dada por uma relação algébrica. A
formulação adotada (`StaggeredFVScheme._rhs_drift`, numpy e JAX):

* células: hold-up HG e pressão como estados, com as mesmas equações de massa
  do modelo a dois fluidos (modo conservativo com limitador de fluxo, mapa de
  massas e acumulador de balanço reutilizados); o hold-up inicial vem da
  correlação quando a condição inicial nodal não o traz (`ic_holdup`);
* faces: qL e qG como estados; a soma qM obedece à equação de momento da
  mistura da tese (fluxo ρS·vM² com velocidade doadora, gradiente de pressão,
  tensão de parede bifásica e gravidade, por `_momentum` com ρS e α = 1);
* divisão entre as fases: a relação de escorregamento HG_BB(P, qL, qG) = HG da
  face (hold-up doador das células) é imposta na forma diferenciada no tempo com
  estabilização de Baumgarte, d/dt[HG_BB − HG_f] = −(HG_BB − HG_f)/τ
  (`drift_relax`, 0.05 s), o que dá uma EDO explícita para qG com o coeficiente
  ∂HG/∂qG − ∂HG/∂qL (> 0; piso `drift_den_min`). É a mesma ideia do modelo nodal
  (hold-up algébrico diferenciado no tempo), sem DAE, sem matriz de massa e sem
  inverter a correlação. As derivadas do hold-up são as diferenças finitas da
  tese (`bb_derivatives="fd"`) ou jvp exatas avaliadas a 1e−6 kg/s do lado do
  escoamento (em vazão de gás exatamente nula a correlação está na região
  cortada e a derivada automática é zero, o que derrubava o coeficiente no piso
  e fazia o lado direito explodir).

Verificação no poço 10.3.1 (`tests/test_fv_drift.py`, script `well_fv.py` no histórico do git até ao commit 22ecbc9,
figura `notebooks/figs_well_fv.png`): estacionário com resíduo 1e−11 e todos os
autovalores da Jacobiana negativos (a primeira versão, com o sinal do coeficiente
trocado, tinha autovalores +4400 e parava em 3 ms); transiente de fechamento e
reabertura do choke em 232 passos / 1.4 s (numpy), líquido no topo 5.8–25.8 kg/s
contra 5.9–23.4 no esquema nodal, pressão de fundo 143.8–158.1 contra
146.1–158.4 bar, balanço de massa exato com o acumulador BDF; reprodução
numpy/JAX a 1e−13 no modo "fd".

**Severe slug com Beggs & Brill** (`severe_slug_case(model=2)`): não é
simulável com esta correlação, e o motivo é físico. Partindo da condição
bloqueada (bolsão com 75 % de gás sobre líquido a 0.3 m/s, riser cheio) a
restrição de escorregamento exige vazões de gás de vários m/s para que a
correlação "explique" 75 % de gás com esse líquido, cria uma recirculação de
±6 kg/s nas faces da entrada e o integrador para em t ≈ 2.3 s, com derivadas
exatas ou por diferenças finitas. Um bolsão parado sobre líquido em movimento
(escorregamento infinito) não é um estado da correlação, que foi ajustada em
escoamento co-corrente permanente. Partindo do hold-up da própria correlação
(`ic_holdup="correlation"`) a mesma geometria vai para um estado estacionário em
157 passos (P_in 4.66 bar, produção igual à alimentação, 20–30 % de gás no
riser): a correlação amarra a vazão de gás à do líquido e não há mecanismo de
bloqueio do gás pela coluna de líquido, que é o que gera o ciclo. Para ter
severe slug com um modelo de deriva seria preciso uma lei de escorregamento em
velocidade (vG = C0·vM + vd) que admita gás parado com líquido em movimento, não
uma correlação de hold-up.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
