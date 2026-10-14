# 002 · Volumes finitos escalonados com upwind pela velocidade de fase (`fv_upwind`)

| | |
|---|---|
| tipo | feature |
| área | esquema numérico |
| estado | feito (2026-10-13); esquema padrão do severe slug |
| onde | `scheme.type = "fv_upwind"`; `mfsim/discretization/staggered.py` |
| dados | `notebooks/data/severe_slug_run.npz`, `severe_slug_run_long.npz`; `nb_severe_slug` |

## Antes

A tese discretiza os quatro estados nodais `[αG, P, qL, qG]` no mesmo nó (Galerkin,
Petrov-Galerkin, diferenças centrais ou regressivas). Com fases que aparecem e desaparecem
(riser que enche de líquido e depois se esvazia) os esquemas colocados oscilam nó a nó, o
"upwind" original não é *upwind* (001) e a tese relata que a intermitência severa não foi
simulada. A fração de gás nas extremidades por onde entra massa não era informada (a
dificuldade apontada na tese).

## Depois

Arranjo escalonado: fração de gás e pressão nas **células**, vazões de cada fase nas
**faces**. Os fluxos de massa usam a célula doadora segundo o sinal da vazão **da própria
fase** (não da mistura), o gradiente de pressão é centrado e as células fantasma das
extremidades recebem a fração do vértice quando ela é imposta (014). Há classes concretas
por modelo (dois fluidos, deriva 012, monofásico) e a variante conservativa (004). O
esquema é o mesmo em numpy e em JAX (036).

## Benefício

- Primeiro ciclo-limite de intermitência severa obtido com o modelo a dois fluidos da tese:
  geometria linha–riser (300 m a −3°, riser 20 m, 4 bar, D = 50 mm), 18 ciclos em 8000 s com
  período ≈ 312 s na malha fina e 24 ciclos com ≈ 230 s na grossa; *blowdown* com fração de
  gás no riser de 0 a 0.9 em ≈ 20 s e picos de líquido 10–20× a alimentação.
- Estado estacionário discreto único e independente da malha (001, ponto 3).
- Os casos do capítulo 10 continuam disponíveis nos esquemas nodais da tese (bit-idênticos);
  o esquema novo é uma opção, escolhida no caso.

## Limitações

Precisou de um conjunto de ingredientes para a fase que desaparece (003) e da formulação
conservativa para fechar o balanço de massa (004); a primeira versão com limitadores
"generosos" de fração produzia ciclos falsos por destruição de massa.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Intermitência severa"; `tests/test_fv_single.py`,
`tests/test_conservative_residual.py`.
