# 053 · Newton do BDF: busca em linha de Armijo, política da Jacobiana e iterado inicial

| | |
|---|---|
| tipo | feature |
| área | integrador (BDF próprio) |
| estado | feito (2026-10-17); opções `integration.newton_damping` (`legacy` \| `armijo`), `integration.jac_policy` (`legacy` \| `step` \| `rate`, com `jac_rate` e `jac_refresh_max`) e `integration.predictor` (`extrapolate` \| `none` \| `euler`), omissões iguais ao comportamento anterior (goldens de nível A intactos); recomendação: `newton_damping = armijo` nos casos novos |
| onde | `mfsim/integrators/implicit.py` (`_newton`, laço de `integrate`), `mfsim/core/options.py` (`IntegrationSpec`), `mfsim/simulation.py`, schema; `tests/test_integrator_options.py` |
| dados | estados da corrida base do artigo 01 (`runs/paper01/offshore_dynamics/ab1.000.npz`) aos 2500 s (bloqueio), 4100 s (sopro) e 4600 s (fim do sopro), gravados nos nós do esquema em `cases/paper01/newton/`; lote `batches/paper01_offshore_newton.json` (150 s de integração por janela e variante) → `runs/paper01/offshore_newton/` (fora do git) |

## Antes

Nas fases rápidas do ciclo offshore o BDF rejeitava ~30 % das tentativas, todas por falha do Newton (052): 694 de
3163 tentativas «lentas», 313 divergentes, 40 «enormes», 47 com lado direito não finito; os Newtons que convergiam
precisavam de 2 a 10 iterações. O Newton modificado (Jacobiana reusada até `jac_reuse` passos ou até um Newton
com mais de 4 iterações) só amortecia o passo quando o resíduo crescia e nunca na primeira iteração — exatamente a
que diverge nas frentes, em que o preditor extrapolado erra por muito.

## Depois

Três opções, todas com a omissão igual ao código anterior:

- **`newton_damping = "armijo"`**: busca em linha em todas as iterações, incluindo a primeira — o passo inteiro é
  aceite se o resíduo escalado desce (critério de Armijo, fator 1 − 10⁻⁴ λ); senão λ é reduzido a metade até quatro
  vezes, e a tentativa falha («diverge») se nenhum λ serve.
- **`jac_policy`**: `"step"` reavalia a Jacobiana no início de cada passo; `"rate"` reavalia-a dentro do Newton, no
  iterado atual, quando a contração err/err_prev passa `jac_rate` (omissão 0,3), até `jac_refresh_max` vezes
  (omissão 3) por Newton.
- **`predictor`**: iterado inicial do Newton — o preditor extrapolado (omissão), o estado anterior (`"none"`) ou
  Euler explícito (`"euler"`); a estimativa de erro local continua a usar o preditor extrapolado.

## Benefício medido

Passos / rejeições / Jacobianas em 150 s de integração a partir de cada estado, no lote (o tempo de relógio variou
com a carga da máquina e não se compara; os passos são o critério):

| janela | referência | Armijo | `rate` (0,3; 3) | preditor `none` | Armijo + `none` |
|---|---|---|---|---|---|
| 2500 s, bloqueio | 423 / 136 / 185 | **249 / 80 / 104** | 299 / 105 / 558 | **156 / 54 / 59** | 295 / 106 / 110 |
| 4100 s, sopro | 3987 / 1416 / 1976 | **3341 / 1147 / 1857** | 2987 / 985 / 6074 | 3907 / 1471 / 1844 | 3323 / 1181 / 1723 |
| 4600 s, fim do sopro | 1072 / 337 / 515 | **965 / 300 / 487** | 815 / 250 / 1571 | 1003 / 326 / 497 | 953 / 305 / 516 |

Traços exploratórios no sopro (fora do lote): Armijo + `step` 3348 / 1114 / 3348; Armijo + `rate` (0,6; 1)
3317 / 1076 / 3255; `step` 4024 / 1400 / 4024; preditor `euler` 12108 / 4331 / 10507.

- A busca em linha de Armijo é ganho limpo e consistente: −41 %, −16 % e −10 % de passos nas três janelas,
  rejeições e Jacobianas a cair na mesma proporção.
- As políticas da Jacobiana trocam passos por Jacobianas: a `rate` corta até 25 % dos passos e triplica as
  Jacobianas (35 ms cada sob carga contra 2 ms do lado direito), e a `step` não muda os passos. Numa máquina livre
  (Jacobiana a 13 ms) a `rate` afinada pode compensar; não foi o caso sob carga.
- O iterado inicial é um efeito de segunda ordem com sinal misto: o estado anterior corta 63 % dos passos no
  bloqueio (o preditor extrapola a oscilação da coluna de líquido contra o bolsão de gás e o Newton diverge dele) e
  quase nada no sopro e no fim do sopro; combinado com Armijo não é melhor do que Armijo sozinho. O Euler explícito
  é catastrófico num sistema rígido (3× os passos, 5× as Jacobianas).
- As trajetórias ficam dentro da tolerância da referência em todas as variantes (pressão de entrada a menos de
  0,13 bar, vazão de saída a menos de 0,5 kg/s no pico do sopro, fração de gás a menos de 0,01 no fim das
  janelas): as opções mudam o custo, não a física.
- Com Armijo, as falhas que sobram no sopro são «lentas» e divergentes a partir de um preditor que erra por muito
  nas frentes: um preditor específico para frentes (limitado pelo conteúdo da célula doadora) fica como pista.
- Recomendação: `newton_damping = armijo` como omissão dos casos novos (o gerador do offshore tem o parâmetro);
  `predictor = none` só se a corrida for dominada pela fase bloqueada.

Os testes conferem que cada opção leva o caso de exemplo ao mesmo estado final que a omissão (2 × 10⁻³), que a
omissão fica intacta e que as chaves fazem o caminho JSON e o schema.

## Referências

- Issue 052 (as quinas dos fechamentos não eram o gargalo) e 051 (as descontinuidades que eram).
- Dennis, J.E., Schnabel, R.B., 1996. Numerical Methods for Unconstrained Optimization and Nonlinear Equations.
  SIAM (busca em linha de Armijo). Hairer, E., Wanner, G., 1996. Solving Ordinary Differential Equations II. Springer
  (Newton simplificado nos BDF).
