# 059 · Sensibilidade ao estado inicial propagada no BDF próprio (matriz de monodromia)

| | |
|---|---|
| tipo | feature |
| área | integrador (BDF próprio), análise |
| estado | feito (2026-10-17); opção `integration.sensitivity = "state"` (omissão `null`: nada muda; goldens de nível A intactos) |
| onde | `mfsim/integrators/implicit.py` (`_jac_at`, `_JT_dense`, propagação no laço de `integrate`, `info["S"]`), `mfsim/core/options.py`, schema, `mfsim/simulation.py`; `tests/test_sensitivity.py`; `tools/periodic_orbit.py` (órbitas periódicas por *shooting*, o primeiro uso) |
| dados | problema-modelo do artigo 01 e `cases/examples/hello_world.json` (testes); caso offshore de 2 km do artigo 01 (órbitas, `runs/paper01/offshore_orbits/`) |

## Antes

O simulador não dava derivadas da trajetória: sensibilidades só por diferenças finitas (uma integração por
direção, ruidosas ao nível da tolerância do integrador), e a matriz de monodromia de um ciclo — o que a continuação
de órbitas periódicas do artigo 03 precisa — custaria n integrações por iteração de Newton. O plano
`docs/future/sensibilidades_jax.md` (degrau 1) previa a derivada por passo pelo teorema da função implícita; o
estudo 056 mostrou no problema-modelo que a derivada do mapa discreto (`jvp` através do laço adaptativo) concorda
com diferenças finitas a 1 %.

## Depois

Com `integration.sensitivity = "state"` o integrador carrega S_n = ∂x_n/∂x_0 (n × n, densa) e, em cada passo aceite,
resolve a derivada do próprio passo,

    (J_T − γ J)(x_{n+1}) S_{n+1} = w1 J_T(x_n) S_n + w2 J_T(x_{n−1}) S_{n−1},

com a ordem, os pesos do histórico e o γ do passo tal como o integrador os usou, J a Jacobiana exata (JAX) ou colorida
no estado aceite e J_T a Jacobiana analítica do mapa conservativo. A Jacobiana desta recorrência é própria (não toca
na do Newton nem na sua idade): **a trajetória é a mesma com e sem sensibilidades**, bit a bit. Estados presos num
limite físico (projeção) têm linha nula em S. O controlo de passo não é diferenciado: S é a derivada da trajetória
discreta com a sequência de passos que o integrador escolheu (a «grelha congelada» da 056), que é o que um Newton
sobre φ_T(x₀) tolera. `info["S"]` traz S no instante final; integrando um período, é a matriz de monodromia.

Custo por passo aceite: uma Jacobiana (13–35 ms no offshore), uma LU esparsa e uma substituição com n lados
direitos (n = 200: ~5 ms) — um ciclo de 1500 s do caso de 2 km, ~2000 passos, custa ~2 min com a monodromia inteira.

## Benefício medido

- Problema-modelo (a = 0,95, 20 s, rtol 1e−6): colunas de S contra diferenças finitas centrais a menos de 2 % da maior
  entrada; trajetória idêntica com e sem sensibilidades.
- `hello_world` (JAX, 20 s, tolerâncias do caso): colunas da fração de gás e da vazão a menos de 5 % das diferenças
  finitas; a coluna da pressão não serve de referência por diferenças finitas (a perturbação excita a acústica e a
  resposta fica ao nível do ruído do integrador) — a sensibilidade analítica é aí a única utilizável.
- **Limite encontrado no caso offshore (a = 0,3, órbita de 718 s, 411 passos):** a monodromia propagada não é
  fiável. O teste do multiplicador trivial (M·ẋ = ẋ numa órbita de um sistema autónomo) e a comparação M·v com
  diferenças finitas variam sem padrão com a tolerância do integrador — |μ| dominante 1,47, 11,1 e 2,28 com rtol
  3e−3, 1e−3 e 3e−4; cos(M·ẋ, ẋ) de 0,99 a −0,32; erro contra diferenças finitas de 50 % a 65×; as quinas suaves da
  052 não mudam o quadro. A recorrência está certa (problema-modelo e caso de exemplo); o que é áspero é o mapa
  discreto φ_T: frentes a atravessar células, chaveamentos de padrão e projeções nos limites fazem-no suave só por
  troços à escala da tolerância, e a derivada ε → 0 de um troço não representa o mapa. Consequência: os
  multiplicadores de Floquet por AD não servem no caso offshore; a derivada útil para o Newton das órbitas e para
  a estabilidade é a secante a ε finito (diferenças finitas direcionais, Newton–Krylov), que é a via do artigo 03.
- Primeiro uso: `tools/periodic_orbit.py` — órbitas periódicas do caso offshore por *shooting* simples (Newton
  amortecido em (x₀, T) com a secção de Poincaré em P_in); as sementes das corridas da campanha já são pontos fixos
  do mapa à tolerância do integrador (resíduo 0,002 nas escalas das tolerâncias).

## Referências

- `docs/future/sensibilidades_jax.md` (degraus 1 e 2); issue 056 (derivada do mapa discreto no problema-modelo); 048
  (espetro com a matriz de massa).
- Hairer, E., Wanner, G., 1996. Solving Ordinary Differential Equations II, cap. IV.6 (sensibilidades internas);
  Seydel, R., 2010. Practical Bifurcation and Stability Analysis, 3rd ed., Springer (shooting e monodromia).
