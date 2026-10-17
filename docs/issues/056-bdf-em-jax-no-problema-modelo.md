# 056 · BDF 100 % em JAX no problema-modelo: custo e sensibilidades através da integração

| | |
|---|---|
| tipo | estudo |
| área | integrador (BDF próprio), sensibilidades |
| estado | feito (2026-10-17); protótipo fora do simulador, em `papers/01-severe-slugging-two-fluid/toy/model_problem_jaxbdf.py`; nada muda no `mfsim` |
| onde | `toy/model_problem_jaxbdf.py` (BDF 1–2 de passo variável em `lax.while_loop`, Newton com Armijo, Jacobiana por `jacfwd`, LU denso; reposição em grelha congelada com `lax.scan`); resultados em `runs/paper01/toy/jaxbdf/` (fora do git) |
| dados | problema-modelo do artigo 01 (linha de transmissão com terminação de resistência negativa, N = 50, 100 estados) nos quatro valores de `a` do artigo: 1,2 (estável), 0,95 (Hopf), 0,85 (ciclo maior), 0,5 (relaxação); rtol 1e-4, atol 1e-6, passo máximo 0,5, 300 s |

## Pergunta

O dono perguntou (17/10) se vale a pena escrever o BDF inteiro em JAX — o laço de passos compilado, o Newton com a
busca em linha de Armijo (053) lá dentro — para ganhar (i) tempo e (ii) as sensibilidades dx/dp através de toda a
integração (`docs/future/sensibilidades_jax.md`). Enquanto a questão do estratificado não fecha, o problema-modelo
(secção 2 do artigo) serve de banco de ensaio: é pequeno, tem matriz de massa, o mesmo tipo de resíduo
conservativo M x − c_hist − γ h f(x), e passa de estável a ciclo de relaxação com um só parâmetro.

## Método

O protótipo reproduz o algoritmo de `ImplicitBDF`: preditor extrapolado nas primitivas, histórico nas conservadas,
erro local pela diferença corretor − preditor (½ na ordem 1, r/(1 + 2r) na ordem 2), controlo de passo
0,9 err^(−1/(q+1)) em [0,3, 2] com arrefecimento de três passos depois de uma rejeição, recuo à ordem 1 depois de
falhas seguidas, Newton modificado (Jacobiana no preditor, LU reusada) com a tolerância na correção escalada. Tudo
dentro de um `lax.while_loop` cujo *carry* leva o estado, o histórico, o passo, a ordem, os contadores e o buffer de
saída (interpolação linear entre passos aceites, como o simulador); a Jacobiana vem de `jax.jacfwd` do lado direito e
o sistema linear de `lu_factor`/`lu_solve` densos. A busca em linha de Armijo corre dentro do Newton, com a regra da
055 (sem busca quando a correção já está abaixo de `newton_tol`): sem ela, no caso estável, o resíduo no fundo da
convergência não desce e o protótipo rejeitava 2556 tentativas (10 077 passos em vez de 626).

Três medidas por valor de `a`:

1. passos e tempo de relógio contra `model_problem.run` (o `ImplicitBDF` do simulador, laço em Python, LU esparsa,
   Jacobiana analítica);
2. dG/da por modo direto (`jax.jvp` através de toda a integração adaptativa, o `while_loop` é diferenciável nesse
   sentido) contra diferenças finitas centrais (δa = 1e-3, duas integrações adaptativas), com G a variância do fluxo
   y nos últimos 100 s (o funcional «amplitude do ciclo»);
3. dG/da por modo inverso (`jax.grad`), que o `while_loop` não permite: reposição em grelha congelada — os passos
   aceites (h e ordem) são reintegrados por `lax.scan` com seis iterações de Newton de contagem fixa, e o gradiente é
   o desse operador de reposição (o mesmo artifício do jaxdae, arXiv 2607.23202).

## Resultado

| a | regime | passos: simulador / JAX (rej.) | relógio: simulador / JAX quente | max \|Δy\| (amplitude) | dG/da: jvp / dif. finitas / reposição | tempo jvp / grad |
|---|---|---|---|---|---|---|
| 1,2 | estável | 626 / 626 (1) | 0,20 s / 0,04 s | 5e−10 (0) | ≈ 0 / ≈ 0 / ≈ 0 | 0,19 s / 0,26 s |
| 0,95 | Hopf | 5285 / 5285 (223) | 1,5 s / 0,42 s | 1,3e−5 (0,64) | −1,918 / −1,897 / −1,893 | 1,7 s / 2,1 s |
| 0,85 | ciclo maior | 11 640 / 11 640 (402) | 3,2 s / 0,91 s | 3,7e−5 (1,41) | −1,562 / −1,569 / −1,564 | 3,8 s / 4,6 s |
| 0,5 | relaxação | 43 587 / 43 587 (396) | 10,9 s / 3,46 s | 1,5e−4 (2,73) | +4,824 / +4,825 / **+5,600** | 13,7 s / 21,3 s |

Compilação: 0,3 s na primeira chamada (uma só função de `a`; a reposição recompila por `a` porque a grelha tem
forma estática). Sem a busca em linha os passos e os tempos são os mesmos (0 *backtracks* em todos os casos: o
problema é suave e o Newton converge sem ela).

- **Fidelidade.** O protótipo dá exatamente a sequência de passos do simulador nos quatro casos e trajetórias a
  menos de 1,5e−4 (interpolação e arredondamento); o funcional G coincide a 5 algarismos. O algoritmo do `ImplicitBDF`
  cabe inteiro num `while_loop` sem perder nada do controlo (rejeições, arrefecimento, recuo de ordem, saída
  interpolada).
- **Tempo.** 3–5× mais rápido do que o laço em Python — mas num problema de 100 estados em que o passo do simulador
  custa 0,25 ms (laço, LU esparsa e chamadas em Python dominam) e o compilado 0,08 ms. No caso offshore o passo
  custa 13–35 ms de Jacobiana mais 2–10 iterações de 2 ms: o laço em Python é menos de 5 % e o compilado não
  reduziria o tempo de relógio. O ganho de tempo é do problema pequeno, não do método.
- **Modo direto.** `jvp` através do laço adaptativo (grelha incluída, como a derivada do mapa discreto) concorda com as
  diferenças finitas a 1 % junto da Hopf e a 0,004 % na relaxação, ao custo de 4× uma integração (as diferenças
  finitas custam 2× e trazem o ruído da tolerância). Dezenas de parâmetros a 4× cada ficam caros; é o modo certo
  para poucos parâmetros.
- **Modo inverso.** O gradiente da reposição em grelha congelada concorda a 1 % na Hopf e no ciclo maior, mas
  **desvia 16 % na relaxação** (5,60 contra 4,82): é o gradiente exato do operador de reposição, não do integrador
  adaptativo, e com rtol 1e−4 a dependência da grelha em `a` (os passos seguem as frentes do ciclo de relaxação) não é
  desprezável — a convergência condicional O(tol^{2/3}) que os autores do jaxdae reportam. Custa ~6× uma
  integração e guarda todos os intermédios do `scan` (aqui 43 587 passos × LU de 100², 3,5 GB; no offshore, com
  288 estados e 10⁵ passos, obrigaria a `jax.checkpoint` por passo).

## Conclusão

Escrever o BDF em JAX é possível e fiel: o controlo inteiro do `ImplicitBDF` cabe num `while_loop`. Não é, porém,
um caminho para acelerar o caso offshore (o custo está na Jacobiana e no Newton, não no laço), e as sensibilidades
têm cada uma o seu preço: o modo direto é exato e barato para poucos parâmetros; o modo inverso exige a grelha
congelada e, nos ciclos de relaxação, dá a derivada de outro operador. Para o simulador, o degrau 1 do plano
(`docs/future/sensibilidades_jax.md`: função implícita por passo com a LU do Newton e ∂f/∂p por `jacfwd`, dentro do
laço em Python) dá as mesmas sensibilidades diretas sem reescrever o integrador; o degrau 2 (adjunto) só compensa com
muitos parâmetros e com *checkpointing*. O protótipo fica como referência e banco de ensaio no problema-modelo.

## Referências

- Issues 053 (busca em linha) e 055 (sem busca depois de convergido), 048 (espetro com a matriz de massa).
- `docs/future/sensibilidades_jax.md`; jaxdae, arXiv 2607.23202 (reposição em grelha congelada, convergência
  condicional do gradiente).
