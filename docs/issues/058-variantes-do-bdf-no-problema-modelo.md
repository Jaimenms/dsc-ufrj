# 058 · Variantes do BDF no problema-modelo: estimativa de erro de ordem consistente; ordens 3–5 presas pela estabilidade

| | |
|---|---|
| tipo | estudo |
| área | integrador (BDF próprio) |
| estado | feito (2026-10-17); protótipo fora do simulador (`papers/01-severe-slugging-two-fluid/toy/model_problem_bdf_variants.py`); **implementado em ficheiro à parte** (decisão do dono): `mfsim/integrators/milne.py` (`ImplicitBDFMilne` com `newton_start = "linear"`, `integrate_bdf_milne`); no offshore −45 % de passos na fase bloqueada, neutro no sopro, ~−13 % por ciclo; por dobrar numa opção `integration.error_estimate = "milne"` do `ImplicitBDF` quando `implicit.py` estiver livre |
| onde | `mfsim/integrators/milne.py`, `tests/test_milne.py`, `tools/milne_windows.py` (janelas do Newton do offshore); `toy/model_problem_bdf_variants.py` (`run` e `plot`); resultados em `runs/paper01/toy/variants/` (`summary.json`, `cost_ratio.json`, `work_precision.png`), `runs/paper01/toy/variants_R5/` e referências Radau em `runs/paper01/toy/reference/` (fora do git) |
| dados | problema-modelo (100 estados) nos casos oscilantes do artigo, a = 0,95 (Hopf), 0,85 (ciclo maior) e 0,5 (relaxação), rtol de 1e−3 a 1e−6 (atol = rtol/100), 300 s; referência Radau (scipy) com rtol 1e−10 |

## Pergunta

O dono pediu (17/10) que se explorassem alterações ao solver que aumentem o desempenho no problema-modelo. O
protótipo em JAX da issue 056 reproduz passo a passo o `ImplicitBDF` e corre em segundos, por isso serve para ensaiar
variantes com diagramas trabalho–precisão: contar passos a tolerância igual engana quando a variante muda a exatidão
que a tolerância entrega — o custo tem de ser lido a **erro global igual**, medido contra uma referência apertada.

## Variantes

- **`order` (estimativa de erro de ordem consistente, dispositivo de Milne):** na ordem 2 o simulador extrapola o
  preditor por dois pontos (erro O(h²)) e estima o erro local por r/(1 + 2r)·(x − pred); como x − pred é O(h²), a
  estimativa não é o erro O(h³) do BDF2 e o controlo comporta-se como o de um método de 1.ª ordem. A variante extrapola
  por q + 1 pontos e usa a constante que sai das posições dos nós (2/11 no passo constante da ordem 2).
- **ordens 3 a 5:** BDF de coeficientes variáveis (derivada do polinómio de Lagrange nos últimos q + 1 instantes),
  com a estimativa `order` e a progressão de ordem do simulador.
- **controlador H211b** (Söderlind) em vez do elementar; **crescimento até 5×** sem arrefecimento; **tolerância do
  Newton 1e−2** em vez de 1e−3.

Métricas: tentativas (passos + rejeições: uma LU e um Newton cada), iterações de Newton, e três erros relativos à
amplitude do ciclo — RMS nos primeiros 30 s (antes de a deriva de fase se acumular), erro do funcional G (variância
do fluxo nos últimos 100 s) e RMS em toda a trajetória.

## Resultado

Custo a exatidão igual à do integrador do simulador («thesis») com rtol 1e−4 (interpolado em log–log nas curvas
trabalho–precisão; «fora» = a variante não chega a essa exatidão no intervalo de tolerâncias):

| variante | a = 0,95 | a = 0,85 | a = 0,5 | (erro nos 30 s iniciais) |
|---|---|---|---|---|
| thesis | 5508 (1,00×) | 12 044 (1,00×) | 43 983 (1,00×) | |
| **order2** | **4093 (0,74×)** | **7740 (0,64×)** | **23 166 (0,53×)** | com o erro de G: 0,81× / 0,78× / 0,62× |
| order3 | 1404 (0,25×) | 159 090 (13×) | 190 316 (4,3×) | |
| order4, order5 | fora | fora | fora | custo independente da tolerância |
| thesis_h211b | 0,97× | 0,92× | 0,93× | |
| order2_h211b | 0,77× | 0,66× | 0,53× | |
| thesis_grow5 | 1,00× | 0,99× | 1,02× | o limite de crescimento nunca prende |
| order5_ntol1e-2 | fora | fora | fora | o Newton converge em ≤ 2 iterações: a tolerância não muda nada |

- **A estimativa consistente é o ganho real: −20 a −50 % de tentativas a erro igual**, maior quanto mais rígido o
  ciclo. A assinatura está no escalamento: com «thesis» o custo cresce como rtol^(−1/2) (5508 → 16 736 → 53 257 por
  década, ×3,2), com «order2» como rtol^(−1/3) (2066 → 4042 → 8392, ×2). A semântica da tolerância muda: «order2» com
  rtol 1e−5 entrega o erro de «thesis» com rtol 1e−4 (1,8 % da amplitude nos 30 s iniciais) com 4042 tentativas em vez
  de 5508.
- **As ordens 3–5 ficam presas pela estabilidade, não pela exatidão.** Com a linha do artigo (R = 0,05) os modos
  acústicos da linha de transmissão estão a 89,8° do eixo real, fora da região A(α) do BDF3 (86°), do BDF4 (73°) e do
  BDF5 (51°): o passo é limitado pela estabilidade e o custo não depende da tolerância (1,7e5, 1,1e5 e 6,6e4
  tentativas para erros de 1e−6). Com uma linha amortecida (R = 5, modos a 69°) as ordens 3 e 4 voltam a comportar-se
  (a = 0,5: 5,5 mil tentativas para 4 % de erro, 9,8 mil para 0,9 %) e o BDF5 continua preso (38 mil, qualquer
  tolerância) — exatamente os ângulos de A(α). É a razão de o BDF2, A-estável, ser a ordem certa para um sistema
  hiperbólico com ondas de pressão pouco amortecidas, como o modelo a dois fluidos.
- O controlador H211b vale ±5 %; o limite de crescimento e a tolerância do Newton não mudam nada neste problema.

## Implementação (ficheiro à parte)

`mfsim/integrators/milne.py`: `ImplicitBDFMilne(ImplicitBDF)` — na ordem 2, com três estados no histórico, o
preditor é o polinómio quadrático por esses estados (nas primitivas ou nas conservadas, conforme `predict_space`) e
o erro local é C·(x − pred) com C de `milne_order2` (posições dos nós; 2/11 no passo constante); na ordem 1, nos dois
primeiros passos e depois de uma projeção (o integrador recua à ordem 1) nada muda. O laço `integrate` é uma cópia do
da classe-mãe com as duas alterações marcadas; `integrate_bdf_milne` replica a construção de
`mfsim.simulation._integrate_bdf`. Ambas as cópias desaparecem quando a opção `integration.error_estimate`
entrar no `ImplicitBDF` (ficheiro em edição noutra sessão, 055). `tests/test_milne.py`: constante 2/11, preditor
exato em quadráticas, caso de exemplo ao mesmo estado (64 → 63 passos: ali o passo é limitado por outra coisa).

## Medição no caso offshore (janelas do Newton, `tools/milne_windows.py`)

Três janelas de 150 s do ciclo com o choke aberto (estados gravados em `cases/paper01/newton/`), todas com a busca em
linha de Armijo; passos / rejeições / Jacobianas (resultados em `runs/paper01/offshore_newton_milne/`):

| janela | `ImplicitBDF` (rtol 3e−3) | Milne, Newton do quadrático | **Milne, Newton do linear** | Milne + tampão 0,8 | tese + tampão 0,8 | `ImplicitBDF` (rtol 1e−3) |
|---|---|---|---|---|---|---|
| bloqueio, 2500 s | 249 / 80 / 104 | 314 / 109 / 118 | **138 / 45 / 47** | 395 / 132 / 161 | 211 / 64 / 91 | 199 / 70 / 86 |
| sopro, 4100 s | 3341 / 1147 / 1857 | 3441 / 1193 / 1984 | 3505 / 1196 / 1909 | 3412 / 1174 / 1822 | 3241 / 1083 / 1779 | 3317 / 1105 / 1884 |
| fim do sopro, 4600 s | 965 / 300 / 487 | 923 / 282 / 477 | 905 / 276 / 487 | 1054 / 328 / 528 | 992 / 297 / 503 | 883 / 270 / 464 |

As trajetórias ficam dentro da tolerância da referência (desvios RMS no fim de 3e−4 a 6e−2 nas escalas das
tolerâncias, i.e. abaixo de 0,1 % das faixas dos estados).

- **Com o Newton a partir do preditor quadrático a estimativa piora o bloqueio (+26 %)**: nas frentes e nas quinas dos
  fechamentos a extrapolação quadrática erra por mais do que a linear e o Newton diverge dela. Com o Newton a partir
  da extrapolação linear (o quadrático só estima o erro) o **bloqueio cai a 138 passos (−45 %) e 47 Jacobianas
  (−55 %)**: é a fase limitada pela exatidão, onde a estimativa de 1.ª ordem obrigava a passos pequenos.
- **No sopro nada muda (+5 %)**: o passo é limitado pela convergência do Newton a partir do preditor, não pelo erro
  local (053/055). No fim do sopro, −6 %.
- Pesando as fases pela sua duração num ciclo (bloqueio ~3000 s, sopro ~300 s, fim e reenchimento ~1000 s), o
  ganho por ciclo é de **~13 % nos passos e ~11 % nas Jacobianas** — modesto, concentrado na fase barata.
- **Tampão do passo depois de falhas do Newton** (`newton_fail_cap = 0,8`, a relaxar 1,25× por passo): −15 % no
  bloqueio e −3 % no sopro com a estimativa da tese, mas **pior com a de Milne** (395 no bloqueio): fica desligado
  por omissão. O sintoma que o motivou mantém-se — com a estimativa da tese, rtol 1e−3 custa *menos* do que 3e−3 no
  bloqueio (199 contra 249), porque o controlador cresce o passo até o Newton falhar, corta, arrefece dois passos e
  volta a falhar.

**Conclusão:** `ImplicitBDFMilne` com `newton_start = "linear"` (omissão da classe) é um ganho limpo onde o passo é
limitado pela exatidão e neutro onde é limitado pelo Newton; no ciclo offshore vale ~10–13 %, no problema-modelo
20–50 %. Entra na campanha quando for opção do `ImplicitBDF` (`integration.error_estimate = "milne"`), para passar
pelo JSON como tudo o resto.

## Referências

- Issues 056 (protótipo em JAX), 053 e 055 (Newton), 047 (passo máximo junto de equilíbrios).
- Hairer, E., Wanner, G., 1996. Solving Ordinary Differential Equations II, cap. V.1 (estabilidade A(α) dos BDF) e
  III.5 (estimativa de Milne). Söderlind, G., 2003. Digital filters in adaptive time-stepping. ACM TOMS 29(1), 1–26.
