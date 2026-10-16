# 040 · Partida a partir do estado estacionário

| | |
|---|---|
| tipo | feature |
| área | integração |
| estado | feito (2026-10-16) |
| onde | `initial_state = {"source": "steady", "from", "perturb", "max_iter", "on_failure"}` no JSON; `mfsim.simulation.steady_start` |
| dados | testes em `tests/test_steady_start.py`; uso no mapa de intermitência severa (043) |

## Antes

Uma corrida partia das condições iniciais do caso (`edges`: receita bloqueada, linha fechada
ou arrays gravados) ou da estimativa automática (`guess`). Para começar no estado
estacionário era preciso o `"steady": true` de uma entrada de lote, que corre o Newton a partir
da estimativa automática e recorre à continuação pseudo-transiente. Num mapa de estabilidade
isso não serve: a continuação pseudo-transiente só converge para estacionários estáveis, e a
partir da estimativa automática o Newton não converge nos casos de intermitência severa
(resíduo na casa das centenas; a estimativa usa uma só fração de gás por duto, a do ângulo
médio, 2 % na linha e no riser). A partida bloqueada, por sua vez, obriga a cada corrida a
atravessar um transitório: com o choke muito fechado a linha leva milhares de segundos a
pressurizar (15 células do mapa ainda derivavam aos 8000 s).

## Depois

`initial_state.source = "steady"` faz, antes de integrar, o Newton para rhs(t0, X) = 0 a
partir de `from` (`"edges"`, padrão, ou `"guess"`), com a Jacobiana exata em JAX quando o
backend é JAX e sem continuação pseudo-transiente. O resultado (convergiu, iterações,
resíduo) fica em `info["initial"]` da corrida. `perturb` multiplica as pressões por
(1 + perturb), para a trajetória sair de um equilíbrio instável sem esperar pelo
arredondamento. Sem convergência, `on_failure = "error"` (padrão) falha e `"start"` segue a
partir do estado de partida, registando-o.

A semente robusta é o estacionário de um caso vizinho gravado como condição inicial do caso
(`ic.arrays`): o Newton converge em 3 a 10 iterações de um ponto para o seguinte numa
continuação na abertura do choke, e o caminho de execução continua a ser um só (tudo no JSON).

Para isso a condição inicial de um duto pode ficar nos nós do próprio esquema
(`"ic": {"arrays": ..., "at": "scheme"}`, `Pipe.ic_at`; `state_to_case_ic(X, exact=True)`). Nos
nós da malha, que é como a tese guarda e continua a ser o padrão, a ida e volta estado → caso →
estado interpola e, no esquema escalonado, suaviza o degrau de fração de gás na curva (0,915
passou a 0,639 numa célula, resíduo de 3 em vez de 1e−11); nos nós do esquema é a identidade.
Escrever um estado nas condições iniciais passa também a apagar a receita do duto, que antes
tinha prioridade na escrita do JSON. O registo legado não tem equivalente para a condição
inicial nos nós do esquema e recusa-a.

## Benefício

- Ponto de partida reprodutível no próprio equilíbrio, que é o que um mapa de estabilidade
  linear pede: um estacionário estável fica parado, um instável cresce a partir da perturbação.
  Comparar com a partida bloqueada separa instabilidade de biestabilidade (histerese).
- Sem o transitório da partida bloqueada (os 2500 s descartados de cada corrida do mapa e a
  deriva das células com o choke muito fechado).
- Medido nos testes: o caso 10.1.1 parte do estacionário com resíduo < 1e−9; a partida
  bloqueada do caso de referência, longe de qualquer estacionário, falha ou segue do estado
  de partida, conforme `on_failure`.

## Referências

Issue 027 (condições iniciais por receita) e 043 (mapa). Kelley e Keyes (1998), *Convergence
analysis of pseudo-transient continuation*, sobre por que a continuação pseudo-transiente
procura equilíbrios estáveis.
