# Backend numpy/JAX e lado direito compilado

*Documentação do mfsim (índice no [README](../README.md)).*

## Lado direito em JAX para toda a tese (`mfsim/backend`, `mfsim/compiled.py`)

Desde a reorganização de 2026-10 (fase 2) não existe uma segunda implementação:
fluido, fechamentos, identificação de padrão, modelos de duto, esquemas, relações
das arestas e `NetworkModel` são escritos uma só vez sobre um backend injetado
(`mfsim/backend`: `NumpyBackend`, `JaxBackend`), com regras de forma funcional —
sem escrita in-place (`bk.set`/`bk.add`), laços por `bk.while_loop` com o mesmo
critério de paragem, `if` só sobre configuração estática (formas, opções),
`where` em vez de ramos em valores, NaN tratado por último, `np.` só em constantes
de construção. `CompiledNetwork(model)` (`build_jax_rhs(model)`/`get_jax(model)`,
com cache no modelo) reconstrói o `NetworkModel` com o backend JAX e devolve
`(rhs, jac)` compilados: `jit` do núcleo puro `rhs_core(x, E_t, opt_t)` e
Jacobiana exata por `jacfwd`; as tabelas no tempo (`time_inputs(t)`) são
avaliadas em numpy e passadas como argumentos. Cobre tudo o que o numpy cobre:
modelos de duto (dois fluidos com padrão fixo, misturas por ângulo/fração e
identificação automática, deriva de Beggs & Brill, monofásicos), esquemas nodais
(diferenças centrais e regressivas, Galerkin, Petrov-Galerkin) e escalonados
(dois fluidos primitivo/conservativo, ordem 2 MUSCL, deriva e monofásicos, e
`fv_upwind_cons`, antes fora do caminho JAX), vértices de contorno e dinâmicos,
acoplamentos, chokes, reservatórios, válvulas de retenção e tabelas no tempo.
Paridade numpy × JAX a 1e−9 (na prática 1e−14) em toda a matriz de
configurações (`tests/test_compiled.py`, `test_backend_parity.py`,
`test_jax_network.py`, `test_jax_rhs.py`, `test_jax_regime_auto.py`,
`test_fv_drift.py`, `test_fv_single.py`), com `make_jaxpr` a garantir que o núcleo
traça sem ramos em valores. O caminho numpy ficou **bit-idêntico** ao anterior
(goldens de nível A do capítulo 10, das 76 corridas gravadas e do severe slug) e
com o mesmo custo (referência 150 s: 35 s contra 33 s, mesmos 1871 passos; JAX
3.2 s contra 4.9 s). A identidade de valores entre backends é garantida por
construção: sigmoide com a mesma fórmula (o `custom_jvp` só corrige a tangente
no corte de ±700), `masked_apply` (numpy avalia os fechamentos só no subconjunto
ativo, como antes; JAX avalia tudo e mascara com `where`), nível estratificado e
`solve_pressure` com o mesmo critério de paragem, derivadas internas do Beggs &
Brill por diferenças finitas no numpy e exatas (`jvp`) no JAX por omissão. Os
goldens do caminho JAX foram regenerados nessa transição (rhs/Jacobiana nos
estados históricos a 2e−15 do anterior, nível B; as trajetórias longas seguem
outra sequência de passos pela sensibilidade caótica, com o 1.º blowdown da
referência em 48 s como no numpy — antes 42 s — e o mesmo ciclo; registo em
`tests/golden/CHANGELOG.md`). Uso: `simulate(case, jax=True)`,
`steady_state(case, jax=True)`, `integrate_implicit(model, ..., jax=True)` ou
`ImplicitBDF(rhs, jac_fun=jac)`. A compilação (0.1–5 s por modelo) fica em
cache no disco (`~/.cache/mfsim_jax`, variável `MFSIM_JAX_CACHE`).

Comparação numpy × JAX nos casos dos notebooks (mesmo estado inicial, tempos
sem a compilação; estados coincidem dentro da tolerância, as diferenças
maiores são dos casos sensíveis em que a sequência de passos muda):

| caso | estacionário numpy → JAX | simulação numpy → JAX | passos | diferença máx. relativa |
|---|---|---|---|---|
| 10.1.1 (Galerkin) | 9.4 → 0.6 s | 0.57 → 0.35 s | 169 | 2e−6 |
| 10.1.2 (anular) | — | 1.1 → 0.7 s | 219 | 1e−9 |
| 10.1.3 (bolhas) | — | 3.6 → 0.9 s | 147 | 1e−6 |
| 10.1.4 (Beggs & Brill) | 0.3 → 0.4 s | 8.0 → 1.6 s | 290 | 7e−5 |
| 10.2.1A / 10.2.1B / 10.2.2 (redes) | — | 0.35 → 0.41 / 0.36 → 0.41 / 1.2 → 0.7 s | 150–176 | 1e−11 |
| 10.3.1 (poço) | — | 4.5 → 1.2 s | 568 | 3e−4 |
| 10.3.2a / 10.3.2b (gas lift) | — | 1.2 → 0.7 / 9.8 → 1.7 s | 102 / 526 | 2e−7 / 1e−2 |
| 10.3.3 (poço offshore) | — | 1.5 → 1.2 s | 80 | 6e−11 |
| gas lift intermitente, choke 12 % / 10 % (16 000 s) | — | 77 → 8.7 s / 83 → 10.3 s | 2.9 / 3.6 mil | ciclos iguais (picos deslocados no tempo) |
| severe slug grossa / fina (config. rápida) | — | 5 min → 45 s / 58 min → 105 s | 22.6 / 39 mil | mesmo ciclo |

Nos casos com ciclos-limite (gas lift intermitente, severe slug) a diferença
pontual dos estados pode chegar a dezenas de por cento porque os picos se
deslocam ligeiramente no tempo quando a sequência de passos muda; a figura de
comparação do notebook de gas lift mostra as duas curvas sobrepostas.

**Derivadas internas do Beggs & Brill.** A tese calcula ∂(momento, fluxo de
momento, hold-up)/∂(P, qL, qG) por diferenças finitas com perturbação de 1e−3
relativo; no caminho JAX elas passam a ser exatas, por `jax.jvp` nas três
direções (`PipeParams.bb_derivatives="exact"`, o padrão; `"fd"` reproduz a
tese, e é o que o caminho numpy sempre usa). Onde o fechamento é diferenciável
as duas concordam a ~1e−4 relativo; nos pontos não diferenciáveis (clips do
hold-up em λL, troca de fórmula do atrito bifásico em y = 1.2, `max(C, 0)`) a
diferença finita faz uma média lateral e a automática devolve a derivada
lateral, ambas legítimas. O custo do lado direito dobra nos casos com Beggs &
Brill (os jvp substituem três avaliações extra do fechamento por propagação de
tangentes), e as trajetórias dos casos sensíveis (10.3.1, gas lift) mudam ~10 %
pontualmente, como entre quaisquer duas Jacobianas diferentes.

Os casos curtos (≤ 200 passos, < 1 s) ficam no empate: o custo é o laço do
integrador em Python, não o lado direito. O ganho aparece onde há muitos passos
e Jacobianas. Cada notebook tem ao final uma seção "Lado direito compilado
(JAX)" que repete a simulação principal e imprime os tempos e a diferença.

Regressão encontrada e corrigida nessa verificação: o padrão `slow_exit`
(recálculo da Jacobiana dentro do Newton quando a convergência é lenta) tinha
sido desligado durante o trabalho do severe slug e isso quebrava o caso de gas
lift (evento da válvula de retenção): voltou a ser ligado por padrão; os scripts
do severe slug o desligam explicitamente.

**Nota da fase 5 (rede sobre o grafo, 2026-10-14).** Com o `NetworkSystem` montado sobre o
grafo, o rhs JAX ficou bit-idêntico ao anterior em todas as chamadas de uma integração
(10.3.2b: 2999 chamadas), mas a Jacobiana (`jacfwd`) dos casos com válvula de retenção
difere 1 ulp em 3 entradas (equação do tanque em relação à pressão do duto a jusante): o
tangente da pressão lida pela válvula deixa de passar por um `set`/leitura de E e o XLA
funde o grafo de outra forma. Consequência: as células JAX dos notebooks de gas lift
seguem outra sequência de passos (10.3.2: 348 → 347 passos; gas lift intermitente 12 %:
2860 → 2914), registada em `tests/golden/CHANGELOG.md`; os goldens noturnos de gas lift
(2000 s) e a referência de 3000 s não mudaram.
