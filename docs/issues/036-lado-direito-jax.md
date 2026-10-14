# 036 · Lado direito compilado (JAX) e Jacobiana exata por diferenciação automática

| | |
|---|---|
| tipo | feature |
| área | integrador / desempenho |
| estado | feito (2026-10-13); uma só implementação numpy/JAX desde a reorganização (backend injetado) |
| onde | `integration.backend = "jax"`; `mfsim/backend/`, `mfsim/compiled.py` (`CompiledNetwork`, `get_jax`) |
| dados | `..._arc_fast1_jax.npz`, `..._arc_fine_fast1_jax.npz`; tabela de tempos em [docs/jax_backend.md](../jax_backend.md) |

## Antes

Lado direito em numpy com 0.43 ms por avaliação dominado por overhead (fechamentos,
nível estratificado), Jacobiana por diferenças finitas coloridas a 10.7 ms; a configuração
rápida (035) ainda custava 5 min na malha grossa e 58 min na fina.

## Depois

Fluido, fechamentos, identificação de padrão, modelos, esquemas e rede escritos **uma só
vez** sobre um backend injetado (`NumpyBackend`, `JaxBackend`), com regras de forma
funcional (sem escrita in-place, laços por `while_loop` com o mesmo critério de paragem,
`if` só sobre configuração estática, `where` em vez de ramos em valores). `CompiledNetwork`
reconstrói a rede com o backend JAX e devolve `rhs` com `jit` e Jacobiana exata por `jacfwd`;
o integrador recebe-a como Jacobiana externa. Compilação em cache no disco.

## Benefício

- Avaliação 0.43 → 0.08 ms; Jacobiana 10.7 → 0.9 ms.
- Severe slug: malha grossa 5 min → **45 s** (22.6 mil passos; a referência original levava
  20 min); malha fina 30 + 10: 58 min → **105 s**. Mesmo ciclo, balanço exato.
- Capítulo 10: ganhos onde há muitos passos (10.1.4: 8.0 → 1.6 s; 10.3.2b: 9.8 → 1.7 s;
  estacionário 10.1.1: 9.4 → 0.6 s); gas lift intermitente 16 000 s: 77 → 8.7 s.
- A Jacobiana exata **não reduz** as iterações de Newton (a dificuldade é a não-suavidade
  dos fechamentos); o ganho é o custo por passo (4–6×).
- Paridade numpy × JAX a 1e−14 em toda a matriz de configurações; o caminho numpy ficou
  bit-idêntico ao anterior (goldens de nível A) e com o mesmo custo.

## Limitações

Nos casos com ciclos-limite a diferença pontual entre backends pode chegar a dezenas de
por cento porque a sequência de passos muda e os picos se deslocam no tempo (sensibilidade
caótica); o ciclo é o mesmo. Armadilhas da diferenciação automática em 037.

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Intermitência severa — Lado direito em JAX») em 2026-10-14.*

**Lado direito em JAX** (`mfsim/compiled.py`, `build_jax_rhs(model)` → `(rhs, jac)`).
Na primeira versão foi uma reescrita em JAX do subconjunto usado nos casos de
intermitência severa; desde a reorganização (secção "Lado direito em JAX para
toda a tese") é o mesmo código numpy executado com o backend JAX injetado.
Reproduz `NetworkModel.rhs` a 1e−13 (`tests/test_jax_rhs.py`). `jit` leva a avaliação de 0.43 para 0.08 ms
e a Jacobiana exata (`jax.jacfwd`) custa 0.9 ms contra 10.7 ms das diferenças
finitas coloridas; o integrador a recebe por `ImplicitBDF(jac_fun=...)`. A
Jacobiana exata não reduz as iterações de Newton (a dificuldade é a
não-suavidade dos fechamentos), mas o custo por passo cai 4–6×: a configuração
rápida da malha grossa passa de 5 min para **45 s** (22.6 mil passos; a
referência original levava 20 min) e a malha fina (30 + 10) com a mesma
configuração roda em **105 s** (39 mil passos; 58 min na referência), com o mesmo
ciclo (pressão de entrada 4.14–5.86 bar) e balanço exato. Armadilhas da
diferenciação automática encontradas: raiz quadrada de quantidade que se anula
(0 × ∞ = NaN na derivada; *double where*) e derivadas em quinas (clips em
α = 0, chaveamentos em q = 0), onde diferenças finitas e AD divergem
legitimamente. O que resta em Python (≈ 2 ms por passo: mapa conservativo,
especificações, fatoração esparsa) é o próximo alvo se for preciso mais.

*Texto movido de `docs/jax_backend.md` (secção «Lado direito em JAX para toda a tese») em 2026-10-14.*

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

*Texto movido de `docs/jax_backend.md` (secção «Lado direito em JAX para toda a tese — Os casos curtos») em 2026-10-14.*

Os casos curtos (≤ 200 passos, < 1 s) ficam no empate: o custo é o laço do
integrador em Python, não o lado direito. O ganho aparece onde há muitos passos
e Jacobianas. Cada notebook tem ao final uma seção "Lado direito compilado
(JAX)" que repete a simulação principal e imprime os tempos e a diferença.

*Texto movido de `docs/jax_backend.md` (secção «Lado direito em JAX para toda a tese — Nota da fase 5») em 2026-10-14.*

**Nota da fase 5 (rede sobre o grafo, 2026-10-14).** Com o `NetworkSystem` montado sobre o
grafo, o rhs JAX ficou bit-idêntico ao anterior em todas as chamadas de uma integração
(10.3.2b: 2999 chamadas), mas a Jacobiana (`jacfwd`) dos casos com válvula de retenção
difere 1 ulp em 3 entradas (equação do tanque em relação à pressão do duto a jusante): o
tangente da pressão lida pela válvula deixa de passar por um `set`/leitura de E e o XLA
funde o grafo de outra forma. Consequência: as células JAX dos notebooks de gas lift
seguem outra sequência de passos (10.3.2: 348 → 347 passos; gas lift intermitente 12 %:
2860 → 2914), registada em `tests/golden/CHANGELOG.md`; os goldens noturnos de gas lift
(2000 s) e a referência de 3000 s não mudaram.

## Referências

[docs/jax_backend.md](../jax_backend.md) (visão geral); [docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
