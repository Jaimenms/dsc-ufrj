# Backend numpy/JAX: visão geral

*Documentação do mfsim (índice no [README](../README.md)). O detalhe está em
[issues/036](issues/036-lado-direito-jax.md) (desenho, paridade, tempos),
[037](issues/037-tangentes-nan.md) (armadilhas da diferenciação automática),
[038](issues/038-derivadas-beggs-brill.md) (derivadas do Beggs & Brill) e
[033](issues/033-slow-exit-regressao.md).*

Não existe uma segunda implementação: fluido, fechamentos, identificação de padrão,
modelos de duto, esquemas, relações das arestas e a montagem da rede são escritos uma só
vez sobre um backend injetado (`mfsim/backend`: `NUMPY`, `JAX`). `CompiledNetwork`
(`mfsim/compiled.py`, `get_jax(model)`) reconstrói a rede com o backend JAX e devolve o lado
direito com `jit` e a Jacobiana exata por `jacfwd`, que o BDF próprio usa como Jacobiana
externa. As tabelas no tempo são avaliadas em numpy e passadas como argumentos.

```bash
uv run mfsim run cases/severe_slug/reference.json --backend jax     # ou "backend": "jax" no caso
```

Em Python: `simulate(case, jax=True)`, `steady_state(case, jax=True)` ou
`ImplicitBDF(rhs, jac_fun=jac)` com `rhs, jac = get_jax(model)`. A compilação (0.1–5 s por
modelo) fica em cache no disco (`~/.cache/mfsim_jax`, variável `MFSIM_JAX_CACHE`).

Garantias: paridade numpy × JAX a 1e−9 (na prática 1e−14) em toda a matriz de
configurações (`tests/test_compiled.py`, `test_backend_parity.py`, `test_jax_network.py`,
`test_jax_rhs.py`, `test_jax_regime_auto.py`, `test_fv_drift.py`, `test_fv_single.py`), com
`make_jaxpr` a garantir que o núcleo traça sem ramos em valores; o caminho numpy é
bit-idêntico ao anterior (goldens de nível A). Nos casos com ciclos-limite a diferença
pontual entre backends pode chegar a dezenas de por cento porque a sequência de passos muda
e os picos se deslocam no tempo; o ciclo é o mesmo. Cada notebook termina com uma secção
"Lado direito compilado (JAX)" que repete a simulação principal e imprime tempos e diferença.
