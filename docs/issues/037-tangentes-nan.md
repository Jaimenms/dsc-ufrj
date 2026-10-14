# 037 · Tangentes NaN na diferenciação automática (0 × ∞ e sigmoide no corte)

| | |
|---|---|
| tipo | bugfix |
| área | fechamentos / backend JAX |
| estado | feito (2026-10-13 e 2026-10-13 noturna) |
| onde | `_safe_sqrt`/*double where* nos fechamentos; `bk.sigmoid` (logística estável); guardas em `dbmax` (bolhas) e `dmax` (identificação de padrão) |
| dados | `tests/test_jax_rhs.py`, `tests/test_jax_regime_auto.py`, `tests/test_backend_parity.py::test_closure_tangents_finite_at_rest`, `tests/test_compiled.py::test_lab_shut_in_jacobian_finite` |

## Antes

Três fontes de NaN só na Jacobiana (os valores eram finitos), que derrubavam o integrador
com "Factor is exactly singular":

1. raiz quadrada de quantidade que se anula: a derivada de sqrt(r) em r = 0 é ∞ e o produto
   pela tangente nula dá 0 × ∞ = NaN;
2. a sigmoide `1/(1 + exp(clip(−k(x1 − x2), ±700)))` do chaveamento de padrão tem tangente
   NaN quando o argumento encosta no corte (exp ≈ 1e304 vezes uma derivada grande dá
   inf/inf); com k = 1e6 isso acontece sempre que um critério está a 7e−4 do limiar;
3. o diâmetro máximo de bolha de Hinze (`dbmax`) e o `dmax` da identificação de padrão com a
   mistura **parada** (linha fechada, arrasto físico): encontrado pela suíte noturna.

## Depois

*Double where* nas raízes (o ramo nulo não é diferenciado), logística estável
(`jax.nn.sigmoid`, mesmos valores a 1e−300) no chaveamento de padrão e no Beggs & Brill, e
guardas `where(arg == 0, inf, ...)` nos diâmetros máximos — tudo sem alterar os valores no
caminho numpy (goldens de nível A intactos).

## Benefício

Jacobiana finita em todos os estados dos casos de campo e de laboratório, incluindo a
partida de linha fechada; testes que avaliam as tangentes em repouso. Diferença legítima
que fica: nas quinas (clips em α = 0, chaveamentos em q = 0) diferenças finitas e
diferenciação automática divergem, e os testes de paridade comparam longe das quinas.

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Lado direito em JAX" e "Identificação automática"; [docs/jax_backend.md](../jax_backend.md).
