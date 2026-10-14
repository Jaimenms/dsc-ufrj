# 038 · Derivadas internas do Beggs & Brill: exatas (jvp) em vez de diferenças finitas

| | |
|---|---|
| tipo | feature |
| área | modelo de deriva / backend JAX |
| estado | feito (2026-10-13); `derivatives = "exact"` por omissão no JAX, `"fd"` reproduz a tese (e é o que o numpy usa sempre) |
| onde | `model.options.derivatives = "exact" | "fd"` |
| dados | casos 10.1.4, 10.3.1, 10.3.2, 10.3.3 e gas lift |

## Antes

A tese calcula ∂(momento, fluxo de momento, hold-up)/∂(P, qL, qG) do modelo de deriva por
diferenças finitas com perturbação de 1e−3 relativo: três avaliações extra do fechamento
por nó e uma média lateral nos pontos não diferenciáveis.

## Depois

No backend JAX as derivadas são exatas por `jax.jvp` nas três direções, avaliadas a
1e−6 kg/s do lado do escoamento (em vazão de gás exatamente nula a correlação está na
região cortada e a derivada automática é zero, o que no esquema escalonado de deriva, 012,
derrubava o coeficiente no piso e fazia o lado direito explodir).

## Benefício

Onde o fechamento é diferenciável as duas concordam a ≈ 1e−4 relativo; nos pontos não
diferenciáveis (clips do hold-up em λL, troca de fórmula do atrito bifásico em y = 1.2,
`max(C, 0)`) a diferença finita faz uma média lateral e a automática devolve a derivada
lateral, ambas legítimas. Custo: o lado direito dobra nos casos com Beggs & Brill (os jvp
substituem três avaliações por propagação de tangentes); as trajetórias dos casos sensíveis
(10.3.1, gas lift) mudam ≈ 10 % pontualmente, como entre quaisquer duas Jacobianas
diferentes, sem mudar os ciclos.

## Detalhe

*Texto movido de `docs/jax_backend.md` (secção «Lado direito em JAX para toda a tese — Derivadas internas do Beggs & Brill») em 2026-10-14.*

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

## Referências

[docs/jax_backend.md](../jax_backend.md) (visão geral). O texto integral está na secção «Detalhe» acima.
