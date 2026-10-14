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

## Referências

[docs/jax_backend.md](../jax_backend.md); [docs/severe_slug.md](../severe_slug.md) § "Lado direito em JAX".
