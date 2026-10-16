# 045 · Orçamento de tempo de relógio da integração

| | |
|---|---|
| tipo | feature |
| área | integrador |
| estado | feito (2026-10-16); usado nos lotes de comparação do artigo 01 |
| onde | `integration.max_cpu` (s) no JSON; `ImplicitBDF.max_cpu`; `integrate_scipy(max_cpu=)`; `tests/test_max_cpu.py` |
| dados | `batches/paper01_ablation.json`, `batches/paper01_discretization.json` |

## Antes

Uma corrida que rasteja (passos de microssegundos durante horas) só parava no passo mínimo do BDF
próprio (`min_step`, 1e−9 s por omissão no severe slug) ou nunca, no BDF do SciPy. Nos lotes que comparam
configurações que se espera falharem (esquemas colocados, ablação dos ingredientes) não havia forma de
limitar o custo sem mexer no passo mínimo, e um passo mínimo de 1e−6 s faz falhar a própria referência
(o primeiro blowdown da partida bloqueada do caso do poço precisa de passos menores, aos 8 s).

## Depois

`integration.max_cpu` é um orçamento de tempo de relógio da integração. No BDF próprio, excedido o
orçamento, a integração termina com o erro «orçamento de tempo excedido em t=…» e, com `partial` (os lotes),
fica gravada a trajetória até aí. No SciPy (`integrator = "bdf" | "radau" | "LSODA"`), com orçamento, a
integração faz-se em blocos de instantes de saída e pára entre blocos; sem orçamento é uma só chamada ao
`solve_ivp`, como antes. Com a paragem adaptativa (041) o orçamento vale por janela.

## Benefício

Os lotes de comparação do artigo 01 correm com o passo mínimo normal e 5400 s de relógio por corrida: uma
configuração que rasteja fica registada como «pára em t = …» com a trajetória parcial, em vez de bloquear o
lote. Sem orçamento o comportamento é idêntico ao anterior (goldens inalterados).

## Referências

Issues 032 (BDF próprio) e 041 (paragem adaptativa); artigo 01 (`papers/01-severe-slugging-two-fluid`).

## Adenda (2026-10-16): falhas do SciPy com a trajetória parcial

Com `partial=True` (o que o CLI e os lotes usam), uma exceção dentro do `solve_ivp` — no artigo 01 a fatoração LU
singular da matriz de iteração do BDF do SciPy — passa a devolver a trajetória até ao último bloco concluído e o erro
com o último instante em que o lado direito foi avaliado («… em t=…»), como o BDF próprio já fazia; a corrida fica
`PARCIAL` em vez de `FALHOU` e o instante da falha entra nas tabelas. Sem `partial` a exceção sobe como antes. Teste
em `tests/test_max_cpu.py`.
