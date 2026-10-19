# 062 · Janela da paragem adaptativa escalada com o período (ciclos rápidos) e classe "ciclo rápido" nos marcadores

| | |
|---|---|
| tipo | feature |
| área | integração (paragem adaptativa, 041/061), análise (`cycle_class`), ferramentas (massa de dados do artigo 02) |
| estado | feito (2026-10-19); chaves novas de `time.settle` (`window_periods`, `window_min`) e parâmetro novo de `cycle_class` (`period_fast`), todos opcionais: sem eles nada muda (goldens intactos) |
| onde | `mfsim/simulation.py` (`effective_window`, `sub_signatures`, `_integrate_settled`), `mfsim/io/schema/mfsim-case-1.schema.json`, `mfsim/analysis.py` (`cycle_class(period_fast=…)`), `tools/paper02_dataset.py` (`--window-periods`, `--window-min`, `PERIOD_FAST`), `papers/02-…/figures/make_figures.py` (classe `fast`); `tests/test_settle.py` |
| dados | caso 10.1.1 da tese (teste); campanhas do artigo 02 (`runs/paper02/base2d*`) |

## Antes

A paragem adaptativa compara janelas inteiras de `window` segundos (3000 s nas campanhas do artigo 02, escolhidos
para caberem 2 a 6 períodos de golfada severa, 500–1500 s) e aceita um ciclo quando duas janelas consecutivas têm o
mesmo envelope. No canto de abertura alta × pressão de gás alta (≈ 165–245 bar, abertura ≥ 0,5) o modelo tem um
ciclo rápido do topo do riser: período ≈ 60 s, líquido no choke de 2 a 16 kg/s, fração de gás das células do topo
0,6–0,95, cabeça do poço plana (± 0,2 bar), injeção constante. Esse ciclo só era aceite aos 9000 s (a primeira
janela leva o transiente de arranque), a 0,3 s por passo — 30 mil passos, mais do que o orçamento de 40 min —, e 7
de 9 pontos do canto esgotavam o orçamento. O ciclo aparece abruptamente no reinício do integrador entre janelas,
depois de a amplitude decair a 0,001 na janela anterior (modo instável que passos grandes do BDF amortecem e os
pequenos resolvem); a (1,0, 260 bar) o reinício não excita nada — aí o poço é mesmo estável. Nos marcadores,
`cycle_class` lia esse ciclo como golfada severa sempre que o pico passava o dobro da média, com 0,3 bar de
oscilação na cabeça contra 2,3–17 bar da golfada severa.

## Depois

- `time.settle.window_periods = N` (omissão 0 = desligado) e `window_min` (s, omissão 600): no fim de cada janela,
  se algum sinal vigiado cicla (≥ 4 cruzamentos, período finito) e N·período < `window`, a janela de comparação
  passa a L = max(N·período, `window_min`), avaliada **já sobre a série integrada** (os dois últimos troços de L
  segundos, `sub_signatures`), e as janelas seguintes passam a ter L segundos. Um ciclo de 60 s com N = 10 é aceite
  aos 3000 s em vez dos 9000; a captura de N_c ciclos continua a usar o período medido. `info.settle.window_eff`
  regista L. Ciclos lentos (10 períodos não cabem na janela) e estacionários: nada muda.
- `cycle_class(..., period_fast=300)`: classe `fast` quando a pressão cicla com período menor do que `period_fast`;
  sem o parâmetro, a classe é a de antes. Nas três campanhas os ciclos assentados de golfada e de pressão têm
  período ≥ 546 s e os ciclos rápidos 52–108 s. Os marcadores do artigo 02 usam 300 s (`PERIOD_FAST`); a tabela e
  o mapa das decisões ganham a classe.
- `tools/paper02_dataset.py design`: `--window-periods` (omissão 10) e `--window-min` (600 s) na especificação da
  paragem; as campanhas em curso receberam as chaves nos `design.json` das rondas e os pontos falhados voltaram à
  fila (`run --retry-failed --from-base`).
- `run --only ids` corre só os pontos indicados; `run --resume-exhausted` retoma um ponto que esgotou o orçamento a
  partir do seu próprio `final_state.npz` (start `resume`, orçamento inteiro outra vez, marcadores da continuação) —
  pedido do dono: no canto, o transiente de arranque custa a hora inteira e só a continuação chega ao ciclo.

## Benefício medido

Ciclo rápido de 60 s: aceitação aos 3000 s em vez de 9000 (teste sintético: `effective_window` → 600 s,
`sub_signatures` + `settled` → "cycle"); nas campanhas, um ponto do canto passa de > 40 min (orçamento esgotado) para
≈ 10 mil passos. Caso 10.1.1 com a opção ligada: resultado idêntico ao de antes (estacionário aos 40 s).

## Referências

- 041 (paragem adaptativa) e 061 (vários sinais, envelope, captura): a janela e o critério que esta issue encurta.
- `papers/02-gas-lift-surrogate-optimization/PLAN.md`, secção "Verificação do ótimo e o ciclo rápido do topo": o
  diagnóstico (figura `why_not_settled`, corridas (1,0, 224 bar) e (0,85, 240 bar) do caso base) e a contraprova a 260 bar.
- Artigo 02, §3.2 (orçamento e ciclo rápido) e §3.3 (classes).
