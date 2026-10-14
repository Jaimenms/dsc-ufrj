# 027 · Condições iniciais por receita: linha bloqueada e linha fechada

| | |
|---|---|
| tipo | feature |
| área | condições iniciais |
| estado | feito (2026-10-13) |
| onde | `ic = {"recipe": "blocked", ...}` ou `{"recipe": "shut_in", ...}` no duto; `mfsim/initial/recipes.py`; `initial_state.source = "guess"` para a estimativa |
| dados | `severe_slug_blocked_*` (campo), `severe_slug_lab_*` (laboratório) |

## Antes

A simulação partia de uma estimativa de escoamento estabelecido (`guess`) e o primeiro
blowdown só aparecia em ≈ 2500 s de transiente de pressurização — cada experimento custava
uma hora antes de o ciclo começar, e a condição inicial não ficava registada de forma
reprodutível.

## Depois

- `blocked`: bolsão de gás estratificado na linha na fração de equilíbrio (0.746, igual à
  observada no ciclo), riser cheio de líquido, pressões hidrostáticas e bolsão a uma margem
  (`dP_margin_bar`, 0.3 bar no campo) acima da frente de líquido;
- `shut_in`: linha fechada (linha com `alpha_line`, riser com `alpha_riser`) e rampa das
  vazões de alimentação; é a partida dos casos de laboratório, onde a partida bloqueada com
  0.05 bar de sobrepressão dá um blowdown inicial violento à escala de 1 atm.

A receita fica no JSON do caso e nas corridas gravadas (`case_json`), portanto o estado
inicial é reconstruído sem npz.

## Benefício

O ciclo começa em poucos minutos simulados (primeiro slug em ≈ 12 s na partida bloqueada):
as corridas de 3000 s passaram a cobrir 4–5 blowdowns e tornaram viáveis os estudos de
sensibilidade (016–024). A sobrepressão inicial muda o primeiro blowdown mas não o ciclo
(linha longa: 0.3 e 0.1 bar dão 3916 e 3928 s).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Condição inicial bloqueada"; [docs/data_formats.md](../data_formats.md).
