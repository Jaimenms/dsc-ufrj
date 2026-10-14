# 028 · Critério de Bøe corrigido

| | |
|---|---|
| tipo | bugfix |
| área | análise |
| estado | feito (2026-10-13); tabelas da varredura e do item 5 recalculadas |
| onde | `mfsim.analysis.boe_criterion(vSL, vSG, P, rhoL, alphaG, L_gas)`; `mfsim scan` |
| dados | tabela da fronteira em [docs/severe_slug.md](../severe_slug.md) |

## Antes

A função dividia L por sin|θ|, o que não consta das fontes (Taitel 1986; Jansen et al.
1996) e dava valores 19× maiores a −3°: vSL_mín = 1.03 m/s no caso base, o que levou a
classificar o caso como "fora de Bøe" e a interpretar a ausência de ciclo com o riser
intermitente (022) como previsão do critério.

## Depois

Forma das fontes, vSL ≥ P·vSG / (ρL·g·(1 − αG)·L), com vSG à pressão do separador,
αG = 0.5 e L = linha de entrada + descendente (sem fator de inclinação).

## Benefício

- Caso base: vSL_mín = 0.06 m/s (está **dentro** da região de Bøe); a falta de ciclo com o
  intermitente passa a ser lida como limitação do fechamento, não do critério.
- Verificação externa: Fabre caso 2 fica no limiar (vSL_mín ≈ 0.13 contra 0.13 de
  alimentação) e o caso 1 fora (0.31 > 0.13), coerente com os tipos observados (ciclo
  clássico e penetração contínua).
- Todos os pontos da varredura (029) ficam dentro da região (vSL_mín 0.009–0.061).

## Referências

[docs/lab_validation.md](../lab_validation.md) achado 2; [docs/severe_slug.md](../severe_slug.md) § "Fronteira".
