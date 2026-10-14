# 029 · Fronteira da intermitência severa: varredura contra Bøe e Taitel

| | |
|---|---|
| tipo | estudo |
| área | análise |
| estado | feito (2026-10-13) |
| onde | `mfsim scan <corridas>` (`mfsim.analysis.scan_summary`) |
| dados | `..._scan_P{2,4}_v{015,03,05,08}.npz`, `..._scan_P2_g{005,02,02_int}.npz`, `..._tol_lh1_{200,500}.npz` |

## Pergunta

Onde o modelo cicla e onde não, comparado com o critério de Bøe (forma correta, 028) e com a
condição de Taitel (1986) de linha estratificada?

## O que se encontrou

Configuração recomendada (padrão fixo, `noslip`, anular > 90 %, massa virtual 0.5, bolhas
< 1 %), linha de 1000 m a −3°, riser de 20 m, 9000 s:

| P_out (bar) | vSL (m/s) | vSG (m/s) | Bøe mín | linha | ΔP_in | blowdowns | intervalo | pico |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.15 | 0.15 | 0.019 | estratificada | 1.82 bar | 2 | 4714 s | 10.2 kg/s |
| 2 | 0.30 | 0.15 | 0.019 | estratificada | 1.62 bar | 3 | 3602 s | 8.6 kg/s |
| 2 | 0.50 | 0.15 | 0.019 | estratificada | 1.44 bar | 4 | 2858 s | 8.2 kg/s |
| 2 | 0.80 | 0.15 | 0.019 | estratificada | 1.33 bar | 4 | 2099 s | 7.2 kg/s |
| 4 | 0.30 | 0.12 | 0.019 | estratificada | 1.65 bar | 4 | 2265 s | 7.9 kg/s |
| 4 | 0.80 | 0.12 | 0.019 | estratificada | 1.42 bar | 6 | 1349 s | 6.2 kg/s |
| 2 | 0.30 | 0.07 | 0.009 | estratificada | 1.63 bar | 2 | 6524 s | 8.6 kg/s |
| 2 | 0.30 | 0.29 | 0.036 | estratificada | 1.67 / 1.83 bar | 4 / 5 | 2017 / 2140 s (bolhas / intermitente) | 8.9 / 9.1 kg/s |
| 4 (L = 300 m) | 1.2 e 1.6 | 0.12 | 0.061 | golfadas | < 0.05 bar | 0 | — | — |
| 4 (L = 300 + 500 m de entrada) | 0.30 | 0.12 | 0.024 | estratificada | 1.95 bar | 3 | 1077 s | 8.2 kg/s |

1. Onde Bøe é satisfeito e a linha é estratificada o modelo dá o ciclo clássico em todos os
   pontos, com intervalo a cair com vSL, com a pressão e com vSG (1350–6500 s).
2. Onde a linha deixa de ser estratificada (vSL ≥ 1.2 m/s em 300 m, 4 bar) o escoamento é
   permanente, como Taitel (1986).
3. A fronteira de Bøe do lado do gás **não** é testada por esta varredura: todos os pontos
   estão dentro da região; para chegar à fronteira a 2 bar seria preciso vSG ≳ 2.5 m/s, onde a
   linha já não é estratificada.
4. O caso base (300 m, 4 bar) é o único ponto em que a resposta depende do fechamento do
   riser (022).
5. Gasoduto a montante como linha de entrada de 500 m: intervalo +29 % e pico +35 % (026).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Fronteira da intermitência severa".
