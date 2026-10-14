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

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Fronteira da intermitência severa: varredura contra Bøe e Taitel») em 2026-10-14.*

Varredura com a configuração recomendada (padrão fixo, `noslip`, anular > 90 %,
massa virtual 0.5, bolhas < 1 %), linha de 1000 m a −3°, riser de 20 m,
sobrepressão inicial 0.1 bar, 9000 s. "Bøe mín" é o vSL mínimo do critério de
Bøe na forma de Taitel (1986) e Jansen et al. (1996), vSL ≥ P vSG/(ρL g (1 − αG) L),
com vSG à pressão do separador, αG = 0.5 e L = linha de entrada + descendente
(coluna recalculada em 2026-10-13: a versão anterior de `boe_criterion` dividia L
por sin|θ|, o que não consta das fontes e dava valores 19× maiores, como o 1.03 m/s
do caso base); "linha" é o padrão que a
identificação da tese atribui ao estado médio da linha (`mfsim scan`, `mfsim.analysis.scan_summary`).

| P_out (bar) | vSL (m/s) | vSG (m/s) | Bøe mín | linha | ΔP_in | blowdowns | intervalo | pico |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.15 | 0.15 | 0.019 | estratificada | 1.82 bar | 2 em 9000 s | 4714 s | 10.2 kg/s |
| 2 | 0.30 | 0.15 | 0.019 | estratificada | 1.62 bar | 3 | 3602 s | 8.6 kg/s |
| 2 | 0.50 | 0.15 | 0.019 | estratificada | 1.44 bar | 4 | 2858 s | 8.2 kg/s |
| 2 | 0.80 | 0.15 | 0.019 | estratificada | 1.33 bar | 4 | 2099 s | 7.2 kg/s |
| 4 | 0.30 | 0.12 | 0.019 | estratificada | 1.65 bar | 4 | 2265 s | 7.9 kg/s |
| 4 | 0.50 | 0.12 | 0.019 | estratificada | 1.49 bar | 4 | 1863 s | 7.3 kg/s |
| 4 | 0.80 | 0.12 | 0.019 | estratificada | 1.42 bar | 6 | 1349 s | 6.2 kg/s |
| 2 | 0.30 | 0.07 | 0.009 | estratificada | 1.63 bar | 2 | 6524 s | 8.6 kg/s |
| 2 | 0.30 | 0.29 | 0.036 | estratificada | 1.67 bar (bolhas) / 1.83 (intermitente) | 4 / 5 | 2017 s / 2140 s | 8.9 / 9.1 kg/s |
| 4 (L = 300 m) | 0.30 | 0.12 | 0.061 | estratificada | 1.8 bar (bolhas) / 0.2 (intermitente) | ciclo só com bolhas | 749 s | 7.1 kg/s |
| 4 (L = 300 m) | 1.2 e 1.6 | 0.12 | 0.061 | golfadas | < 0.05 bar | 0 | — | — |
| 4 (L = 300 m + 200 m de linha de entrada) | 0.30 | 0.12 | 0.039 | estratificada | 1.88 bar | 4 | 841 s | 6.9 kg/s |
| 4 (L = 300 m + 500 m de linha de entrada) | 0.30 | 0.12 | 0.024 | estratificada | 1.95 bar | 3 | 1077 s | 8.2 kg/s |

Leitura: (i) onde Bøe é satisfeito e a linha é estratificada o modelo dá o ciclo
clássico em todos os pontos, com intervalo a cair com vSL (o riser enche mais
depressa), com a pressão (gás mais denso) e com vSG (bolsão pressuriza mais
depressa), de 1350 s a 6500 s; (ii) onde a linha deixa de ser estratificada o
escoamento é permanente (vSL ≥ 1.2 m/s em 300 m, 4 bar); (iii) a fronteira de
Bøe do lado do gás **não** é testada por esta varredura: todos os pontos,
inclusive vSG = 0.29 m/s (vSL_mín = 0.036), estão dentro da região de Bøe, e o
modelo cicla com os dois fechamentos do riser (2017 e 2140 s); para chegar à
fronteira a 2 bar seria preciso vSG ≳ 2.5 m/s, onde a linha já não é
estratificada e Taitel (1986) exclui o bloqueio; (iv) o caso base de 300 m a 4 bar é
o único ponto em que a resposta depende do fechamento do riser; (v) gasoduto a
montante (linha de entrada de 500 m, ≈ 0.74 m³ de gás a mais): intervalo +29 %
e pico +35 %, o comportamento documentado nos experimentos para o volume a
montante. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_scan_P{2,4}_v{015,03,05,08}.npz`,
`..._scan_P2_g{005,02,02_int}.npz`, `..._tol_lh1_{200,500}.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
