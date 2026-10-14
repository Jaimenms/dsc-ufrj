# 026 · Volume de gás a montante (item 6)

| | |
|---|---|
| tipo | feature (resultado parcial) |
| área | rede / condições de contorno |
| estado | feito em parte (2026-10-13): o tanque de mistura não representa um gasoduto; vértice separador pendente |
| onde | `severe_slug_case(upstream_volume=V)` → vértice `tank` na entrada; alternativa `L_h1` (linha de entrada mais longa) |
| dados | `..._tol_{vup01,vup05,vup20,lh1_200,lh1_500}.npz` |

## Antes

A alimentação era um vértice de fronteira com vazões constantes; o "gasoduto
comprimível" a montante, que é o parâmetro L do critério de Bøe, não tinha representação.

## Depois

Um vértice dinâmico de volume V recebe as vazões de alimentação; o duto recebe αG e P do
vértice e as vazões da face de entrada passam a ser estados.

## O que se encontrou

| V_up (m³) | passos | intervalo | pico de líquido | ΔP_in |
|---|---|---|---|---|
| 0 (referência) | 16.8k | 833 s | 6.1 kg/s | 1.85 bar |
| 0.1 | 17.9k | 845 s | 6.4 kg/s | 1.86 bar |
| 0.5 | 17.6k | 749 s | 7.3 kg/s | 1.87 bar |
| 2.0 | 20.9k | 769 s | 9.3 kg/s | 1.91 bar |

O ciclo mantém-se e o pico cresce, mas o vértice **não representa um gasoduto**: é um tanque
de mistura cuja fração de gás vai para ≈ 0.05 (o líquido acumula até a composição de saída
igualar a de entrada), de modo que o volume de gás efetivo com V = 2 m³ é ≈ 0.1 m³ e a taxa
de pressurização da linha não muda (0.24 bar por 1000 s com e sem). O caminho físico é
prolongar a linha horizontal de entrada, onde o estratificado guarda o gás no topo e passa o
líquido por baixo: com L_h1 = 500 m (≈ 0.74 m³ de gás a mais) o intervalo sobe 29 % e o
pico 35 %, o comportamento documentado nos experimentos para o volume a montante.

## Pendente

Vértice do tipo separador (gás acumulado no topo, líquido a passar, fase certa em refluxo),
já na lista de trabalho futuro; e o choke no topo / gas lift (item 9).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Volume de gás a montante (item 6)») em 2026-10-14.*

`severe_slug_case(upstream_volume=V)` troca o vértice de alimentação por um
vértice dinâmico de volume V (m³) que recebe as vazões de alimentação; o duto
recebe αG e P do vértice e as vazões da face de entrada passam a ser estados. É
o "gasoduto comprimível" que entra no critério de Bøe. Caso base com a
configuração recomendada (`noslip`, anular > 90 %, massa virtual, bolhas < 1 %;
o volume de gás da própria linha é ≈ 0.44 m³):

| V_up (m³) | passos | blowdowns (s) | intervalo | pico de líquido | ΔP_in |
|---|---|---|---|---|---|
| 0 (referência) | 16.8k | 786, 1618, 2452 | 833 s | 6.1 kg/s | 1.85 bar |
| 0.1 | 17.9k | 516, 1370, 2206 | 845 s | 6.4 kg/s | 1.86 bar |
| 0.5 | 17.6k | 632, 1100, 1976, 2878 | 749 s | 7.3 kg/s | 1.87 bar |
| 2.0 | 20.9k | 908, 1714, 2446 | 769 s | 9.3 kg/s | 1.91 bar |

O ciclo mantém-se e o pico de líquido cresce (6.1 → 9.3 kg/s). Mas o vértice
**não representa um gasoduto a montante**: é um tanque de mistura cuja fração de
gás vai para ≈ 0.05 (o líquido alimentado acumula até a composição de saída
igualar a de entrada), de modo que o volume de gás efetivo com V = 2 m³ é
~0.1 m³ e a taxa de pressurização da linha não muda (0.24 bar por 1000 s com e
sem o vértice, no caso de linha longa). O que falta é o vértice tipo separador
(já na lista de pendências): gás acumulado no topo, líquido a passar. Para
representar a tubulação a montante o caminho físico é prolongar a linha
horizontal de entrada (`L_h1`), onde o estratificado guarda o gás no topo e passa
o líquido por baixo; os resultados com L_h1 = 200 e 500 m (gás da linha de
entrada ≈ 0.3 e 0.74 m³, contra 0.44 m³ no trecho descendente) estão na tabela
da fronteira. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{vup01,vup05,vup20,lh1_200,lh1_500}.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
