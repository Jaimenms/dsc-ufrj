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

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Volume de gás a montante (item 6)" e tabela da fronteira.
