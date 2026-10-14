# 025 · Vértice dinâmico da tese é um sumidouro de massa quando saturado → vértices conservativos

| | |
|---|---|
| tipo | bugfix |
| área | rede |
| estado | feito (2026-10-13) |
| onde | vértices `tank` no mapa de massas (`NetworkSystem.transform`, `_vertex_rhs`); fluxos de contorno limitados pelo vértice doador |
| dados | `severe_slug_blocked_network_v{002,005,050}.npz`; caso 10.2.2 conservativo; `tests/test_conservative_residual.py` |

## Antes

O vértice dinâmico da tese tem hold-up e pressão como estados e um **limitador do hold-up
nos limites**. Ao montar o severe slug como rede (linha, pescoço, riser, saída ligados por
vértices de volume `vertex_volume`), cada duto conservava massa a 0.01 kg, mas o vértice entre
a linha e o pescoço, saturado de líquido, **absorveu 741 kg dos 1590 kg alimentados sem os
armazenar**: o limitador anula a derivada do hold-up e o volume não retém o excesso. Também
tornava o sistema rígido (475 mil passos).

## Depois

1. Em modo conservativo os vértices entram no mapa de massas, T = (ρG·HG, ρL·(1 − HG)) por
   vértice, com a EDO d(m)/dt = (QG, QL)/V exata e sem limitador — é uma mudança de variáveis,
   não um DAE: mL/ρL(P) + mG/ρG(P) = V é invertível e dá P e HG explicitamente;
2. os fluxos de contorno dos dutos são limitados pelo conteúdo do vértice quando ele impõe a
   fração (o vértice é a célula doadora), e os dutos que chegam a um vértice recebem a sua
   fração no fantasma de saída (014);
3. o balanço do vértice usa exatamente os fluxos limitados das faces vizinhas, e o
   acumulador (005) inclui as vazões líquidas dos vértices.

Numpy e JAX, o mesmo código; vale para junções e bifurcações (N → 1, 1 → N).

## Benefício

- Balanço exato do sistema (dutos + vértices) e passos 8× menores; caso 10.2.2 com o esquema
  conservativo: erro de massa 2e−10 kg de líquido e 1e−8 kg de gás em 3000 s.
- Rede do severe slug comparada ao duto único curvado (3000 s):

| configuração | blowdowns / intervalo | pico | P na entrada | passos | erro de massa |
|---|---|---|---|---|---|
| duto único curvado | 4 / 765 s | 7.9 kg/s | 4.14–6.00 bar | 25.9 mil | 4e−5 kg |
| rede, V = 0.002 m³ | 4 / 748 s | 11.7 kg/s | 4.07–5.94 bar | 65 mil | 2e−2 kg |
| rede, V = 0.005 m³ | 5 / 565 s | 10.0 kg/s | 4.04–5.87 bar | 61 mil | 1e−4 kg |
| rede, V = 0.05 m³ | 6 / 582 s | 4.4 kg/s | 4.32–5.80 bar | 35.5 mil | 1e−5 kg |

Com volume de vértice da ordem de uma célula a rede reproduz o intervalo e a faixa de
pressão do duto único; volumes maiores acrescentam estoque nas junções e mudam o ciclo. A
rede custa 2–3× mais passos (vértices pequenos são rígidos); os picos são mais altos
(junções sem perda de momento).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Severe slug como rede de dutos: vértices dinâmicos conservativos») em 2026-10-14.*

`severe_slug_case(single_pipe=False)` monta o mesmo caso como rede: linha
(horizontal + descendente), pescoço (o arco como duto próprio), riser e saída,
ligados por vértices dinâmicos (volume `vertex_volume`), com a mesma condição
inicial bloqueada interpolada pelos dutos. A primeira tentativa, com a
formulação de vértice da tese (hold-up e pressão como estados, limitador do
hold-up nos limites), falhou de forma instrutiva: cada duto conservava massa a
0.01 kg, mas o vértice entre a linha e o pescoço, saturado de líquido, absorveu
741 kg dos 1590 kg alimentados sem os armazenar (o limitador anula a derivada do
hold-up e o volume não retém o excesso) — um sumidouro de massa que também
tornava o sistema rígido (475 mil passos). Correções, todas em
`NetworkModel` (numpy e JAX, o mesmo código): (1) em modo conservativo os vértices entram no mapa de
massas, T = (ρG·HG, ρL·(1 − HG)) por vértice, com a EDO d(m)/dt = (QG, QL)/V
exata e sem limitador — é uma mudança de variáveis, não um DAE: a relação
mL/ρL(P) + mG/ρG(P) = V é invertível e dá P e HG explicitamente (o DAE só
apareceria no limite incompressível, que a compressibilidade artificial
regulariza); (2) os fluxos de contorno dos dutos são limitados pelo conteúdo do
vértice quando ele impõe a fração de gás (o vértice é a célula doadora), e os
dutos que chegam a um vértice recebem a sua fração no fantasma de saída; (3) o
balanço do vértice usa exatamente os fluxos limitados das faces vizinhas, e o
acumulador do integrador (`boundary_net_fluxes`) inclui as vazões líquidas dos
vértices. Resultado: balanço exato do sistema (dutos + vértices) e passos 8×
menores; a formulação vale para junções e bifurcações (N → 1, 1 → N), verificado
no caso 10.2.2 com o esquema conservativo (erro de massa 2e−10 kg de líquido e
1e−8 kg de gás em 3000 s; `tests/test_conservative_residual.py`).

Comparação com o duto único curvado (mesma malha, formulação conservativa,
ν = 0.05, JAX, `tol_range` 0.01, 3000 s; seção "Rede de dutos" do notebook,
dados `notebooks/data/severe_slug_blocked_network_v*.npz`):

| configuração | blowdowns / intervalo | pico de líquido | P na entrada | passos | erro de massa |
|---|---|---|---|---|---|
| duto único curvado | 4 / 765 s | 7.9 kg/s | 4.14–6.00 bar | 25.9 mil | 4e−5 kg |
| rede, V = 0.002 m³ | 4 / 748 s | 11.7 kg/s | 4.07–5.94 bar | 65 mil | 2e−2 kg |
| rede, V = 0.005 m³ | 5 / 565 s | 10.0 kg/s | 4.04–5.87 bar | 61 mil | 1e−4 kg |
| rede, V = 0.05 m³ | 6 / 582 s | 4.4 kg/s | 4.32–5.80 bar | 35.5 mil | 1e−5 kg |

Com volume de vértice da ordem de uma célula (0.002 m³ contra 0.004 m³ por
célula) a rede reproduz o intervalo entre blowdowns e a faixa de pressão do duto
único; volumes maiores acrescentam estoque nas junções e mudam o ciclo (mais
curto, picos menores com V = 0.05 m³). Os picos de vazão da rede são mais altos
(junções sem perda de momento e vértice como célula de mistura). A rede custa
2–3× mais passos que o duto único: os vértices pequenos são rígidos.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
