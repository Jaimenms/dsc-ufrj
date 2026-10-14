# 024 · Arraste e deposição de gotas no anular (item 7)

| | |
|---|---|
| tipo | estudo |
| área | fechamentos |
| estado | avaliado (2026-10-13); sem modelo de gotas |
| onde | pós-processamento no `nb_severe_slug` (fração arrastada de Ishii & Mishima, 1989) |
| dados | corridas de referência, linha de entrada de 500 m, linha longa |

## Pergunta

A tese não tem arraste de gotas. Faz falta nestes casos?

## O que se encontrou

E = tanh(7.25e−7·We^1.25·Re_L^0.25) avaliado nas células do riser das trajetórias gravadas:

| caso | riser > 90 % de gás | E > 0.1 nesse tempo | E máximo (vSG máx) | líquido produzido com E_topo > 0.1 |
|---|---|---|---|---|
| base, regras físicas | 21 % | 0 % | 0.12 (9.3 m/s) | 0 % |
| base + 500 m de linha de entrada | 47 % | 0.1 % | 0.16 (11.1 m/s) | 5 % |
| linha de 1000 m, 2 bar | 11 % | 1.0 % | 0.56 (28.9 m/s) | 1.1 % |
| linha de 1000 m, 2 bar, vSG = 0.29 | 16 % | 0.5 % | 0.61 (27.4 m/s) | 1.5 % |

O arraste só é apreciável (E > 0.5) nos poucos segundos do pico de gás do blowdown dos
casos de linha longa, e o líquido produzido nesses instantes é 1–5 % do total. Não se
justifica um campo de gotas; se um dia for preciso, o caminho é um termo de arrasto de
gotas no fechamento anular (fração E do líquido forçada à velocidade do gás com tempo de
resposta ≈ 0.03–0.05 s).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Arraste de gotas no anular (item 7)".
