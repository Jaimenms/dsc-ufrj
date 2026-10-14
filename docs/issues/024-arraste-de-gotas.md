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

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Arraste de gotas no anular (item 7)») em 2026-10-14.*

A tese não tem arraste e deposição de gotas; antes de modelar, estimou-se se o
arraste importa nestes casos. Fração arrastada de Ishii & Mishima (1989),
E = tanh(7.25e−7·We^1.25·Re_L^0.25) com We = ρG·jG²·D/σ·(Δρ/ρG)^(1/3) e
Re_L = ρL·jL·D/μL, avaliada em pós-processamento nas células do riser das
trajetórias gravadas (célula no `nb_severe_slug`):

| caso | riser > 90 % de gás | E > 0.1 nesse tempo | E máximo (vSG máx) | líquido produzido com E_topo > 0.1 |
|---|---|---|---|---|
| base, regras físicas | 21 % | 0 % | 0.12 (9.3 m/s) | 0 % |
| base + 500 m de linha de entrada | 47 % | 0.1 % | 0.16 (11.1 m/s) | 5 % |
| linha de 1000 m, 2 bar, bolhas + regras | 11 % | 1.0 % | 0.56 (28.9 m/s) | 1.1 % |
| linha de 1000 m, 2 bar, vSG = 0.29 | 16 % | 0.5 % | 0.61 (27.4 m/s) | 1.5 % |

O arraste só é apreciável (E > 0.5) nos poucos segundos do pico de gás do
blowdown dos casos de linha longa (vSG de 27–29 m/s no topo), e o líquido
produzido nesses instantes é 1–5 % do total. Conclusão: as gotas não mudam o
ciclo nem os picos destes casos; podem alterar a forma do pico de produção por
alguns segundos. Não se justifica um campo de gotas no modelo; se um dia for
preciso, o caminho é um termo de arrasto de gotas no fechamento anular
(fração E do líquido forçada à velocidade do gás com tempo de resposta
≈ 0.03–0.05 s), o que fica anotado.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
