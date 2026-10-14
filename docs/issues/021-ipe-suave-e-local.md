# 021 · Identificação automática de padrão suave e local (`automatic`, `smooth`, `local`)

| | |
|---|---|
| tipo | feature |
| área | fechamentos / identificação de padrão |
| estado | feito (2026-10-13); recomendação: `automatic` só com `smooth`, `local`, `k_switch = 20` e `low_void` |
| onde | `model.regime = {"type": "automatic", "k_switch": 20, "smooth": true, "local": true}` |
| dados | `severe_slug_blocked_regime6_{smooth,smooth_k50,local}.npz` |

## Antes

Mesmo com os defeitos corrigidos (020), o chaveamento sigmoidal absoluto da tese alterna
fechamentos a cada 16 ms: depois do blowdown fica um bolsão quase parado no joelho
(αG = 0.8, vLS ≈ 0) e o mapa de regime permanente, que só olha as velocidades
superficiais, chama-o de "bolhas" sempre que a velocidade do gás, que toca a ≈ 30 Hz, cai.
Com `smooth` sozinho, 13458 passos até 101 s.

## Depois

- `smooth`: sigmoides com largura **relativa** ao valor crítico (`k_switch` ≈ 20 → transição em
  ±10 % do limiar, igual para critérios de escalas tão diferentes como FG* ≈ 0.02 e K ≈ 1–10);
  equilíbrios com |vLS|, |vGS| (os modelos são co-correntes); hD interpolado na tabela
  (contínuo no estado); sentido do escoamento por tanh(vM/0.02); aG_bol regularizado.
- `local`: os critérios são avaliados no **estado local** em vez dos equilíbrios de regime
  permanente (nível e fração reais no lugar de hD_est/aG_est; fração de bolhas max(aG_bol, αG);
  estratificado impossível em dutos verticais; fração acima de 0.9 é filme de parede).

## Benefício

| padrão (3000 s, configuração de referência) | passos | CPU | intervalo | pico | P_in (bar) | produção média |
|---|---|---|---|---|---|---|
| fixo (estratificado + bolhas no riser) | 25.9k | 50 s | 765 s | 7.9 kg/s | 4.14–6.00 | 0.515 kg/s |
| automático `smooth + local`, k = 20 | 35.3k | 207 s | 851 s | 4.1 kg/s | 4.01–5.86 | 0.498 kg/s |
| automático só `smooth`, k = 20 | 175.9k | 2034 s | 861 s | 5.7 kg/s | 4.01–5.88 | 0.495 kg/s |
| automático da tese (k = 1e6) | — | — | não passa de t ≈ 2 s | | | |

A identificação automática reproduz o ciclo (mesma alimentação, mesma faixa de pressão,
intervalo 11–15 % maior) com picos menores e mais largos: no blowdown o riser é identificado
como intermitente/anular, não bolhas. O modo local custa 5× menos passos e 10× menos CPU do
que o suave sozinho. O custo por passo é ≈ 3× o do padrão fixo (tabelas de equilíbrio por nó).

## Limitações

A escala de laboratório (1 atm) com identificação automática rasteja; os casos de
laboratório usam padrão fixo com as regras físicas (030).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Identificação automática de padrão (regime 6) — correções, opções e resultado») em 2026-10-14.*

**Correções e opções** (numpy e JAX, mesma implementação):

* sempre ativas, sem efeito quando os indicadores são 0/1: `flag_vertical` em
  graus; partição da unidade (Int = (1−Est)(1−Anu)(1−p_bub));
* `low_void_regime` passa a valer também no modo automático (sem gás não há
  padrão a identificar: bolhas abaixo de `low_void_alpha` com a rampa C1);
* `PipeParams.ipe_smooth=True`: sigmoides com largura *relativa* ao valor
  crítico (`k_switch` ≈ 20 → transição em ±10 % do limiar, igual para critérios
  de escalas tão diferentes como FG* ≈ 0.02 e K ≈ 1–10), equilíbrios avaliados
  com |vLS|, |vGS| (os modelos são co-correntes), hD interpolado na tabela
  (contínuo no estado; em JAX o vLS de equilíbrio é reavaliado com gradiente nas
  duas entradas que enquadram vLS), sentido do escoamento por tanh(cos θ·vM/0.02)
  e aG_bol = |vGS|/(|1.25 vM + ub| + 1e−3);
* `PipeParams.ipe_local=True`: os critérios são avaliados no **estado local** em
  vez dos equilíbrios de regime permanente: nível e fração reais (hD, αG) no
  lugar de hD_est/aG_est (Kelvin–Helmholtz, Jeffreys, trajetória e turbulência
  julgam a interface que existe, e deixa de haver busca em tabela nesses
  critérios); fração de bolhas max(aG_bol, αG) (máximo suave); estratificado
  impossível em dutos verticais (os critérios degeneram com cos θ → 0 e o
  indicador de sentido oscila com vM ≈ 0); fração de gás acima de 0.9 é filme de
  parede (anular) qualquer que seja a velocidade do gás. Motivo: depois do
  *blowdown* fica um bolsão quase parado no joelho (αG = 0.8, vLS ≈ 0) e o mapa
  de regime permanente, que só olha as velocidades superficiais, chama-o de
  "bolhas" sempre que a velocidade do gás, que toca a ~30 Hz, cai (aG_bol entre
  0.02 e 0.6); o fechamento alterna bolhas ↔ intermitente a cada 16 ms e
  sustenta o toque (passos de 0.1–1 ms em toda a fase rápida).

Custo da partida bloqueada (JAX, mesma configuração da referência): até 10 s,
`ipe_smooth` com k = 20 e bolhas abaixo de 1 % gasta 170 passos (padrão fixo:
370; k = 50: 427); até 101 s (primeiro *blowdown* incluído), `ipe_smooth`
13458 passos e `ipe_smooth + ipe_local` 986. O modo da tese (k = 1e6, absoluto)
continua disponível e é o usado nos testes de equivalência.

**Resultado (3000 s, malha 15/5, arco, conservativa, ν = 0.05, JAX, `tol_range`
0.01; dados em `notebooks/data/severe_slug_blocked_regime6_*.npz`):**

| padrão | passos | CPU | blowdowns (s) | intervalo | pico de líquido | P_in (bar) | produção média | erro de massa |
|---|---|---|---|---|---|---|---|---|
| fixo (estratificado + bolhas no riser, mistura 20°) | 25.9k | 50 s | 30, 740, 1518, 2324 | 765 s | 7.9 kg/s | 4.14–6.00 | 0.515 kg/s | 0 |
| IPE, `ipe_smooth + ipe_local`, k = 20, bolhas < 1 % | 35.3k | 207 s | 34, 870, 1726, 2588 | 851 s | 4.1 kg/s | 4.01–5.86 | 0.498 kg/s | 0 |
| IPE, só `ipe_smooth`, k = 20, bolhas < 1 % | 175.9k | 2034 s | 46, 876, 1750, 2628 | 861 s | 5.7 kg/s | 4.01–5.88 | 0.495 kg/s | 6e−4 kg |
| IPE, só `ipe_smooth`, k = 50, bolhas < 1 % | 163.1k | 1892 s | 36, 876, 1760, 2684 | 883 s | 5.4 kg/s | 3.98–5.91 | 0.491 kg/s | 1e−4 kg |
| IPE da tese (k = 1e6), com ou sem bolhas < 1 % | — | — | não passa de t ≈ 2 s (27k passos em 4.5 min) | | | | | |

A identificação automática reproduz o ciclo de intermitência severa (mesma
alimentação 0.530 kg/s, mesma faixa de pressão, intervalo entre *blowdowns* 11–15 %
maior) com picos de líquido menores e mais largos: o padrão identificado no riser
durante o *blowdown* é intermitente/anular (não bolhas), com mais atrito
interfacial. As três variantes do IPE concordam entre si no ciclo; o modo local
custa 5× menos passos e 10× menos CPU do que o suave sozinho, que passa a fase
rápida a alternar bolhas ↔ intermitente no bolsão do joelho. O que ela identifica ao longo do ciclo (mapa no notebook): linha
descendente estratificada, bolsão do joelho estratificado, riser cheio de líquido
= bolhas (regra de fração baixa), riser com gás = intermitente → anular (filme) à
medida que esvazia, trecho horizontal de entrada estratificado/intermitente
conforme a fração de gás. O custo por passo é ~3× o do padrão fixo (tabelas de
equilíbrio de 2 × 700 entradas por nó em cada avaliação).

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
