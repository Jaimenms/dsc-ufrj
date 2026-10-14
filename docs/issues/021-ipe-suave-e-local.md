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

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Identificação automática de padrão (regime 6)".
