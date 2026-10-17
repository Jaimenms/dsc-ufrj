# 054 · Critérios de bolhas com a fração de gás local

| | |
|---|---|
| tipo | bugfix |
| área | identificação de padrão |
| estado | feito (2026-10-17); opção `regime.local_bubble` (com `local`), desligada por omissão |
| onde | `mfsim/closures/regime.py` (`identify_regime(local_bubble=)`); `PipeParams.ipe_local_bubble`; `RegimeSpec.local_bubble`; esquema `mfsim-case-1`; `tests/test_local_bubble.py` |
| dados | caso offshore de 2 km do artigo 01 (1500 m + 500 m a −2°, ponto base, choke aberto, arranque bloqueado, Armijo), 8000 s; estados exatos de Newton falhados gravados no estudo A/B de 2026-10-17 (`runs/ab_study/L2000/`) |

## Antes

No modo local da identificação automática (021), a fração de bolhas dos critérios era
`max(aG_bol, alphaG)`. Aqui, `aG_bol` é a fração de deriva das bolhas calculada das velocidades superficiais
(a que um mapa de regime permanente usa para traduzir o critério de Taitel, Bornea e Dukler, αG = 0,25, em
vazões). No modelo a dois fluidos a fração é um estado, e o máximo fazia o padrão de uma célula com pouco
gás depender da vazão de gás da face.

No caso offshore, a frente de gás entra na curva do riser (78,5°) por uma célula com αG = 0,020. Nessa
célula, `aG_bol` vale 0,27 com qG = 0,15 kg/s, e o padrão passava de bolhas a intermitente com qG crescente:

| qG (kg/s) | padrão | arrasto do gás Γ_G | f_qG |
|---|---|---|---|
| 0,05 | bolhas | 1839 | −922 |
| 0,10 | bolhas (99 %) | 8025 | −5038 |
| 0,15 | 16 % bolhas, 84 % intermitente | 3061 | −1540 |
| 0,20 | intermitente | 433 | −173 |

O arrasto cai 18 vezes quando a vazão de gás sobe, o que equivale a um arrasto efetivo negativo. A derivada
∂f_qG/∂qG chegava a +1e5 s⁻¹ e o maior autovalor de T′⁻¹J a +4,8e4 s⁻¹, uma instabilidade espúria com
constante de tempo de ~20 µs. Com passos de 1e-5–1e-4 s, γ·λ passava de 1 e a matriz do Newton ficava quase
singular: o passo colapsava e a corrida morria no passo mínimo sem que reduzir h ajudasse. Era o ponto frágil
comum a todas as variantes do estudo A/B por volta de 2300 s; atravessá-lo ou não dependia de detalhes da
trajetória (a referência passava com uma busca em linha e falhava com outra).

Com o padrão fixo (intermitente), no mesmo estado, df/dqG ∈ [−954, −27] s⁻¹ e o maior autovalor é 46 s⁻¹. As
opções do esquema (`alpha_face`, `switch_frac`, `relax_time`) e a pressão interfacial não mudam o resultado.

## Depois

Com `local_bubble`, a fração de bolhas dos critérios é só a da célula: a transição bolhas/intermitente a 0,25
e a verificação das bolhas dispersas olham para a fração que existe, e o padrão de uma célula com 2 % de gás
deixa de seguir a vazão de gás. O critério das bolhas dispersas de Barnea (dmax contra ddef e dmigr) continua
a depender da velocidade da mistura, como deve. Sem `local`, a opção não tem efeito.

No estado da falha, f_qG passa a monótona e amortecida em toda a faixa varrida (df/dqG ≤ −50 s⁻¹) e o maior
autovalor de T′⁻¹J cai de 4,8e4 para 4,7 s⁻¹.

## Benefício

Caso offshore de 2 km, 8000 s:

| corrida | período | P na base (bar) | pico de q (kg/s) | passos | mal posto na trajetória |
|---|---|---|---|---|---|
| sem a opção | 1554 s | 18,6–32,4 | 37,8 | 79 560 | 2,66 % |
| `local_bubble` | 1459 s | 17,5–30,8 | 34,5 | 10 033 | 0,22 % |
| `local_bubble` + massa virtual no estratificado | 1463 s | 16,3–30,8 | 42,4 | 9 953 | 0,23 % |

- O integrador dá 8 vezes menos passos e a fração de células mal postas da trajetória é 12 vezes menor.
  A intermitência severa mantém-se, com período 6 % mais curto e picos parecidos.
- A variante com massa virtual também no estratificado, que antes falhava em 2290 s, passa a correr até ao fim.
- As variantes com o limiar de vazio baixo a 0,025 e com pressão interfacial δ = 1,2 ainda falham (2082 e
  4256 s), mas por outra razão: a busca em linha de Armijo (053) testa a descida do resíduo antes do critério
  de convergência e recusa um Newton já convergido (correção 1e-5 contra a tolerância de 1e-3). Corrigido
  na 055.
- Validação de laboratório por fazer: o caso `cases/paper01/lab_fabre2_auto.json` não arranca nem sem esta
  opção (pesos NaN do mapa automático e NaN do fecho intermitente com o escoamento em repouso), um defeito
  anterior corrigido na 057.

## Referências

Issues 021 (identificação automática local), 046 (ponte do filme local) e 053 (Newton); Taitel, Bornea e
Dukler (1980), *Modelling flow pattern transitions for steady upward gas–liquid flow in vertical tubes*,
AIChE J. 26; Barnea (1987), *A unified model for predicting flow-pattern transitions for the whole range of
pipe inclinations*, IJMF 13; artigo 01.
