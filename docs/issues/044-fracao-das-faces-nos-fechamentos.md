# 044 · Fração de gás dos fechamentos nas faces: a da célula doadora

| | |
|---|---|
| tipo | bugfix |
| área | esquema numérico |
| estado | feito (2026-10-16); opção `closure_alpha` do esquema escalonado, `"mean"` por omissão (corridas antigas e goldens inalterados), `"donor"` no exemplo base do artigo 01 |
| onde | `scheme.options.closure_alpha = "mean" \| "donor"`; `mfsim/discretization/staggered.py` (`_cell_face_states`); `tests/test_closure_alpha.py` |
| dados | sondas de 2026-10-16 (fora do git): estacionário do poço a 15 bar, espetro do estacionário, corridas de 1500–3000 s com as duas opções e duas malhas; caso `cases/severe_slug/well_choke_auto.json` |

## Antes

Na equação de momento de cada face do esquema escalonado (002), a fração de área que multiplica o
gradiente de pressão e a gravidade é a da **célula doadora** de cada fase, mas os fechamentos (atrito de
parede e interfacial, velocidades da relaxação da fase que desaparece) eram avaliados no estado da face
com a **média** das frações das duas células vizinhas. O balanço de cada face em regime permanente
relacionava então a fração doadora com a média, e o estacionário discreto ganhava um modo par-ímpar:

- a fração de gás no riser do caso do poço (043) alternava de célula para célula com amplitude crescente
  para cima: 0,110 / 0,147 / 0,097 / 0,179 / 0,073 / 0,238 / 0,015 (padrão dinâmico, abertura 0,13,
  15 bar), e o mesmo xadrez com o padrão fixo (0,119 / 0,134 / 0,115 / 0,151 / 0,110 / 0,180 / 0,094);
- o espetro do estacionário tinha um modo real rápido e instável na curva (+41 /s com padrão dinâmico,
  +27 /s com padrão fixo, issue 042), preso às células onde o bolsão estratificado encontra o líquido do
  ponto baixo.

## Depois

Com `closure_alpha = "donor"` a fração com que se avaliam os fechamentos de cada face é a da célula
doadora pelo sinal da vazão volumétrica da mistura (escrita como vazão mássica equivalente de líquido,
qL + qG ρL/ρG, para usar as larguras do chaveamento do líquido: `switch_frac` × |q| local + `q_switch_L`).
Todos os termos do balanço da face usam assim a mesma fração. As faces de contorno com fração imposta
pelo vértice continuam com a regra de 014. O mesmo código serve numpy e JAX (paridade 1e−14 nos testes).

## Benefício

- Estacionário liso: no mesmo ponto o riser fica 0,127 / 0,128 / 0,125 / 0,126 / 0,130 / 0,134 / 0,139 /
  0,144 (cresce com a expansão do gás), e o descendente uniforme até ao ponto baixo. Num riser vertical
  isolado em bolhas a amplitude do modo par-ímpar (segundas diferenças) cai 5× (teste).
- O modo de +41 /s da curva desaparece; os modos instáveis que sobram são fracos (≈ 4–5 /s, 40–45 rad/s)
  na linha estratificada e, a partir da abertura 0,22, um modo real na célula em que o bolsão encontra o
  líquido (≈ 20 /s), que satura em amplitude desprezável nas corridas.
- Custo e ciclo: ver a tabela abaixo (malha grossa e fina, 3000 s, referência com padrão dinâmico).

| malha | fração das faces | passos | intervalo entre blowdowns | P_in (bar) | pico (kg/s) |
|---|---|---|---|---|---|
| grossa 15 + 5 | média | (a preencher) | (a preencher) | | |
| grossa 15 + 5 | doadora | 154 mil | 985 s | 4,03–5,84 | 4,4 |
| fina 30 + 10 | média | (a preencher) | | | |
| fina 30 + 10 | doadora | (a preencher) | | | |

## Referências

Issues 002 (esquema escalonado), 014 (faces de contorno), 042 (modos locais da curva), 043 (mapa) e o
artigo 01 (`papers/01-severe-slugging-two-fluid`, secção de discretização).
