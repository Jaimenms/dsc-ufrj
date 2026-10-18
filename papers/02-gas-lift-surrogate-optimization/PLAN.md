# Artigo 02 — plano: gas lift ao longo da vida com restrição de estabilidade dinâmica

*Documento de trabalho do artigo (português, como o resto da documentação; o manuscrito é em inglês).
Escrito em 2026-10-17 a partir do pedido do dono e da discussão do escopo; atualizar à medida que as decisões e
as corridas chegam.*

## O pedido, em uma frase

Pegar no modelo dinâmico do artigo 01, acrescentar o gas lift modelado desde a pressão de gás na plataforma, gerar
algumas centenas de simulações dinâmicas sorteando o estado do poço (IPR, RGO), a pressão de gas lift e a abertura
do choke, treinar com elas um modelo substituto (redes neurais), e otimizar a operação. A função objetivo junta a
receita do óleo, o custo da compressão do gás e uma penalização da intermitência severa. Por fim, seguir o IPR e o
RGO ao longo da vida dos poços e mostrar a pressão de gas lift e a abertura do choke ótimas em cada fase.

## Decisões do dono (2026-10-17)

- **Vários poços, todos com a mesma geometria** (a do artigo 01), a partilhar o gás de lift e, portanto, o
  compressor. O acoplamento que dá sentido à otimização é a alocação de um gás limitado entre poços em fases
  diferentes da vida.
- **Água como limitação declarada.** O modelo é bifásico (óleo e gás); a fração de água, que costuma dominar a
  economia do gas lift no fim da vida, fica fora e é discutida como limitação.
- **Gas lift modelado desde a pressão na plataforma**, no início da linha de gas lift.

## Decisões do dono (2026-10-18)

- **Caso base com gas lift construído sobre a geometria do artigo 01** (poço, linha de 2 km e riser iguais), com
  uma linha de gás da plataforma à cabeça do poço, um anular e uma válvula de retenção na base do poço; é o caso
  base do artigo e não substitui o do artigo 01.
- **Especificação de pressão do gás na plataforma** como controlo; desligar o gas lift é baixar essa pressão até a
  retenção fechar, e o caso fica igual à geometria sem gas lift (o ramo fica inerte).
- **Modelo a dois fluidos em todos os dutos**, incluindo os só de gás (linha de gás e anular), e não os dutos
  monofásicos de gás da tese.
- **O RGO do reservatório é só o gás do poço**; o gás de lift soma-se-lhe no ponto de injeção.
- **Diâmetros da prática de projeto de linhas de gas lift**, com exemplos práticos (ver abaixo).

## O que torna o artigo publicável (e o que não chega)

A otimização de gas lift com modelos substitutos tem muita literatura; quase toda usa curvas de desempenho de
regime permanente (análise nodal). O que este trabalho tem de próprio:

1. **O viés do ótimo estacionário.** No mesmo ponto, a produção média no tempo de um ciclo de golfada severa ou de
   *heading* difere da do estacionário (instável) que um otimizador nodal usa. Medir quanto se perde por escolher
   um ponto instável é a mensagem principal.
2. **Um esquema híbrido para treinar o substituto.** O estacionário (Newton) e os seus autovalores com a matriz de
   massa (issue 048) custam segundos e dão a classe estável/instável e, na região estável, a produção; a dinâmica
   (minutos por corrida) só é necessária onde o estacionário é instável.
3. **Alocação ótima ao longo da vida** entre poços em estágios diferentes, com a estabilidade como restrição
   física (pico de líquido no separador, amplitude de pressão) e não como um peso arbitrário.

O que não chega sozinho: com duas decisões por poço, o ótimo de **um** poço acha-se por enumeração numa grelha
sobre o substituto. É a partilha do gás entre poços que dá um problema de otimização a sério.

## Rede e modelo

### Hipótese de rede (a confirmar)

Cada poço tem a sua linha e o seu riser até à plataforma (poços satélite, todos com a geometria do artigo 01: poço
vertical de 1500 m, linha de 2 km com 500 m a −2°, riser de 317 m). Os poços só se acoplam pelo coletor de gás de
lift e pelo compressor na plataforma. Consequência importante: **um só substituto, o de um poço**, serve para todos
os poços, e o problema com N poços é uma soma mais as restrições do compressor.

A alternativa, um coletor submarino de produção comum, acopla os poços hidraulicamente e em dinâmica (as golfadas
interagem). Fica fora deste artigo.

### O que se acrescenta ao modelo do artigo 01 (feito em 2026-10-18)

O modelo do artigo 01 fica congelado: padrão automático com 054 (`regime.local_bubble`), regra de fração baixa a
0,025, Newton com `armijo_tol` (055) e a estimativa de Milne (058), como o gerador do offshore já usa.

Rede de gas lift de cada poço, da plataforma para o fundo — opção `gas_lift=true` de `generators/offshore.py`,
que mantém o poço, a linha e o riser do artigo 01 bit a bit (o caso base regenerado é idêntico) e acrescenta o
ramo; casos `cases/paper01/gaslift/offshore_gl_P150.json` (com injeção) e `offshore_gl_P20.json` (sem injeção),
corridas em `runs/paper01/gaslift/`; issue 060:

1. vértice de pressão na plataforma (`gl_src`): **especificação da pressão do gás** P_gas, a única variável de
   controlo do ramo; com P_gas = 20 bar a retenção fica fechada e o caso reproduz o estado inicial do artigo 01
   (pressão de fundo 97 bar contra 95, vazões iguais);
2. **linha de gás de 4"** (0,1016 m) com o traçado da linha de produção ao contrário (patamar, riser, joelho em
   arco, subida a +2° e horizontal até à cabeça do poço, com metade das células por segmento: 19), com o modelo a dois fluidos
   (quase seca, α_G = 0,998): é o par 6" produção + 4" gas lift dos poços satélite da
   Petrobras — a especificação técnica dos poços satélite do Atapu lista o riser de serviço/gas lift de 4"
   flexível ao lado do de produção; a mesma pairing aparece nos sistemas de coleta do Roncador (P-52/P-54, linhas
   de produção emparelhadas com linhas de gas lift);
3. **anular 9 5/8" × 5 1/2"**: revestimento de produção 9 5/8" 47 lb/ft (ID 0,2205 m) e coluna de 5 1/2" (OD 0,1397
   m, ID 0,124 m ≈ o D_well = 0,127 m do artigo 01); o anular entra como duto de diâmetro de área equivalente
   0,171 m (área 0,0229 m², a que conta para a massa de gás armazenada e a inércia; o diâmetro hidráulico, 0,081 m,
   só mudaria o atrito, desprezável no gás a < 1 m/s), a dois fluidos;
4. **válvula de retenção** (`check_valve`, a lei regularizada da tese, 10.3.2) a 30 m do fundo (válvula operadora
   logo acima do packer), alimentada por um tanque de 1 m³ na base do anular, com CV = 1,5: q = 0,0075·CV·√(ρ ΔP)
   dá ~1 kg/s (≈ 100 000 Sm³/d, a ordem de grandeza das injeções por poço do Campos) a 60 bar de queda; **não é um
   orifício em escoamento crítico** (a hipótese anterior deste plano): a vazão segue √ΔP, mas com ΔP ≈ 110 bar as
   oscilações de ±10 bar do tubing mudam-na só ≈ ±5 %, o que dá o mesmo efeito prático;
5. o poço partido em `well_lower` (30 m, 1 célula) e `well` no vértice de injeção `inj`, que recebe a produção do
   reservatório e o gás da retenção; o RGO do reservatório continua a ser o gás do poço.

Manifold de gas lift a 150 bar (o caso «com injeção»): a coluna estática de gás leva a base do anular a 170 bar,
e a válvula injeta 1,15 kg/s com o tubing a 58 bar no estado inicial bloqueado (a pressão de fundo cai de 97 para
58 bar e a produção de líquido sobe de 4,1 para 5,7 kg/s). Com os 5 dutos o caso tem 323 estados (200 no artigo
01); `slow_exit = true` por causa dos eventos da retenção (033). O estado inicial (`--ic blocked`) marcha a coluna
de gás estática desde a plataforma e itera a injeção com a lei da retenção e a IPR.

O que fica da hipótese anterior: a lei de orifício em escoamento crítico não é necessária (ponto 4); o gerador
separado `generators/gas_lift.py` também não (a opção do gerador do offshore partilha a geometria e garante que o
caso do artigo 01 não muda). A verificação contra o caso 10.3.2 da tese (heading com a mesma topologia) fica por
fazer.

**Primeiras corridas (2026-10-18, 8000 s, JAX, issue 060).** Sem injeção (20 bar) o caso reproduz o do artigo 01
ao mesmo custo (9919 passos, 6 min), desde que a linha de gás e o anular arranquem quase secos (α_G = 0,998; com
2 % de líquido o escorrimento do filme custa 5×). Com injeção a 150 bar o gas lift **suprime a golfada severa**:
depois do sopro inicial a linha fica a 17,9 bar constante, o fundo do poço a 69 bar (94–108 sem injeção), a
injeção a 1,08 kg/s (≈ 110 000 Sm³/d) e a produção de líquido sobe de 3,94 para 5,29 kg/s. Dois avisos para o
desenho do artigo: (i) a fronteira de estabilidade em P_gas fica entre 20 e 150 bar e é o que o mapa linear tem de
localizar; (ii) o caso com injeção custa 5× (51 625 passos, 31 min) por chaveamento de padrão nas duas células
horizontais dos 10 m do topo do riser (estratificado ↔ intermitente ↔ anular a α_G ≈ 0,82 com gás rápido, 75
trocas por 1000 s, vazão no choke a oscilar entre 1,3 e 8,4 kg/s a cada 37 s), não na dinâmica do ramo de gás;
suavizar o chaveamento (`k_switch` 5) tira o custo mas muda a solução (o topo fica estratificado e o estado sai do
equilíbrio), logo é uma decisão de modelo a tomar antes dos lotes — por exemplo excluir o estratificado no troço
horizontal curto a jusante do riser (10 m = 66 D, sem comprimento de desenvolvimento), com issue própria.

Profundidade de injeção (análise de 2026-10-17 com a corrida de referência de 24 000 s): sem injeção, a pressão
no tubing a 1500 m vai de 92 a 110 bar ao longo do ciclo (gradiente médio 5,2–5,5 bar/100 m). Com Δp de 7 bar na
válvula e dimensionando pela pressão máxima do ciclo, injetar a 1500 m exige ≥ 120 bar no anular à cabeça do poço;
com 100 bar, só até 1250 m; com 80 bar, só até 850 m. Fica como decisão de projeto fixa (não é variável do
substituto).

## Variáveis

| tipo | variável | faixa proposta (a decidir) |
|---|---|---|
| estado do poço (vida) | pressão do reservatório P_res | 120–220 bar |
| estado do poço (vida) | índice de produtividade PI | 2–10 Sm³/d/bar |
| estado do poço (vida) | RGO | 30–200 Sm³/Sm³ |
| decisão por poço | pressão de gas lift do poço (ou abertura do choke de gás) | 60–150 bar |
| decisão por poço | abertura do choke de produção | 0,1–1,0 |

São cinco dimensões e não quatro, porque o IPR linear tem dois parâmetros (PI e P_res). Fica por decidir se a
decisão de gás de cada poço é a pressão a jusante do choke de gás ou a abertura desse choke, com a pressão de
descarga comum.

Saídas do substituto, por poço:

- classe: estável, golfada severa, ciclo de pressão, *heading* do anular;
- vazões médias no tempo: óleo produzido, gás produzido, gás injetado;
- pico da vazão de líquido na chegada e amplitude da pressão na chegada (para as restrições do separador);
- período do ciclo, quando há.

## O que o estacionário diz e o que não diz (caso de 2 km, 2026-10-17)

Espetro com a matriz de massa (048) ao longo da abertura do choke, por continuação a partir do estacionário de
0,10, com o modelo da campanha do artigo 01 (054, vazio baixo 0,025), comparado com as corridas dinâmicas de
16 000 s da mesma campanha (`runs/paper01/offshore_dynamics/`):

| abertura | espetro do estacionário | período linear 2π/Im λ | dinâmica: período | dinâmica: amplitude de P |
|---|---|---|---|---|
| 0,10–0,12 | estável | — | estável | 0 |
| 0,20 | Hopf; par rápido (68 s) à frente, par lento quase neutro (546 s) | 68 / 546 s | ~720 s | 0,8 bar |
| 0,30 | Hopf, par lento | 569 s | ~730 s | 2,8 bar |
| 0,40 | Hopf, par lento quase a colidir no eixo real | 1434 s | — | — |
| 0,60 | real (sem parte imaginária) | — | 1238 s | 8,9 bar |
| 1,00 | real | — | período 2 (1394 / 1732 / 3012 s) | 16,9 bar |

- Perto do início da instabilidade, o par lento dá o período com ~30 % de erro.
- O modo com maior parte real nem sempre é o que se vê: a 0,20 o par rápido cresce mais depressa, mas a dinâmica
  mostra o lento. O classificador do substituto não pode usar só o primeiro autovalor.
- Na golfada desenvolvida (a partir de ~0,5) o par colide no eixo real: não há frequência linear, e a amplitude
  nunca sai da análise linear.
- Estimativas mecanicistas a partir do estacionário funcionam bem na relaxação: P_max ≈ P_sep + ρ_L g H_riser =
  10 + 20,1 ≈ 30,1 bar (dinâmica 30,6–31,7); P_min ≈ a do riser cheio de gás (~15 bar; dinâmica 14,7); tempo de
  encher o riser ρ_L A H_riser / q_L ≈ 3740 kg / 3,8 kg/s ≈ 980 s, que com o sopro e a queda dá a ordem de
  1200–1500 s. O pico de vazão (32–45 kg/s) depende da dinâmica do sopro e não sai assim.

## Geração de dados (esquema híbrido)

Cada peça com o seu papel:

1. **Estacionário + autovalores em todos os pontos** (Sobol ou hipercubo latino, milhares de pontos, segundos
   cada; `tools/offshore_linear.py` do artigo 01 já faz a continuação e o espetro): a classe (estável, Hopf,
   real), a fronteira de estabilidade, as vazões na região estável e, perto da fronteira, uma primeira estimativa
   do período.
2. **Estimativas mecanicistas** (P_max, P_min, tempo de enchimento do riser, período de relaxação) calculadas do
   estacionário, como **entradas físicas do substituto**: com elas a rede só aprende a correção, e são precisas
   menos dinâmicas.
3. **Dinâmica só onde o estacionário é instável**, ~300–500 corridas, a partir do estacionário perturbado (040),
   com tf de 16 000 s ou a paragem adaptativa (041) e o orçamento de tempo (045): ensina a correção às estimativas
   e, sobretudo, o pico de vazão de líquido. A 2 km, 8000 s não chegam ao ciclo limite.
4. **Amostragem adaptativa junto da fronteira de estabilidade**, onde as saídas mudam bruscamente e onde um
   substituto treinado com poucos pontos alisa justamente o que interessa.

O cálculo direto das órbitas periódicas (*shooting* com derivação automática) fica para o artigo 03. Se estiver
pronto a tempo, dá ao artigo 02 os gradientes exatos das grandezas do ciclo em relação às decisões, para validar
o substituto e o ótimo junto da fronteira. Não serve para gerar as centenas de pontos: cada órbita custa ~1–3 h.

Custo, em ordem de grandeza: uma dinâmica de 24 000 s do caso base custa ~16 min numa thread; com a rede de gas lift
estimam-se +30–50 %. 300 dinâmicas ≈ 100–125 CPU-h, ~10 h com 12 processos.

## Substituto

- Rede com duas cabeças: classificação (classe do ponto) e regressão (vazões médias, pico, amplitude, período).
- Conjunto de redes, ou krigagem como referência, para ter a incerteza: serve para a amostragem adaptativa e para
  não deixar o otimizador explorar zonas onde o substituto não sabe.
- Validação num conjunto separado de dinâmicas; métricas por saída e matriz de confusão da classe.
- O ótimo final confirma-se sempre com o modelo completo (uma corrida dinâmica no ponto ótimo de cada estágio).

## Otimização

- N poços (4–8, a decidir) em fases diferentes da vida; decisões por poço: gás (pressão ou abertura) e choke.
- Objetivo: soma da receita do óleo − custo da compressão. A potência do compressor é função da vazão total de gás
  e da razão de pressões (politrópica), com a pressão de descarga comum igual à maior pressão pedida pelos poços.
- Restrições: capacidade do compressor (vazão total e potência), pico de vazão de líquido de cada poço abaixo da
  capacidade do separador, amplitude de pressão na chegada. Estas restrições substituem a «penalização da golfada»
  por grandezas com significado operacional.
- Solver: gradiente sobre o substituto diferenciável (JAX ou PyTorch), com SLSQP ou IPOPT e partidas múltiplas;
  enumeração como verificação nos casos de um poço.

## Ao longo da vida

- Trajetórias exógenas de P_res(t), PI(t) e RGO(t) por poço (curvas de declínio típicas, a definir), com os poços
  desfasados no tempo.
- Em cada estágio resolve-se a alocação; o resultado é a trajetória ótima da pressão de gas lift e da abertura do
  choke de cada poço ao longo dos anos.
- Comparação com a mesma alocação feita por um otimizador estacionário (sem restrição de estabilidade), avaliada
  depois com a dinâmica: é a medida do viés (contribuição 1).

## Limitações declaradas

- Sem água: escoamento bifásico óleo–gás.
- Trajetórias do reservatório exógenas (sem balanço material: produzir mais agora não antecipa a depleção).
- Uma só geometria; poços sem coletor de produção comum.
- Fluido com divisão entre fases congelada (como no artigo 01).
- Padrão automático ainda sem validação de laboratório: o ensaio de Fabre 2 com o padrão automático não chega ao
  primeiro ciclo (057). Convém resolver antes da submissão, porque o revisor vai perguntar.

## Dependências e etapas

0. Artigo 01 fechado e modelo congelado (as issues 054, 055, 057 e 058 estão feitas; falta a campanha a 2 km).
1. Rede de gas lift sobre o caso offshore: **feito** (opção `gas_lift` do gerador, retenção em vez de orifício
   crítico, issue 060); falta a verificação contra o caso 10.3.2 (heading) e a análise da profundidade de injeção
   acima com as corridas. Cada alteração ao modelo leva uma issue.
2. Mapa linear (estacionário + autovalores) nas cinco dimensões; fronteira de estabilidade com gas lift.
3. Lote de dinâmicas (`batches/paper02_*.json`) na região instável, com a amostragem adaptativa.
4. Substituto, validação e incerteza.
5. Otimização de N poços, trajetórias ao longo da vida e comparação com o otimizador estacionário.
6. Figuras (só a partir de corridas e de caches) e manuscrito.

## Decisões em aberto para o dono

- Faixas das cinco variáveis e as trajetórias de vida (declínio de P_res e PI, subida do RGO).
- Número de poços e capacidade do compressor.
- Decisão de gás por poço: pressão a jusante do choke de gás, ou abertura do choke de gás com descarga comum.
- Preço do óleo, custo da energia de compressão e capacidade do separador (para as restrições).
- Revista alvo e autores (no artigo 01: Jaime Neiva Miranda de Souza, Miradouro Digital, e André Luiz Hemerly
  Costa, UERJ/IQ, correspondente).

## Bibliografia a levantar

Sem chaves inventadas: cada referência entra em `refs.bib` (ou em `../_common/references.bib`) só depois de o DOI
estar verificado. Temas: revisões de otimização de gas lift (por exemplo, Rashid, Bailey e Couët, 2012, a
verificar); alocação de gás entre poços; modelos substitutos (redes neurais, krigagem) em produção de petróleo;
*heading* em poços com gas lift e válvulas em escoamento crítico; mitigação da golfada severa por gas lift na base do
riser (Jansen et al., 1996, já na bibliografia comum).
