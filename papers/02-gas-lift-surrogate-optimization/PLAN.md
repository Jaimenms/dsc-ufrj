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

### O que se acrescenta ao modelo do artigo 01

O modelo do artigo 01 fica congelado: padrão automático com 054 (`regime.local_bubble`), regra de fração baixa a
0,025, Newton com `armijo_tol` (055) e a estimativa de Milne (058), como o gerador do offshore já usa.

Rede de gas lift de cada poço, da plataforma para o fundo:

1. vértice de pressão na plataforma (descarga do compressor ou do coletor, a jusante do choke de gás do poço);
2. riser de gás (300 m) e linha de gás (2 km) até à cabeça do poço, monofásicos de gás;
3. anular do poço até à válvula operadora, monofásico de gás;
4. válvula operadora no tubing a ~1450 m abaixo da cabeça (logo acima do packer), modelada como orifício em
   **escoamento crítico** (razão de pressões < ~0,55): assim a vazão injetada não segue as oscilações da pressão
   no tubing, que no caso base chegam a ±9 bar com o ciclo da golfada;
5. o poço partido em dois dutos no ponto de injeção, com um vértice de junção que recebe o gás.

O que já existe no mfsim: dutos monofásicos de gás, válvula de retenção regularizada e a rede completa de gas lift
do caso 10.3.2 da tese (linha de gás, anular, válvula, tubing), com corridas de *heading* em
`notebooks_src/issues/nb_gas_lift_intermitencia.py`. A verificar: a lei de um orifício de gás em escoamento crítico
(a lei de válvula homogénea da 039 com gás puro), e um gerador novo (`generators/gas_lift.py`, que compõe o caso
offshore; o `generators/offshore.py` é do artigo 01).

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

## Geração de dados (esquema híbrido)

1. **Estacionário + autovalores em todos os pontos.** Sorteio de Sobol ou hipercubo latino, milhares de pontos,
   segundos cada (a ferramenta `tools/offshore_linear.py` do artigo 01 já faz a continuação e o espetro). Dá a
   classe linear e, na região estável, as vazões.
2. **Dinâmica só onde o estacionário é instável**, ~300–500 corridas, a partir do estacionário perturbado (040),
   com tf de 16 000 s ou a paragem adaptativa (041) e o orçamento de tempo (045). A 2 km, 8000 s não chegam ao
   ciclo limite: o caso base só assenta depois de ~8000 s (ciclo de período 2, ~1150/1800 s).
3. **Amostragem adaptativa junto da fronteira de estabilidade**, onde as saídas mudam bruscamente (Hopf) e onde um
   substituto treinado com poucos pontos alisa justamente o que interessa.

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
1. Rede de gas lift sobre o caso offshore: gerador, válvula em escoamento crítico, verificação contra o caso
   10.3.2 e contra a análise da profundidade de injeção acima. Cada alteração ao modelo leva uma issue.
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
