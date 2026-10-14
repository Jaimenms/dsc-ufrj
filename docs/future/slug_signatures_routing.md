# Catálogo de assinaturas de golfada e alocação de poços a separadores

*Tema futuro; não iniciado. Contexto em [README.md](README.md). Levantado em 2026-10-14 a partir
de uma ideia do dono; a pesquisa de literatura é dessa data.*

## A ideia

Usar o modelo para levantar, para cada poço (ou para cada linha e riser, quando vários poços
partilham um manifold), a **assinatura da intermitência severa em função da abertura do choke**:
forma da golfada, amplitude, período, volume de líquido da golfada e volume de gás do blowdown.
Numa plataforma com dois ou mais separadores de produção, escolher **que poços partilham cada
separador** para que a soma das golfadas seja a mais suave possível e, em conjunto, as aberturas
que maximizam a produção de óleo sem violar os limites de nível e de gás dos separadores.

A alocação poço–separador é um problema conhecido de otimização de produção (Codas e Camponogara
2012; Gunnerud e Foss 2010), mas sempre em regime permanente: vazões médias, capacidades, pressões.
A versão com a dinâmica das golfadas não foi publicada. A pesquisa também não encontrou nenhum
estudo de sincronização de golfadas entre poços que partilham um separador, só experiências com
risers duplos (NTNU 2015) e o efeito das golfadas sobre a separação a jusante (Pedersen, Aalborg).

## A armadilha física que decide o desenho

Dois poços só "combinam bem" se as fases dos ciclos forem controláveis ou acopladas. Se não
forem, as fases são aleatórias, porque o período deriva e é caótico (o repositório regista que o
período do caso de Jansen varia entre 58 e 90 s), e então a variância da soma é a soma das
variâncias: qualquer repartição que equilibre a variância total entre separadores é equivalente.
O que diferencia os subconjuntos é a **resposta do separador**, não a assinatura: o controlador de
nível filtra golfadas pequenas e frequentes mas não as grandes e lentas, e o compressor sofre com
o volume de gás do blowdown. O custo de um subconjunto é a excursão de nível e a variação da vazão
de gás de saída, dados o volume do vaso e a sintonia dos controladores, ou seja, as assinaturas
convoluídas com a resposta do separador. O ganho de misturar vem da estatística (cai com a raiz
do número de poços) e não ajuda um poço gigante ao lado de poços pequenos.

Há três caminhos para a ideia ficar forte, e o modelo serve aos três:

- **Acoplamento pelo separador comum.** Poços que partilham um separador são osciladores
  acoplados pela pressão de jusante (o controle de nível quase não afeta o escoamento a montante,
  mas a pressão do separador afeta o pico de pressão do riser). O modelo pode dizer se dois
  risers no mesmo vaso sincronizam em fase (pior caso) ou em oposição.
- **Faseamento ativo pelo choke.** Fechar brevemente o choke atrasa o blowdown, logo desloca a
  fase do ciclo. Um controlador de travamento de fase pode intercalar os blowdowns dos poços de
  um mesmo separador: a alocação vira escalonamento.
- **Problema conjunto.** Escolher a abertura por poço (que fixa produção e assinatura ao mesmo
  tempo) e o roteamento, para maximizar produção sujeita às restrições dinâmicas dos
  separadores: a extensão natural do MILP de Codas e Camponogara com custos dinâmicos por
  subconjunto pré-calculados no modelo.

Restrições de prática que encolhem o espaço de busca: headers de produção e de teste, classes de
pressão alta e baixa, capacidade de água, conectividade física; em manifolds submarinos a
assinatura é da linha, não do poço, e os poços dessa linha não se separam. Para poços existentes
a assinatura medida (pressão de fundo e de choke, como no conjunto 3W) vale mais do que a simulada;
o modelo entra para extrapolar a aberturas nunca testadas e para a simulação acoplada.

## O que o modelo tem

| peça | estado |
|---|---|
| assinatura de ciclo: blowdowns, período, picos, estatísticas por corrida | `blowdowns`, `cycle_signature`, `cycle_stats`, `scan_summary` em [analysis.py](../../mfsim/analysis.py) (goldens de nível C comparam por assinatura) |
| varredura de parâmetros por lote | `set` com overrides por caminho em `mfsim batch` ([runs.py](../../mfsim/runs.py)) |
| abertura do choke no tempo | `Choke.opening` é um `Schedule` (constante, linear por partes, rampa) em [schedules.py](../../mfsim/network/schedules.py); `Choke.pid` existe como campo mas não está implementado |
| vértice de volume | `TankVertex` com estados de fração de gás e pressão e volume ([graph.py](../../mfsim/network/graph.py)): tanque de mistura, sem nível nem saídas separadas |
| vários dutos no mesmo vértice | a rede é um grafo geral; `cases/severe_slug/reference_network.json` tem quatro dutos e três tanques |
| custo | o caso de campo de referência cicla em 40 s de CPU por 3000 s simulados ([why_no_severe_slug_in_thesis.md](../why_no_severe_slug_in_thesis.md)) |

## O que falta

- **Vértice separador** (item 6 das pendências do [README](../../README.md)): estados de nível de
  líquido e pressão de gás, saídas separadas de líquido e de gás com controladores PI de nível e
  de pressão, volume e geometria do vaso; sinais nos pontos de medição (nível, pressão, vazões de
  saída).
- **Choke como entrada de controle** (item 9): implementar o `pid` do choke com variável de
  processo medida na rede (pressão na base do riser, nível do separador), para a estabilização e
  para o faseamento.
- **Assinatura como estatística**: distribuição de período, volume de líquido da golfada e volume
  de gás do blowdown ao longo de vários ciclos (integrais de `q_out` em cada golfada), não só
  médias; forma média da golfada alinhada pelo blowdown.
- **Ferramenta de superposição**: dadas N assinaturas (séries de vazão de líquido e de gás à saída
  do choke) e um modelo de separador, calcular a excursão de nível e a variabilidade do gás para
  um subconjunto, com fases aleatórias por Monte Carlo.
- **Otimizador de roteamento**: MILP com custos pré-calculados por subconjunto (N pequeno) ou
  termos por pares (atribuição quadrática), com abertura por poço como variável.
- **Controlador de fase**: só se o degrau 4 mostrar que a fase é controlável.

## Degraus

1. **Catálogo por abertura.** Lote com `set` sobre a abertura do choke do caso de referência
   (os casos 10.3.x já têm choke; o de referência precisa de um): diagrama de bifurcação abertura
   contra pressão de base, período, volume da golfada e produção média. É o mesmo diagrama que o
   item 9 pede. 1 sessão.
2. **Separador e sensores.** Vértice separador com controle de nível e de pressão; o caso de
   referência a descarregar nele. 1 sessão.
3. **Superposição estatística.** N assinaturas do catálogo, fases aleatórias, custo por
   subconjunto; ranking das repartições para dois separadores. 1 sessão.
4. **Simulação acoplada.** Dois risers iguais no mesmo separador, depois comprimentos e vazões
   diferentes; medir a fase relativa ao longo do tempo e a correlação das pressões de base. Se
   sincronizarem em fase, a alocação estática morre e o faseamento ativo vira obrigatório; se não
   interagirem, a superposição do degrau 3 basta como custo. 1–2 sessões.
5. **Roteamento.** MILP com custos dinâmicos, comparado com o roteamento estacionário por vazões
   médias; produção total contra excursão de nível. 1 sessão.
6. **Faseamento ativo.** Resposta da fase a um pulso de fechamento do choke; controlador de
   travamento em oposição de fase para dois e três poços. 2 sessões, condicional ao degrau 4.

Total: 7–9 sessões. Cada degrau fecha com uma issue em `docs/issues/`, corridas por lote e
notebook só a ler, como os outros temas.

## Dados de validação

- **Catálogo contra abertura**: os ensaios de Jansen, Shoham e Taitel (1996) incluem choking e já
  estão em `cases/severe_slug/lab_jansen.json` ([lab_validation.md](../lab_validation.md)); a
  dependência período–abertura é o que se compara.
- **Assinaturas reais**: o conjunto 3W da Petrobras (público, 1 Hz) tem pressão de fundo, pressão
  na árvore, pressões a montante e a jusante do choke, aberturas dos chokes de produção e de gas
  lift e vazão de gas lift, com 106 instâncias de intermitência severa e 344 de instabilidade;
  não tem geometria, vazões nem nível de separador.
- **Interação entre risers**: experiências de risers duplos (NTNU 2015), qualitativas.
- **Efeito no separador**: Pedersen e colegas (Aalborg) sobre golfadas e separação a jusante.
- Não há dados públicos de vários poços num separador comum; a parte de roteamento valida-se por
  consistência interna (simulação acoplada contra superposição) e por comparação com o
  roteamento estacionário.

## Riscos

Período caótico: assinaturas estatísticas e comparação por assinatura, nunca ponto a ponto (lição
do repositório). Acoplamento forte pelo separador: invalida a alocação estática, mas é resultado
por si. Liberdade de roteamento pequena na prática. Modelo de separador simples: o lado do gás
(compressor) pode ser a restrição ativa, não o nível. Custo computacional aceitável: dez poços
por dez aberturas por 3000 s ficam em cerca de uma hora de CPU.

## Contexto: a linha de controle e otimização

Este tema é o terceiro de uma linha discutida na mesma sessão, que partilha os degraus 1 e 2:

- **Controle anti-slug com maximização de produção.** O choke manual estabiliza fechando e paga
  em pressão de fundo; a realimentação estabiliza o equilíbrio instável em malha aberta com a
  válvula mais aberta, logo com menos pressão e mais produção. A "estratégia ótima" é a maior
  abertura ainda controlável. Referências: Havre e Dalsmo 2002; Storkaas e Skogestad; Godhavn
  et al. 2005; Ogazi, Cao, Yeung e Lao 2010 (grandes aberturas para maximizar produção);
  Jahanshahi e Skogestad 2011–2017; Di Meglio et al. 2012; Campos et al. (Petrobras) 2015;
  de Oliveira, Jäschke e Skogestad 2015; Codas, Jahanshahi e Foss 2016 (duas camadas);
  Diehl et al. (Petrobras e UFRGS) 2018 (NMPC em gas lift, testado em poço real); aprendizagem
  por reforço 2022–2024. Todos usam modelos de 3 a 6 estados ajustados a cada poço e o OLGA como
  planta; nenhum usa uma planta aberta a dois fluidos.
- **Gêmeo digital calibrado em testes de poço.** O modelo a dois fluidos como gêmeo, calibrado
  por gradiente (backend JAX, Jacobiana exata) nas janelas em que o poço está alinhado ao
  separador de teste: o sinal fica isolado, o alinhamento excita a dinâmica, e as vazões fixam a
  IPR e a válvula de gas lift. Sensores a usar: pressão de fundo e de árvore (o nível do
  separador é o sinal mais fraco). Variável de gas lift por poço: abertura do choke de injeção;
  a pressão do header é restrição partilhada. O modelo de duto não entra na malha rápida: serve
  de gêmeo, estimador e camada econômica, linearizado no ponto corrente para a camada
  estabilizadora. Contribuição defensável: identificabilidade dos fechamentos a partir dos sinais
  que as plataformas têm.

## Referências

- Codas, A.; Camponogara, E. *Mixed-integer linear optimization for optimal lift-gas allocation
  with well-separator routing.* EJOR 217 (2012). <https://ideas.repec.org/a/eee/ejores/v217y2012i1p222-231.html>
- Gunnerud, V.; Foss, B. *Oil production optimization: a piecewise linear model, solved with two
  decomposition strategies.* Comp. Chem. Eng. (2010).
- Codas, A.; Jahanshahi, E.; Foss, B. *A two-layer structure for stabilization and optimization of
  an oil gathering network.* IFAC 2016. <https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2429282>
- Campos, M. C. M. M. et al. *Advanced anti-slug control for offshore production plants.* IFAC
  2015. <https://skoge.folk.ntnu.no/prost/proceedings/offshore-control-2015/papers/0023.pdf>
- de Oliveira, V.; Jäschke, J.; Skogestad, S. *An autonomous approach for driving systems towards
  their limit: an intelligent adaptive anti-slug control system for production maximization.*
  IFAC 2015. <https://jaschke.folk.ntnu.no/preprints/2015/deOliveiraOOGP/oliveira-paper.pdf>
- Pedersen, S.; Durdevic, P.; Yang, Z. *Challenges in slug modeling and control for offshore oil
  and gas productions: a review study.* IJMF 2017. <https://vbn.aau.dk/ws/files/265542529/2017_energies_10_02014.pdf>
- Pedersen, S. *Plant-wide anti-slug control for offshore oil and gas processes.* Tese, Aalborg
  2016. <https://vbn.aau.dk/en/publications/plant-wide-anti-slug-control-for-offshore-oil-and-gas-processes>
- Di Meglio, F. et al. *Stabilization of slugging in oil production facilities with or without
  upstream pressure sensors.* J. Process Control 2012. <https://hal-mines-paristech.archives-ouvertes.fr/hal-00733322>
- Experiências de intermitência severa com vários risers, NTNU 2015.
  <https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2350132>
- Conjunto 3W (Petrobras): <https://github.com/ricardovvargas/3w_dataset>; artigo de dados 2.0.0:
  <https://arxiv.org/pdf/2507.01048>
- Controle de nível do separador e escoamento a montante, Petroleum Science 2014.
  <https://cup.edu.cn/petroleumscience/wqwz/2014/deq2014/2rr/6069006.htm>
