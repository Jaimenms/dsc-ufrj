# Artigo 02 — plano: gas lift ao longo da vida com restrição de estabilidade dinâmica

*Documento de trabalho do artigo (português, como o resto da documentação; o manuscrito é em inglês).
Escrito em 2026-10-17 a partir do pedido do dono e da discussão do escopo; revisto em 2026-10-18 depois de a
geometria com gas lift ter ficado pronta no artigo 01 (§8.8, issue 060) e com a estratégia de geração de dados
pedida pelo dono (sorteio das entradas, marcadores em pontos característicos em vez de perfis, horizonte de
simulação com captura de ciclos). Atualizar à medida que as decisões e as corridas chegam.*

## O pedido, em uma frase

Pegar no modelo dinâmico do artigo 01 com a rede de gas lift (linha de gás da plataforma, anular e válvula de
retenção, controlada pela pressão do gás na plataforma), gerar algumas centenas de simulações dinâmicas sorteando
o estado do poço (IPR, RGO), a pressão de gas lift e a abertura do choke, registar de cada uma só os marcadores
(média, mínimo, máximo, frequências) dos estados em pontos característicos da rede, treinar com eles um modelo
substituto (redes neurais), e otimizar a operação. A função objetivo junta a receita do óleo, o custo da
compressão do gás e restrições de estabilidade com significado operacional. Por fim, seguir o IPR e o RGO ao
longo da vida dos poços e mostrar a pressão de gas lift e a abertura do choke ótimas em cada fase.

## Decisões do dono

**2026-10-17**

- **Vários poços, todos com a mesma geometria** (a do artigo 01), a partilhar o gás de lift e, portanto, o
  compressor. O acoplamento que dá sentido à otimização é a alocação de um gás limitado entre poços em fases
  diferentes da vida.
- **Água como limitação declarada.** O modelo é bifásico (óleo e gás); a fração de água, que costuma dominar a
  economia do gas lift no fim da vida, fica fora e é discutida como limitação.
- **Gas lift modelado desde a pressão na plataforma**, no início da linha de gas lift.

**2026-10-18**

- **A geometria é a do artigo 01 com o ramo de gas lift** (§8.1 e §8.8 do artigo 01): poço de 1500 m, linha de
  2 km com 500 m a −2°, riser de 317 m, mais a linha de gás de 4" pelo traçado da linha de produção, o anular
  9 5/8" × 5 1/2" e a válvula de retenção a 30 m do fundo. Pressão do gás na plataforma como único controlo do
  ramo (desligar = pressão baixa, retenção fechada); modelo a dois fluidos em todos os dutos; o RGO do
  reservatório é só o gás do poço.
- **Geração de dados por sorteio das entradas** (condições do poço e decisões), não por grelha.
- **Saídas registadas em pontos característicos da rede, não perfis**: início do poço, topo do poço, início do
  riser, topo do riser e as duas extremidades da linha de gas lift.
- **Horizonte de simulação determinado por uma metodologia**, com captura de pelo menos 2 ciclos (número a
  definir) depois de o regime estar estabelecido.
- **Sem séries temporais no conjunto de dados**: só marcadores por sinal — valor médio, mínimo, máximo, frequência
  mínima e frequência máxima.
- **A IPR entra como duas variáveis, P_res e PI** (confirmado pelo dono depois de discutidas as alternativas de
  fixar P_res ou usar o potencial em fluxo aberto): é P_res contra o peso das colunas que decide se o poço morre e
  é a pressão de fundo que decide se a retenção abre.
- **Arranque quente em cadeias** (proposta do dono: perturbações pequenas das entradas com o estado final de uma
  corrida como estado inicial da seguinte; forma acordada: cadeias curtas sobre um desenho de Sobol, ver «Geração
  de dados»).

**2026-10-18, à noite (simplificação pela velocidade)**

- **O poço é o do caso base** (P_res 200 bar, PI 5, RGO 50) e **só variam as duas decisões**, P_gas (20–160 bar) e a
  abertura do choke (0,1–1,0): o conjunto de dados passa a ser um mapa em 2 dimensões. Os conjuntos em 5
  dimensões (`runs/paper02/cases/`, Sobol; `runs/paper02/grid/`, malha por poço) foram parados depois do ponto
  em curso, com os pontos feitos guardados; a 10 min por ponto não chegavam a cobrir 5 dimensões.
- **Rondas de 25 pontos de Sobol em 2 dimensões, caminho mais curto a partir do caso base** (vizinho mais
  próximo com 2-opt para 25 pontos), arranque quente do estado final do caso base no ponto mais próximo dele;
  conjunto `runs/paper02/base2d/` (`design --fix-well`, semente 1), a correr continuamente com **três instâncias
  em paralelo** sobre a mesma raiz (trincos por cadeia, rondas novas com 3 cadeias), sem duplicar pontos.

## O que torna o artigo publicável (e o que não chega)

A otimização de gas lift com modelos substitutos tem muita literatura; quase toda usa curvas de desempenho de
regime permanente (análise nodal). O que este trabalho tem de próprio:

1. **O viés do ótimo estacionário.** No mesmo ponto, a produção média no tempo de um ciclo de golfada severa ou de
   *heading* difere da do estacionário (instável) que um otimizador nodal usa. Medir quanto se perde por escolher
   um ponto instável é a mensagem principal. O artigo 01 já mostra o primeiro caso: a 100–120 bar de gás o ciclo
   sobrevive onde o critério de Bøe o daria por extinto, com sopros de 26–31 kg/s e produção +10 a +17 %.
2. **Um esquema híbrido para treinar o substituto.** O estacionário (Newton) e os seus autovalores com a matriz de
   massa (issue 048) custam segundos e dão a classe estável/instável e, na região estável, a produção; a dinâmica
   (minutos por corrida) só é necessária onde o estacionário é instável.
3. **Alocação ótima ao longo da vida** entre poços em estágios diferentes, com a estabilidade como restrição
   física (pico de líquido no separador, amplitude de pressão) e não como um peso arbitrário.

O que não chega sozinho: com duas decisões por poço, o ótimo de **um** poço acha-se por enumeração numa grelha
sobre o substituto. É a partilha do gás entre poços que dá um problema de otimização a sério.

## Rede e modelo (fixos: os do artigo 01)

Cada poço tem a sua linha e o seu riser até à plataforma (poços satélite); os poços só se acoplam pelo coletor de
gás de lift e pelo compressor. Consequência: **um só substituto, o de um poço**, serve para todos, e o problema
com N poços é uma soma mais as restrições do compressor. Um coletor submarino de produção comum (acoplamento
hidráulico e dinâmico entre poços) fica fora deste artigo.

O modelo e a geometria são os do artigo 01 e **não mudam**: padrão automático com 054, regra de fração baixa a
0,025, Newton com `armijo_tol` (055), estimativa de Milne (058), e o ramo de gas lift da §8.8 (opção
`gas_lift=true` de `generators/offshore.py`, issue 060): linha de gás de 4" (0,1016 m) com o traçado da linha de
produção e metade das células (19), anular de diâmetro equivalente 0,171 m (10 células) com tanque de 1 m³ na
base, retenção com CV = 1,5 e sem pressão de abertura a 30 m do fundo (poço partido em `well_lower` e `well` no
vértice `inj`), tudo a dois fluidos com o ramo de gás a arrancar quase seco (α_G = 0,998). Casos
`cases/paper01/gaslift/offshore_gl_P{20,100,...,150}.json`, lote `batches/paper01_gaslift.json`, corridas em
`runs/paper01/gaslift/`; 323 estados.

O que o artigo 01 já mostrou com esta geometria (choke aberto, P_res 200 bar, PI 5, RGO 50, 8000 s):

| P_gas (bar) | regime | intervalo entre sopros | fundo (bar) | injeção (kg/s) | produção | passos |
|---|---|---|---|---|---|---|
| 20 (retenção fechada) | golfada severa, sopros alternados | 1315–1625 s | 94–108 | 0 | 3,94 kg/s | 9,9 k |
| 100 / 110 / 120 | golfada severa periódica, riser sem encher (holdup máx. 0,73–0,79), injeção contínua modulada ±20–25 % | 1100 / 1220 / 1410 s | 77–103 | 0,34–0,79 | +10 / +14 / +17 % | 25–46 k |
| 130 / 140 / 150 | estacionário | — | 69–75 | 0,89 / 0,99 / 1,08 | +29 / +32 / +34 % | 26 / 117 / 52 k |

O gas lift intermitente (retenção a fechar durante parte do ciclo) não apareceu: o anular fica sempre acima da
pressão do tubing na injeção. Aviso de custo: as duas células horizontais dos 10 m do topo do riser chaveiam de
padrão com gás rápido (estratificado ↔ intermitente ↔ anular) e multiplicam os passos por 2,5 a 11; suavizar o
chaveamento muda a resposta. **Antes dos lotes do artigo 02 é preciso decidir o fechamento desse troço** (ver
decisões em aberto); sem isso, metade das corridas custa 30 a 60 min em vez de 6.

Verificações pendentes do modelo: caso 10.3.2 da tese (heading com a mesma topologia) e o ensaio de Fabre 2 com o
padrão automático (057).

## Entradas (o que se sorteia)

| tipo | variável | símbolo | faixa proposta (a decidir) |
|---|---|---|---|
| estado do poço (vida) | pressão do reservatório | P_res | 120–220 bar |
| estado do poço (vida) | índice de produtividade | PI | 2–10 Sm³/d/bar |
| estado do poço (vida) | razão gás–óleo | RGO | 30–200 Sm³/Sm³ |
| decisão por poço | pressão do gás na plataforma | P_gas | 20–160 bar |
| decisão por poço | abertura do choke de produção | a | 0,1–1,0 |

Cinco dimensões: o IPR linear tem dois parâmetros. A decisão de gás é a pressão P_gas (decisão do dono de
2026-10-18); com descarga comum do compressor, a pressão de cada poço é a de descarga menos a queda no seu choke
de gás, e o choke de gás fica implícito.

Dois cuidados no sorteio:

- **A retenção fecha abaixo de um P_gas que depende do poço** (a base do anular, ≈ 1,13 P_gas, tem de superar a
  pressão do tubing na injeção, 60–110 bar conforme o poço). Abaixo disso a decisão é «sem gas lift» e a saída
  não depende de P_gas: o substituto tem uma quina. Fica no conjunto de dados (é física), mas a característica
  «ΔP da válvula no estacionário» (abaixo) diz ao substituto onde está a quina.
- **Pontos impossíveis**: poços que morrem (coluna de líquido do riser e do poço acima de P_res) ou que não
  arrancam; a corrida classifica-os (classe «morto») e o substituto aprende a fronteira.

## Pontos característicos e marcadores (o que se regista)

Em vez de perfis ao longo da rede, oito pontos (os seis pedidos pelo dono mais dois que o gas lift exige), com
quatro variáveis em cada um: pressão P, fração de gás α_G, vazão mássica de líquido q_L e de gás q_G.

| ponto | onde no caso | o que mede |
|---|---|---|
| fundo do poço | 1.ª célula de `well_lower` | pressão de fundo (IPR), vazão do reservatório |
| injeção | 1.ª célula de `well` | pressão do tubing na válvula; q_G aqui − q_G do fundo = **gás injetado** |
| topo do poço | última célula de `well` | pressão na cabeça (entrada da linha, o P_in do artigo 01) |
| início do riser (ponto baixo) | célula do joelho da linha | pressão no ponto baixo, tampão |
| topo do riser | última célula do riser (antes do choke) | vazões no choke (pico de líquido, gás para o separador) |
| linha de gás, plataforma | 1.ª célula de `gas_line` | vazão de gás que o compressor fornece (P é a entrada P_gas) |
| linha de gás, cabeça do poço | última célula de `gas_line` | pressão à chegada ao anular |
| base do anular | tanque `an_bot` (estados HG, PT) | pressão a montante da válvula |

Marcadores por sinal (32 sinais), calculados **na janela de captura** (abaixo): **média, mínimo, máximo,
frequência mínima e frequência máxima**. As frequências vêm dos intervalos entre máximos consecutivos do sinal
suavizado (média móvel de 20 s, como no artigo 01): f_min = 1/(maior intervalo), f_max = 1/(menor intervalo);
num ciclo periódico as duas coincidem, num ciclo com sopros alternados diferem (é a assinatura da duplicação de
período), num estacionário ambas são 0 (nenhum máximo acima da tolerância de amplitude). Dos marcadores derivam-se
sem custo as grandezas do substituto: produção média (média de q_L no topo do riser), gás injetado médio, pico de
líquido no separador (máximo de q_L no topo do riser), amplitude de pressão (máx. − mín. de P na cabeça),
período (1/f), razão pico/média, e a classe (estável se a amplitude de P na cabeça é menor que 0,1 bar ou 1 % da
média; golfada severa se o pico de líquido é ≥ 2 vezes a média; ciclo de pressão caso contrário — o critério de
`mfsim.analysis.cycle_class`, já usado no artigo 01).

Metadados por corrida, para filtrar e pesar: horizonte alcançado, motivo da paragem (estável, ciclo capturado,
tempo máximo, orçamento, falha), número de ciclos capturados, passos, CPU, e o estado inicial usado.

As séries completas não se guardam (só com `--keep-runs`, para depuração); o conjunto de dados é uma tabela (Parquet, uma linha por corrida) com as 5
entradas, os metadados, as 32 × 5 = 160 colunas de marcadores (mais o gás injetado) e, por junção pelo identificador do ponto, as
características físicas abaixo. Formato: CSV com dicionário em JSON (`tools/paper02_dataset.py markers`); Parquet
só se o treino o pedir. As trajetórias não se guardam (decisão do dono, 2026-10-18): os marcadores calculam-se no
fim de cada corrida e ficam em `point.json`; mudar a definição de um marcador implica voltar a correr.

## Horizonte de simulação e captura de ciclos (metodologia)

O simulador já tem a paragem adaptativa (041, `time.settle`): integra em janelas, calcula a assinatura de cada
janela (amplitude, média, cruzamentos, período) de um sinal e pára quando a amplitude é pequena (estável) ou
quando duas janelas seguidas têm a mesma amplitude e o mesmo período (ciclo). Falta-lhe o que o artigo 02
precisa, e isso é uma issue (feature com bandeira, 061):

1. **Vários sinais no critério** (P na cabeça do poço e q_L no topo do riser no mínimo): o ciclo só conta como
   estabelecido quando todos repetem.
2. **Ciclos com sopros alternados** (período 2 ou intervalos que não se repetem, como o choke aberto do artigo
   01): comparar janelas de comprimento igual a um múltiplo do período estimado, com tolerância na média, no mínimo
   e no máximo em vez de no período; ou aceitar «estabelecido» quando a soma de intervalos consecutivos se repete
   (o período duplo conservou-se a ±4,5 % onde os intervalos variavam 25 %).
3. **Captura**: depois de «estabelecido», continuar por **N_c ciclos** (proposta: N_c = 3 — um para confirmar que
   o regime se mantém, dois para medir; mínimo 2, decisão do dono) e calcular os marcadores só nessa janela; num
   estacionário a janela de captura é uma janela fixa (600 s) depois da amplitude cair abaixo da tolerância.
4. **Limites**: t_min = 4000 s (o transiente da partida bloqueada tem 3–5 ciclos), tf_max = 24 000 s, orçamento
   de relógio por corrida (045, p. ex. 40 min); uma corrida que chega a tf_max sem se estabelecer entra no conjunto
   com o motivo «tf» e os marcadores dos últimos N_c ciclos, e com peso menor no treino.
5. **Estado inicial**: partida bloqueada do gerador (`--ic blocked`), como no artigo 01; é robusta e chega ao
   primeiro sopro em 300 s. A partida do estacionário (040) só serve onde o Newton converge e pode demorar mais a
   sair do equilíbrio instável. Como a Hopf é supercrítica no artigo 01, o atrator não deve depender da partida;
   verifica-se numa amostra (mesmo ponto com as duas partidas).

O período para dimensionar as janelas vem, por ordem de preferência, do par de troca linha–riser do espetro do
estacionário (artigo 01: 500–1000 s, a 30 % do ciclo), da estimativa mecanicista (tempo de encher o riser,
ρ_L A H/q_L) ou, na falta de ambos, de 1000 s.

## Características físicas (features) para o substituto

Além das cinco entradas, características calculadas **sem integrar no tempo**, para o substituto aprender só a
correção (esquema híbrido, contribuição 2):

- do **estacionário** (Newton, `tools/offshore_linear.py` estendido ao ramo de gas lift): pressão de fundo,
  pressão na cabeça, gás injetado, holdup do riser, produção estacionária, ΔP na válvula (positivo = aberta; a
  quina do gas lift);
- do **espetro com a matriz de massa** (048) no estacionário: parte real e frequência do par de troca
  linha–riser e do par da coluna do riser, e o modo real (classe linear: estável, Hopf, instável real);
- **estimativas mecanicistas**: P_max ≈ P_sep + ρ_L g H_riser, P_min ≈ riser cheio de gás, tempo de enchimento do
  riser ρ_L A H/q_L, razão de Bøe U_SL/(P U_SG/(ρ_L g α_G L)) com o gás injetado contado, razão gás de lift/gás
  livre do poço, fração mássica de gás livre w_G.

Onde o Newton não converge (poço morto, ponto fora da bacia) as características ficam em falta e o substituto
recebe um indicador; é informação.

## Geração de dados (estratégia)

1. **Desenho de experiências**: sequência de Sobol embaralhada (quase-aleatória: cobre o hipercubo de 5 dimensões
   sem buracos nem aglomerados, mantendo o sorteio) sobre as faixas da tabela; conjunto de teste com uma
   sequência independente (ou sorteio uniforme), nunca uma partição da de treino. Proposta: 400 de treino, 100 de
   teste, mais um lote adaptativo.
1a. **Cadeias com arranque quente** (ideia do dono, 2026-10-18: percorrer o espaço com perturbações pequenas e
   o estado final de cada corrida como estado inicial da seguinte). O arranque quente poupa o transiente da
   partida bloqueada (3 a 5 ciclos, 4000–8000 s no artigo 01): com uma perturbação pequena das entradas o sistema
   já está perto do atrator novo e assenta em 1–2 ciclos; a malha não muda com nenhuma das cinco entradas, por isso
   o estado transfere-se diretamente (`state_to_case_ic`); e é o que um poço faz ao longo da vida. Um passeio
   browniano puro não serve para cobrir o hipercubo: é difusivo, e cobrir 5 dimensões com passos de 10 % das
   faixas pediria ~10⁵ pontos. A forma acordada junta as duas coisas:
   - o conjunto de treino é o desenho de Sobol (cobertura garantida), agrupado em 20–25 **cadeias de vizinhos**
     (k-means nas entradas normalizadas); **dentro de cada cadeia a sequência é a do dono (2026-10-18): o primeiro
     ponto é o mais perto do caso base do artigo 01 (choke aberto, 200 bar, PI 5, RGO 50, retenção fechada) e a
     cadeia arranca a quente do estado final desse caso (`runs/paper01/gaslift/gl_P20.npz`, 8000 s); dos outros
     pontos segue-se o caminho mais curto** (caixeiro
     viajante sem regresso: exato por enumeração até 10 pontos, vizinho mais próximo com 2-opt acima;
     `shortest_path_from` da ferramenta); cada cadeia corre em sequência num processo — o primeiro ponto da
     partida bloqueada, os seguintes com arranque quente a partir do anterior — e as cadeias correm em paralelo;
     as distâncias, o centro e os grupos medem-se com cada entrada reescalonada a [−1, +1] (pedido do dono); o
     ensaio desta sequência com 10 pontos sorteados numa só cadeia está em `runs/paper02/seq10/` (semente 11);
   - **malha das decisões em serpentina (pedido do dono, 2026-10-18, `--grid-decisions 6x6`):** as dimensões
     separam-se pelo papel — o estado do poço (P_res, PI, RGO) vem de Sobol em 3 dimensões, um por ronda, e as
     decisões (P_gas × abertura) formam uma malha regular 6 × 6 varrida em serpentina (boustrophedon) a partir do
     canto mais perto do caso base (20 bar, choke aberto), com P_gas monótono (a linha de gás nunca se esvazia
     para a plataforma) e a abertura de ida e volta: cada passo muda uma só decisão de um só nível, é o caminho de
     arranque quente mais curto possível, e o varrimento de ida e volta mede a histerese de graça. Cada ronda é o
     mapa completo das 36 decisões de um poço, o que a otimização ao longo da vida consome diretamente. Custo: 36
     corridas por poço, 35 delas a quente. Conjunto de treino em `runs/paper02/grid/` (semente 1, arrancou em
     2026-10-18 com o poço {136 bar, PI 6,7, RGO 133}); o conjunto de Sobol em 5 dimensões (`runs/paper02/cases/`)
     continua a correr em paralelo como conjunto independente (teste);
   - **malha das decisões em serpentina (pedido do dono, 2026-10-18, `--grid-decisions 6x6`):** as dimensões
     separam-se pelo papel — o estado do poço (P_res, PI, RGO) vem de Sobol em 3 dimensões, um por ronda, e as
     decisões (P_gas × abertura) formam uma malha regular 6 × 6 varrida em serpentina (boustrophedon) a partir do
     canto mais perto do caso base (20 bar, choke aberto), com P_gas monótono (a linha de gás nunca se esvazia
     para a plataforma) e a abertura de ida e volta: cada passo muda uma só decisão de um só nível, é o caminho de
     arranque quente mais curto possível, e o varrimento de ida e volta mede a histerese de graça. Cada ronda é o
     mapa completo das 36 decisões de um poço, o que a otimização ao longo da vida consome diretamente. Custo: 36
     corridas por poço, 35 delas a quente. Conjunto de treino em `runs/paper02/grid/` (semente 1, arrancou em
     2026-10-18 com o poço {136 bar, PI 6,7, RGO 133}); o conjunto de Sobol em 5 dimensões (`runs/paper02/cases/`)
     continua a correr em paralelo como conjunto independente (teste);
   - **rondas** (pedido do dono, 2026-10-18): ao acabar os pontos de uma ronda, `run --rounds` sorteia outros
     tantos — os pontos seguintes da mesma sequência de Sobol, para que as rondas juntas continuem a ser um só
     desenho — e recomeça do caso base no ponto da ronda nova mais próximo dele; `--rounds -1`
     continua até aparecer o ficheiro `STOP`. **Omissão: 25 pontos por ronda; organização pedida pelo dono:**
     `runs/paper02/cases/YYYYMMDDXX/<id>/` — uma pasta por ronda (data de abertura e contador de dois dígitos por
     dia) e, dentro, uma pasta por ponto com o id = 6 caracteres do SHA-1 do tuplo das entradas (o mesmo tuplo dá
     sempre o mesmo id; colisões levam mais caracteres). **Só se guarda o que o treino usa** (pedido do dono):
     `point.json` (entradas, cadeia, metadados da corrida e os marcadores) e `final_state.npz` (o estado final,
     para o arranque quente do ponto seguinte, 3 kB); a trajetória, o caso e o estado inicial não se guardam —
     regeneram-se das entradas e da configuração (`run --keep-runs` guarda-os para depuração, `compact` converte
     pontos antigos). A raiz `rounds.json` guarda a semente, as faixas, a configuração, a lista de rondas e os
     ids; `markers` escreve a tabela de cada ronda e a de todas na raiz a partir dos `point.json`. O conjunto contínuo arrancou em 2026-10-18 (`runs/paper02/cases/2026101801/`, semente 1); a
     pasta anterior `runs/paper02/rounds/` (organização antiga, sem corridas terminadas) ficou por apagar;
   - variante a avaliar no piloto: estado do poço (P_res, PI, RGO) do Sobol e, para cada um, um passeio curto de
     ~20 pontos nas duas decisões (P_gas, abertura) com arranque quente — um passeio em 2 dimensões cobre bem e é
     literalmente operar o poço;
   - **o conjunto de teste é sempre independente**: outra sequência de Sobol, todos os pontos da partida
     bloqueada, para a correlação dentro das cadeias não entrar na avaliação;
   - **histerese**: com arranque quente cada corrida segue o ramo em que está (fronteira de estabilidade, ciclos
     alternados, poço morto, que não volta com uma perturbação pequena). Numa amostra de ~30 pontos corre-se o
     mesmo ponto das duas partidas; onde os marcadores diferem há coexistência de atratores, que é um resultado
     a reportar (o artigo 01 não a viu na Hopf supercrítica) e uma razão para o ponto entrar no conjunto com as
     duas respostas e a partida como metadado;
   - ferramenta: `tools/paper02_dataset.py run` encadeia as corridas (escreve o caso seguinte com o estado final
     do anterior), e a paragem adaptativa com captura de ciclos é a 061 (feita); poupança estimada de metade a
     dois terços do tempo por corrida, a medir no piloto.
   - fumo de ponta a ponta (2026-10-18, 3 pontos, 1500 s): o desenho, o arranque quente, a captura e os
     marcadores funcionam (`dataset.csv` com 186 colunas). Dois avisos: o ponto com gás rápido no riser (155 bar,
     PI 9, choke 0,93) custou 19 970 passos em 1500 s — o fechamento do topo do riser outra vez; e uma descida
     grande de P_gas entre pontos vizinhos (155 → 36 bar) esvazia a linha de gás e o anular para a plataforma
     (−3 kg/s em média durante a janela), um transiente físico mas caro. Medido com os mesmos três pontos da
     partida bloqueada (1500 s): o vizinho próximo (117 → 155 bar) saiu melhor quente (583 s contra 1534 s e o
     orçamento esgotado), o salto grande (155 → 36 bar) saiu 4,7× pior quente (32 082 contra 6808 passos): o
     arranque quente paga quando o passo é pequeno, sobretudo em P_gas — daí a sequência de distância mínima.
2. **Estacionário + espetro em todos os pontos** (milhares, segundos cada): classe linear, características físicas
   e a fronteira de estabilidade nas cinco dimensões.
3. **Dinâmica em todos os pontos sorteados** (não só nos instáveis, ao contrário do plano anterior): o custo de um
   estacionário dinâmico é pequeno (a paragem adaptativa pára em ~3000–4000 s) e o conjunto precisa de exemplos
   estáveis com os seus marcadores para a regressão ser contínua através da fronteira.
4. **Amostragem adaptativa**: segundo lote onde a incerteza do substituto (conjunto de redes) é maior e junto da
   fronteira de estabilidade, onde as saídas mudam bruscamente.
5. **Ferramenta** `tools/paper02_dataset.py` (feita em 2026-10-18, issue 061): `design` (Sobol → cadeias →
   `design.json`), `run` (cadeias em processos paralelos, casos gerados por `generators/offshore.py` com
   `gas_lift=true`, `time.settle` e `max_cpu` em cada caso, retomável, `STOP` para parar a quente), `markers`
   (corridas → `dataset.csv` + `dataset.json`), `status`; saída em `runs/paper02/<conjunto>/`; nada no guião das
   figuras simula.

Custo, em ordem de grandeza: com o chaveamento do topo do riser resolvido, uma corrida de 8000–12 000 s custa
6–10 min numa thread; 500 corridas ≈ 60–80 CPU-h, 6–8 h com 12 processos. Sem o resolver, 3 a 10× mais nos pontos
com gás rápido no riser.

## Substituto

- Rede com duas cabeças: classificação (classe do ponto: estável, ciclo de pressão, golfada severa, morto) e
  regressão (marcadores escolhidos: produção média, gás injetado médio, pico de líquido, amplitude de pressão,
  f_min e f_max, pressão de fundo média).
- Entradas: as 5 variáveis mais as características físicas; saídas normalizadas pelas estimativas mecanicistas
  (o substituto aprende razões, não valores).
- Conjunto de redes, ou krigagem como referência, para ter a incerteza: serve para a amostragem adaptativa e para
  não deixar o otimizador explorar zonas onde o substituto não sabe.
- Validação no conjunto de teste; métricas por saída e matriz de confusão da classe.
- O ótimo final confirma-se sempre com o modelo completo (uma corrida dinâmica no ponto ótimo de cada estágio).

## Substituto: primeira comparação NN × PINN (2026-10-19, `tools/paper02_surrogate.py`)

Abordagem do artigo II do projeto composicional (a dissertação do Gustavo, `research-factory/01-multiphase-
compositional-flow/papers/02-surrogate-models`): MLP em JAX (2 camadas × 32 tanh, Adam 0,01 em lote completo num
`lax.scan`, entradas e saídas padronizadas, divisão 70/30) e PINN = mesma rede com a perda dos dados mais λ vezes
o resíduo da física em pontos de colocação (hipercubo latino no retângulo, sem simular). **Pedido do dono:** a
física vem das médias dos estados — os balanços médios no tempo são exatos porque são lineares nas médias:
R1 q̄_L = J_L (P_res − P̄_fundo); R2 q̄_G = J_G (P_res − P̄_fundo) + q̄_inj; R3 P̄_anular = κ P_gas (coluna estática,
κ = 1,138); R4 lei da retenção nas pressões médias (aproximada, porta suave). A rede prevê as seis saídas pedidas
(q̄_L e q̄_G no topo, q̄ de gas lift, e os três *spans* máx − mín da janela de captura, a assinatura da golfada) mais
três pressões médias auxiliares (fundo, injeção, base do anular) que são o que a física liga. Dados: 150 dos 156
pontos de `base2d` (os 6 de orçamento esgotado ficam de fora), 105 de treino e 45 de teste.

| saída | NN, R² teste | PINN (λ = 1), R² teste |
|---|---|---|
| q̄_L topo | 0,45 (0,60 média de 3 sementes) | 0,88 |
| q̄_G topo | 0,982 | 0,997 |
| q̄ gas lift | 0,983 | 0,998 |
| span q_L | 0,93 | 0,83 |
| span q_G | 0,37 | ≈ 0 |
| span q_inj | 0,43 | 0,43 |
| resíduos R1–R4 (RMS, em desvios padrão) | 0,65 / 0,02 / 0,05 / 0,11 | 0,16 / 0,005 / 0,02 / 0,03 |

Leituras: (i) a física resolve a saída que a rede sozinha não aprende, a produção média de líquido, porque o
balanço R1 a prende à pressão de fundo média, que é suave (R² 0,985); (ii) o gás no topo e o gás de lift ficam
quase exatos com a física (R² 0,997); (iii) os *spans* são marcadores ruidosos por natureza (o máximo de um
sopro em 3 ciclos) e nenhuma das duas redes os prevê bem, fora o de líquido (0,8–0,9): para a assinatura da
golfada é melhor usar a classe e a amplitude de pressão (a fronteira do mapa é nítida) do que o span do gás;
(iv) com poucos dados (10, 20, 40 pontos, 3 sorteios) o PINN é melhor em q̄_G e q̄ de lift em todos os tamanhos
(RMSE a metade) e empata em q̄_L e nos spans; (v) λ entre 0,1 e 100 dá o mesmo ajuste, com os resíduos a cair
com λ — fica λ = 1. Resultados em `runs/paper02/surrogate/` (`results.json`, modelos `nn.json` e `pinn.json`,
mapas, históricos). Próximos passos: alimentar a física também com a classe (fronteira) e com o resíduo do
modelo no estado final (`final_state.npz`, zero nos estacionários), e tratar os spans como alvo de classificação
ordinal em vez de regressão.

## Manuscrito (2026-10-19): um só artigo, otimização em espera

Decisão do dono: **um só artigo**, com a parte do modelo e dos substitutos e a parte da otimização (esta em espera,
marcada a vermelho com `\tbd`). `main.tex` tem (muito sucinto no modelo fenomenológico, com os modelos empíricos, os
resultados e a formulação da otimização): resumo, introdução, §2 modelo fenomenológico (remete para o
artigo 01, "Part 0"), §3 dados (amostragem em cadeias, horizonte adaptativo com captura, marcadores, mapa das
decisões), §4 substitutos (rede e física das médias R1–R4), §5 resultados (tabelas de métricas e de poucos dados,
paridade, mapas, curvas de aprendizagem), §6 otimização operacional (em espera: a formulação está escrita — receita do líquido médio
menos custo de compressão, restrições no span/pico de líquido e na oscilação de pressão, enumeração no plano e
confirmação pelo simulador —, os resultados ficam por fazer), §7 conclusões. Figuras e
tabelas só de `runs/paper02/base2d/dataset.csv` e `runs/paper02/surrogate/` (`figures/make_figures.py`:
`system` reusa o desenho da geometria do artigo 01, `dataset_map`, `surrogate_parity`, `surrogate_maps`,
`surrogate_learning`; `tab_dataset`, `tab_surrogate`, `tab_sparse`). Pendências: título e autores a confirmar
pelo dono; a secção da otimização (resultados) por fazer; os spans do gás (R² < 0,65) ficam declarados como limite.

## Otimização operacional (semente, 2026-10-19, `tools/paper02_optimize.py`)

Pedido do dono: FOB = preço do óleo × óleo produzido − custo de compressão do gás de lift de 5 bar até P_gas, sobre o
PINN; restrições nas vazões **máximas** previstas pelo substituto (o modelo passou a prever também q_L máx no topo e
q_gl máx, além das médias e dos spans): compressor (`--max-gl`) e separador (`--max-oil`). Óleo da vazão média de
líquido (700 kg de líquido por Sm³ de óleo do caso base); compressão politrópica em estágios de razão ≤ 4 com
arrefecimento (k 1,3, 313 K, η 0,75, M ≈ 19,5 g/mol do ρ_G do fluido). Enumeração em 201 × 201 nós admissíveis +
SLSQP com as restrições. Resultados com 70 $/bbl e 0,10 $/kWh, sucção 5 bar: ótimo no canto (choke aberto,
160 bar), FOB 290 k$/d = 292 k de receita − 1,95 k de compressão (813 kW): a compressão é < 1 % da receita, por
isso o ótimo sem restrições é a produção máxima; só com energia ≥ 5 $/kWh ou óleo ≤ 20 $/bbl o ótimo salta para
"sem gas lift, choke a 0,65". Com capacidades de exemplo (compressor 0,8 kg/s, separador 20 kg/s de pico) a
restrição do compressor é ativa e o ótimo vai para 122,6 bar com o choke aberto (FOB 266 k$/d). Verificação do ótimo
pelo simulador em `runs/paper02/optimize/verify/`. **Próximo (dono):** analisar para várias pressões de sucção/porto
e gerar a curva de Pareto; capacidades reais; a secção do artigo está marcada como semente.

**Extensão a 160–260 bar (2026-10-19, observação do dono: o ótimo não pode ficar na fronteira).** Segunda campanha
`runs/paper02/base2d_hi/` (3 rondas de 25, semente 2; 33 feitos quando se retreinou, o resto a correr): acima de 160
bar tudo o que assenta é estacionário; 7 de 33 esgotam o orçamento (chaveamento do topo do riser, a 0,5–0,8 e 168–244
bar) e o classificador lê-os como golfada — ficam fora do treino. Substitutos retreinados nas duas campanhas (176 de
189 corridas, 123/53): NN e PINN empatam em q̄_L (0,90/0,91) e nos gases (0,996–0,999); o PINN mantém os resíduos
4–6× menores e a vantagem com poucos dados (gás: ¼ do erro com 10 pontos). **Ótimo interior:** abertura 0,73,
228 bar, 309 k$/d (+6 % sobre o canto da 1.ª campanha), q̄_L prevista 5,75 kg/s, 1,75 kg/s de gás de lift; fica ao
lado da faixa de corridas com orçamento esgotado — verificação pelo simulador em `runs/paper02/optimize/verify2/`
(pendente ao escrever isto). Com compressor de 0,8 kg/s: (1,0, 117 bar), 268 k$/d, separador quase ativo (18,5 de 20).

**Terceira campanha (2026-10-19, pedido do dono: mais pontos a alta pressão de gás e abertura alta).**
`runs/paper02/base2d_hi_open/`: P_gas 160–260 bar × abertura 0,5–1,0, 2 rondas de 25 (semente 3), 2 instâncias. É a
região do ótimo e também a do chaveamento do topo do riser (orçamento de 40 min esgotado em 7 de 33 pontos da 2.ª
campanha nessa faixa): se a taxa de falhas se repetir, convém decidir o fechamento do troço horizontal ou subir o
orçamento. O guião das figuras já lê as três tabelas; o retreino usa `--dataset` com as três.

**Diagnóstico das falhas e quarta campanha (2026-10-19, meio-dia).** As três campanhas pararam a pedido do dono
(184 + 34 + 3 pontos feitos). O quadro da 2.ª campanha mostrou que o orçamento esgotado não é do ponto mas do caminho:
acima de 0,5 de abertura, os 8 pontos que arrancaram a quente do vizinho falharam todos (50–80 mil passos para
0–6000 s) e os 3 que arrancaram do estado final do caso base passaram em 36–69 s; abaixo de 0,5 todos os arranques
quentes passaram. Uma experiência no ponto (0,51, 216 bar), que falhara a quente, confirmou-o e descartou o troço
horizontal de 10 m como causa: arrancando do caso base assentou em 742 passos com o troço e em 563 sem ele (a
retirada dos 10 m não compra desempenho; a geometria do artigo 01 fica). Decisão do dono: partida estacionária em toda a região alargada, sem caso base nem corrida anterior
(`design --start steady`, implementado: Newton de 040 a partir da estimativa `noslip` do gerador, pressões × 1,01).
**Testado e abandonado no mesmo dia:** num piloto de 4 pontos o Newton não convergiu em nenhum (Jacobiana por
diferenças finitas singular; com a exata em JAX converge só a partir do próprio estado final, e falha a 20 bar de
distância), o pseudo-transiente de passo crescente divergiu nos 4 e a marcha estacionária a dois fluidos (Newton
reduzido nas variáveis estáticas com as vazões fixas, 162 incógnitas) travou nos 4; a causa, medida ao longo da
direção de Newton: a previsão linear levaria o resíduo de 140 a 0,06, mas o resíduo real salta 533 e depois 54 000
na quantidade de movimento do gás nos nós 14–15 da linha, o joelho da base do riser — os fechamentos trocam de
padrão e as equações estacionárias são descontínuas no estado. Um modelo de parâmetros concentrados daria vazões e
pressões, não a configuração de padrões por célula; o que resolveria seria um solver com o padrão como incógnita
discreta (dias, com risco de oscilar no joelho). Decisão do dono (tarde): **todos os pontos arrancam do estado final
do caso base** (`run --from-base`, gravado na ronda), retomando as três campanhas onde pararam; o canto de abertura
alta × pressão alta foi verificado antes de retomar (ver abaixo).

**Canto de abertura alta × pressão alta arrancado do caso base (2026-10-19, 4 corridas em paralelo, orçamento 15
min, máquina carregada):** (1,0, 260 bar) estacionário aos 9600 s em 12 629 passos e 8,5 min; (0,85, 240), (0,70,
255) e (1,0, 200) esgotaram os 15 min aos 5640, 8390 e 3810 s com 14–18 mil passos. Nenhuma rebenta: o integrador
rasteja a ~0,3 s por passo com 20–35 % de passos rejeitados, e isso não se vê nas saídas de 5 s (fração de gás das
células do topo estável em 0,86–0,90, oscilação de 2 bar na cabeça) — o chaveamento é dentro do passo, entre
tentativas do Newton do integrador. Fora do canto, (0,51, 216) custa 742 passos. Com os 40 min da campanha
passariam 2–3 destes 4, a 10–40 min cada; os ~50 pontos pendentes do canto são ~20 h em três instâncias. Em
mesmos dois pontos sem o troço horizontal de 10 m (`L_h2 = 0`, estado do caso base interpolado na malha nova):
(1,0, 200) 20 373 passos para 5695 s, passo médio 0,28 s, igual; (0,85, 240) 19 308 passos até assentar num ciclo
aos 12 200 s, passo médio 0,63 s. **O troço não é a causa**: a oscilação rápida do líquido no topo do riser (2–15
kg/s, período < 1 min) existe com e sem ele e é ela que obriga a passos de 0,3–0,6 s. A geometria fica. Retomadas
as três campanhas perto do meio-dia com `run --from-base --retry-failed` (5 instâncias: 1 + 2 + 2), orçamento de 40 min
como desenhado; os pontos do canto que o esgotarem ficam como falhas para retentar com orçamento maior se o dono
quiser.

**Retreino, otimização e artigo com as três campanhas (2026-10-19, 12:40, pedido do dono; campanhas ainda a
correr).** 231 corridas, 216 no treino/teste (151/65; 15 de orçamento esgotado fora). PINN: q̄_L R² 0,885 (NN 0,807),
gases 0,998–0,999, q_L máx 0,915 (NN 0,923), q_gl máx 0,995, pressões 0,978–0,999, **span de líquido 0,905** (NN
0,767; era < 0,65), spans de gás 0,42/0,54; resíduos R1 e R3 3× e 5× menores que a NN, R2 e R4 iguais (≤ 0,03);
grelha: 24/24 NN e 23/24 PINN com R² médio ≥ 0,8 (a arquitetura não limita); poucos dados: PINN ⅓ do erro nos gases
com 10–20 corridas, ½ com 40, q̄_L melhor a partir de 20. **Ótimo (70 $/bbl, 0,10 $/kWh): (0,94, 205 bar), 303 k$/d**
(receita 306 k, compressão 2,9 k, 1193 kW, 1,56 kg/s), +5 % sobre o canto (1, 160); q̄_L 5,64 kg/s; insensível ao
preço do óleo (20 $/bbl → (0,93, 201)); sai da caixa só pela energia: 2 $/kWh → (1,0, 173 bar), 5 $/kWh → (0,64, 89
bar), a abertura da válvula. Com compressor de 0,8 kg/s e separador de 20: (0,81, 123 bar), 262 k$/d, compressor
ativo, logo acima da fronteira (q_L máx 4,7); compressor de 0,5 → (0,88, 103 bar) dentro da golfada (picos de 18);
separador de 6 → restrição do separador ativa, (0,81, 103 bar). Verificação do ótimo pelo simulador a correr
(`scratchpad/top10/OPT3`, do caso base, orçamento 40 min). Artigo: resumo, destaques, §3.1 (três campanhas, arranque
do caso base para todos, porque o estacionário não é alcançável), §3.2 (orçamento: oscilação rápida no topo, troço
não é a causa), §3.4, §5, §6 e conclusões reescritos com estes números; PDF reconstruído.

**Épocas (2026-10-19, 13:30, pedido do dono: "retreine com mais épocas e rode tudo de novo").** 16 000 épocas
sobreajustam: PINN q̄_L 0,885 → 0,80, span de líquido 0,91 → −0,25, R1 0,08 → 0,23, R² de treino 0,99–1,00. Varredura
(1000 a 12 000, 217 pontos, 152/65): q̄_L de teste é plano entre 1000 e 2000 (NN 0,935/0,918, PINN 0,938/0,930) e cai
monotonamente depois (4000: 0,907/0,921; 8000: 0,82/0,85; 12 000: 0,78/0,80), com R1 a subir; **2000 épocas escolhidas
(paragem antecipada por varredura, declarada no artigo).** Novos números: PINN q̄_L 0,930 (NN 0,918), gases 0,996–0,998,
q_L máx 0,89, span de líquido 0,88 (NN 0,82), spans de gás < 0,4; resíduos R1/R3 3,5× menores; grelha 24/24 nas duas;
poucos dados: ⅓ do erro nos gases com 10–20, ½ com 40. **Ótimo: (1,0, 224 bar), 314 k$/d** (+8 % sobre (1, 160)), planalto
plano a 1 % de 200 a 255 bar e a 0,5 % entre aberturas 0,9 e 1; q̄_L 5,84, 1,73 kg/s de gás, 1356 kW; óleo a 20 $/bbl →
220 bar; energia a 2 $/kWh → 194 bar (compressão ⅙ da receita); 5 $/kWh → (0,61, 87 bar). Com compressor de 0,8:
(0,97, 122 bar), 265 k$/d; 0,5 → (0,87, 103 bar) na golfada (picos 14); separador de 6 → (0,78, 106 bar). Verificações
pelo simulador a correr: (0,94, 205) e (1,0, 224), do caso base. `--sparse ""` aceite pela ferramenta (sem o estudo).

**Verificação do ótimo e o ciclo rápido do topo (2026-10-19, 14:30).** (1,0, 224 bar) do caso base, 40 min: não assenta
(8705 s, 23 304 passos) mas as médias dos últimos 4000 s batem com a PINN — q̄_L 5,79 vs 5,84 (+0,8 %), q̄_G 1,88 vs
1,92, q_gl 1,71 vs 1,73, P_fundo 58,4 vs 56,0 bar, P_an 254 vs 258; o máximo não: a corrida está num **ciclo rápido do
topo do riser** (período ≈ 60 s, q_L no choke 2–16 kg/s, α das células do topo 0,6–0,95, cabeça plana a 21 ± 0,2 bar,
injeção constante), que a PINN, treinada nos pontos que assentaram, não vê (pico previsto 6,4). O ciclo aparece
abruptamente aos 6000 s, no reinício do integrador entre janelas, depois de a amplitude decair a 0,001 na janela
anterior — o modo é instável mas os passos grandes do BDF amortecem-no e os pequenos resolvem-no. Contraprova a
(1,0, 260 bar): reiniciada do estado "estacionário", fica estacionária (42 passos em 3600 s, amplitude 0): a 260 bar é
mesmo estável; a faixa 165–245 bar × 0,5–1,0 tem o ciclo rápido. Consequências: (i) as falhas de orçamento do canto são
físicas — o critério de envelope em janelas de 3000 s só aceita um ciclo de 60 s aos 9000 s, e a 0,3 s/passo isso são
30 mil passos, mais de 40 min com a máquina saturada; (ii) `cycle_class` lê esse ciclo como golfada quando pico > 2×
média (não é golfada severa: ΔP_cabeça 0,3 bar) — rever a classe com a amplitude de pressão ou o período; (iii) uma
restrição no pico de líquido no canto exige o ciclo rápido nos dados (orçamento ≈ 90 min nessa faixa, ou janelas de
paragem mais curtas para ciclos rápidos — opção a desenhar). Figura `scratchpad/top10/why_not_settled.png` (pedido do
dono: porque não assentam). §6 do artigo atualizado com a verificação; a marca pendente que fica é o Pareto.

**Issue 062 (2026-10-19, 15:00, aprovada pelo dono): janela da paragem escalada com o período e classe `fast`.**
`time.settle.window_periods = 10`, `window_min = 600`: quando o período medido cabe 10 vezes na janela, a comparação
passa a janelas de 10 períodos já sobre a série integrada (um ciclo de 60 s é aceite aos 3000 s em vez de 9000);
`cycle_class(period_fast=300)` dá a classe `fast` à intermitência do topo do riser (nas campanhas, ciclos assentados
≥ 546 s, ciclos rápidos 52–108 s). Os 23 pontos de orçamento esgotado das três campanhas foram movidos para
`runs/paper02/_budget_exhausted/<campanha>/<ronda>/<id>/` (nada apagado) e voltaram à fila; as três campanhas
relançadas com o critério novo nos `design.json` e no `rounds.json` (5 instâncias, `--from-base --retry-failed`).
A tabela e o mapa das decisões do artigo ganham a classe; o texto do §3.3/§3.4 atualiza-se quando os pontos do canto
estiverem refeitos (`markers` de novo → classe `fast` nos que ciclam rápido).
Continuação da verificação a (0,94, 205 bar) até 6220 s (lenta: 29 mil passos em 60 min a 0,06 s/passo depois do
reinício do estado gravado): quase estacionário com ondulação rápida de 0,6 kg/s; médias a 2–3 % da PINN, pressões a
2–5 %, pico 5,9 vs 6,3 — a amplitude do ciclo rápido varia ao longo da faixa (14 kg/s a (1,0, 224), 0,6 a (0,94, 205)).

**Campanhas paradas pelo dono (2026-10-19, 16:30).** Estado: 1.ª 187 feitos / 13 por fazer; 2.ª 32 / 43; 3.ª 12 / 38
(os 23 pontos de orçamento esgotado continuam em `runs/paper02/_budget_exhausted/`; `STOP` nas três raízes). Com o
critério 062 ligado, dos pontos devolvidos à fila: 5 da 2.ª campanha assentaram estacionários em 70–95 s (eram falhas
de arranque quente), e 2 da 3.ª, (0,64, 219) e (0,98, 206), esgotaram outra vez os 40 min antes dos 3000 s com 26 mil
passos — aí o custo é o transiente de arranque a 0,1 s/passo, o critério nunca chega a atuar; só orçamento (≈ 60 min)
os resolve. Nenhum ponto foi ainda aceite como ciclo rápido pelo 062 dentro das campanhas (os dois reruns exatos
mostraram o ciclo a nascer aos 3000–4200 s). Tabelas atuais: 231 corridas no CSV antes da paragem (refazer `markers`
com a classe `fast` quando se retomar).

**Reruns dos esgotados e nova ronda de treino/otimização/artigo (2026-10-19, 17:00, pedido do dono).** `run --only`
(opção nova) sobre os 23 esgotados, 8 instâncias, orçamento 60 min: 7 já assentaram estacionários em 73–160 s (eram
falhas do arranque quente), 3 falharam outra vez sob os 40 min antes da paragem e voltaram à fila, 13 a correr.
Tabelas refeitas com a classe `fast` (nenhum `fast` ainda — os ciclos rápidos são exatamente os esgotados fora da
tabela): 230 corridas assentadas, 161/69. Com 2000 épocas: NN e PINN iguais nas médias (q̄_L 0,89, RMSE 0,24;
gases 0,999; pressões 0,99–1,00); máximos q_gl 0,998/0,997, q_L 0,90/0,84; **spans dependem da partição** (span de
líquido NN 0,90 vs PINN 0,64 nesta, invertido na anterior) — o artigo passa a dizê-lo; resíduos R1 4× e R3 3× menores
na PINN; poucos dados: gases ⅓ do erro em 10–40, q̄_L melhor a partir de 20. **Ótimo (1,0, 242 bar), 308 k$/d**, +6 %
sobre (1, 160); planalto plano a 1 % de 205 a 255 bar e a 0,5 % em abertura 0,8–1 (os ajustes sucessivos movem o ótimo
20 bar ao longo do planalto: é a resolução que os dados dão); verificações a (1,0, 224) e (0,94, 205) batem a < 1 % nas
médias e pressões com a PINN nova. Com compressor 0,8: 123 bar, 260 k$/d, abertura indiferente acima de 0,7; energia
2 $/kWh → 173 bar; 5 $/kWh → (0,77, 85 bar) sem gas lift; óleo 20 $/bbl → 228 bar. Artigo: resumo, destaques, §3.2
(janela encurtada, orçamento 60 min), §3.3 (classe ciclo rápido), §3.4, §4, §5, §6 e conclusões; PDF reconstruído.
Pedido do dono: compressor de 1,0 kg/s (`runs/paper02/optimize_gl1/`): ótimo (1,0, 142 bar), 284 k$/d (−8 % face ao
irrestrito), q_gl máx = 1,0 ativo, pico de líquido 5,9, acima da fronteira; ao longo da restrição −1 % a abertura 0,9 e −6 % a
0,5. Com separador de 20 nada muda.
Compressor de 1,5 kg/s + separador de 20 (`runs/paper02/optimize_gl15/`): (1,0, 200 bar), 304 k$/d (−1 %), compressor ativo,
separador não; ao longo da restrição −0,6 % a 0,9 e −3,5 % a 0,5. O §6 passou a apresentar a varredura de capacidades
1,5 / 1,0 / 0,8 / 0,5 kg/s como uma descida pela lei da válvula.
Decisão do dono (17:40): **o problema do artigo é o restringido** — compressor de 1,75 kg/s e separador de 20 kg/s
(`runs/paper02/optimize/`; o irrestrito ficou em `runs/paper02/optimize_unconstrained/`): ótimo (1,0, 229 bar), 308 k$/d,
compressor ativo, pico de líquido 5,7; ao longo da restrição −0,3 % a 0,9 e −2,4 % a 0,5. A análise sem restrições
(planalto, sensibilidades a preços, canto da 1.ª campanha) saiu do texto; a estrela da Fig. 6 é este ótimo e a curva
de 1,75 entrou nas de nível; a verificação a (1,0, 224) fica a 4 bar do ótimo sobre a restrição.

**Retreino com os 8 esgotados refeitos (2026-10-19, 18:30, pedido do dono).** 232 corridas (162/70): NN e PINN iguais
nas médias (q̄_L 0,92, RMSE 0,20; gases 0,997–1,000); máximos de líquido 0,77 (NN) vs 0,90 (PINN) nesta partição; span
de líquido 0,62 vs 0,85; resíduos R1 4× e R3 6× menores; poucos dados: PINN ¼–⅓ do erro nos gases e melhor em q̄_L em
todos os tamanhos. **Ótimo do exemplo (1,75 / 20): (1,0, 224 bar), 315 k$/d** — exatamente o ponto verificado pelo
simulador (médias a 1 %, P_fundo a 6 %); varredura: 1,5 → 194 bar (−2 %), 1,0 → 141 bar (−11 %), 0,8 → 125 bar (−17 %),
0,5 → 102 bar na golfada (picos 17), + separador 6 → 106 bar. Fig. 6 com as regiões inadmissíveis pintadas e só as curvas
1,75 e 20 (pedidos do dono). Texto, figuras e PDF atualizados.

**Fecho dos reruns e retreino com 4 tabelas (2026-10-19, 21:00).** 22 dos 23 esgotados refeitos com 60 min: 7
estacionários em < 3 min, 4 ciclos aceites (37–57 min; 3 ciclos rápidos pela janela encurtada, um a 0,11 s/passo
com 71 mil passos), 11 esgotados outra vez a 0,04–0,1 s/passo (3 na fronteira em golfada severa, 8 no canto do
ciclo rápido); (0,73, 174) ainda a correr. Tabela extra `runs/paper02/extra/` (corridas de verificação assentadas:
(1,0, 260) e (0,95, 260), ambas estacionárias com 5,80 kg/s) entra na Fig. 2 e no treino. 247 corridas, 236
assentadas, 165/71: PINN q̄_L 0,894 (NN 0,876), gases 0,996–0,998, máximos de líquido 0,73 nas duas (partição
difícil), resíduos R1 4× e R3 2× menores; poucos dados: PINN ⅓–½ do erro nos gases. Ótimo (1,75 / 20): (1,0, 229
bar), 309 k$/d; verificações a (1,0, 224), (0,94, 205) e (0,95, 260) batem a < 1,5 % nas médias e pressões. Lição
registada no artigo: máximos e spans variam com a partição (0,73–0,92 nos máximos de líquido).
Pedido do dono (21:30): o exemplo do artigo passa a ter **só a restrição do compressor (q_gl máx ≤ 1,75)**; o separador
de 20 fica na formulação e na varredura, e a região q_L máx > 20 da Fig. 6 fica como referência. Ótimo idêntico
(`runs/paper02/optimize/` refeito só com q_gl).

**4000 épocas (2026-10-19, 22:00, pedido do dono "mais épocas").** Varredura nas 4 tabelas (237 assentadas): a PINN é
insensível ao tempo de treino (q̄_L 0,91–0,92 de 1000 a 8000 épocas), a NN sobreajusta a partir de ~1000 (q̄_L 0,91 → 0,83,
máximo de líquido 0,87 → 0; mínimo da perda de teste às 674/848 épocas). Artigo com 4000 épocas nas duas redes **para
mostrar a diferença** (a física como regularizador), painel (d) novo na Fig. 5 com perda de treino e teste por época.
Números: PINN q̄_L 0,916 (NN 0,889), gases 0,999, máximo de líquido 0,71 vs 0,49, resíduos R1/R3 3,6×/2,6× menores;
ótimo (1,75): (1,0, 228 bar), 308 k$/d; verificações a < 1,2 %; 1,5 kg/s → 190 bar com pico previsto 13 (os 3 ciclos
rápidos já no treino); tabela extra com (0,75, 242).

## Otimização (plano original, vários poços)

- N poços (4–8, a decidir) em fases diferentes da vida; decisões por poço: P_gas e abertura do choke.
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
- Em cada estágio resolve-se a alocação; o resultado é a trajetória ótima de P_gas e da abertura do choke de cada
  poço ao longo dos anos.
- Comparação com a mesma alocação feita por um otimizador estacionário (sem restrição de estabilidade), avaliada
  depois com a dinâmica: é a medida do viés (contribuição 1).

## Limitações declaradas

- Sem água: escoamento bifásico óleo–gás.
- Trajetórias do reservatório exógenas (sem balanço material: produzir mais agora não antecipa a depleção).
- Uma só geometria; poços sem coletor de produção comum.
- Fluido com divisão entre fases congelada (como no artigo 01).
- Padrão automático ainda sem validação de laboratório (057); e o fechamento do troço horizontal curto a jusante do
  riser, onde nenhum padrão está estabelecido, é uma escolha, não um resultado.

## Dependências e etapas

0. Artigo 01 fechado e modelo congelado: **feito**, incluindo a §8.8 com a geometria do gas lift.
1. Rede de gas lift sobre o caso offshore: **feito** (issue 060). Pendentes: decisão sobre o fechamento do topo do
   riser (issue própria se mudar o modelo); verificação contra o caso 10.3.2 (heading).
2. Issue 061: paragem adaptativa com vários sinais, ciclos alternados e captura de N_c ciclos: **feito**
   (chaves opcionais de `time.settle`; o comportamento anterior fica intacto).
3. `tools/offshore_linear.py` estendido ao ramo (estacionário e espetro em função de P_gas e das cinco entradas).
4. `tools/paper02_dataset.py` (design, run, markers, status): **feito** (061), com fumo de ponta a ponta numa
   cadeia de 3 pontos; falta aplicar `markers` aos sete casos do artigo 01 (`runs/paper01/gaslift/`, sem janela
   de captura: últimos 4000 s) como primeiro conjunto de dados.
5. Lote piloto (50 pontos em 2–3 cadeias, mais os mesmos pontos da partida bloqueada) para calibrar faixas,
   horizonte, N_c, custo e a poupança do arranque quente, e para a primeira verificação de histerese; depois o
   lote de treino em cadeias, o de teste independente e o adaptativo.
6. Substituto, validação e incerteza.
7. Otimização de N poços, trajetórias ao longo da vida e comparação com o otimizador estacionário.
8. Figuras (só a partir de corridas e de caches) e manuscrito.

## Decisões em aberto para o dono

- **Fechamento do troço horizontal curto a jusante do riser** (10 m = 66 D): excluir o estratificado aí (issue
  própria) ou aceitar o custo. Condiciona o custo de metade dos lotes.
- **N_c** (número de ciclos capturados depois de o regime estar estabelecido): proposta 3, mínimo 2.
- **Definição das frequências**: intervalos entre máximos do sinal suavizado (proposta) ou cruzamentos da média
  (o que `window_signature` faz hoje); e se se guarda também a frequência dominante do espectro.
- **Faixas das cinco variáveis** e as trajetórias de vida (declínio de P_res e PI, subida do RGO).
- **Desenho de experiências**: Sobol embaralhada (proposta) ou sorteio uniforme; tamanhos dos conjuntos (400/100
  propostos) e do lote adaptativo; cadeias de vizinhos em 5 dimensões ou passeios nas duas decisões por estado do
  poço (a decidir com o piloto).
- Número de poços e capacidade do compressor; preço do óleo, custo da energia de compressão e capacidade do
  separador (restrições).
- Revista alvo e autores (no artigo 01: Jaime Neiva Miranda de Souza, Miradouro Digital, e André Luiz Hemerly
  Costa, UERJ/IQ, correspondente).

## Bibliografia a levantar

Sem chaves inventadas: cada referência entra em `refs.bib` (ou em `../_common/references.bib`) só depois de o DOI
estar verificado. Temas: revisões de otimização de gas lift (por exemplo, Rashid, Bailey e Couët, 2012, a
verificar); alocação de gás entre poços; modelos substitutos (redes neurais, krigagem) em produção de petróleo;
desenho de experiências quase-aleatório (Sobol, Owen) e amostragem adaptativa; *heading* em poços com gas lift;
mitigação da golfada severa por gas lift na base do riser (Jansen et al., 1996, já na bibliografia comum).
