# Artigo 01 — plano: análise, casos, exemplo base

*Documento de trabalho do artigo (português, como o resto da documentação; o manuscrito é em inglês).
Escrito em 2026-10-16 a partir do pedido do dono e atualizado à medida que as corridas chegam.*

## O pedido, em uma frase

Um artigo novo, em crescendo — modelo e a sua boa colocação, discretizações testadas, integrador
adotado, backends —, com o caso principal na intermitência severa (golfada severa) e no mapa de
estabilidade com choke, padrão de escoamento calculado dinamicamente (nenhum trecho com padrão fixo),
perfis bonitos no tempo e no espaço, a estabilidade da célula da curva e a evolução da golfada ao longo
do duto. Sem comparar com os resultados da tese (os erros ficam lá); as lições de "não seguir por aqui"
aparecem nas entrelinhas, como observações gerais medidas com o código de hoje.

## Análise das ideias

| ideia do dono | o que o código e as corridas já dizem | como entra no artigo |
|---|---|---|
| impacto negativo de uma abordagem **não conservativa** (lida como: integrar a forma primitiva, sem resíduo conservativo) | issue 004: a forma primitiva perde 2–6 % do líquido e 8–12 % do gás, em degraus, um por blowdown; o erro só se mede com o acumulador BDF-consistente (005), a regra dos trapézios engana (−1,7 kg). O custo da conservativa: pressão escrava da massa (×1,45 passos sem arco; mais barata com arco) | secção de discretização: tabela e figura do balanço de massa (estoque − integral dos fluxos) nas duas formulações, no caso base com padrão dinâmico; lição: "verificar o balanço com a mesma recorrência do integrador" |
| o modelo **mal posto** em certos padrões | sondas de 2026-10-16: estratificado mal posto com camada de gás fina (αG ≤ 0,003, qualquer escorregamento); anular mal posto em quase todo o mapa (pressões das fases iguais); bolhas em αG ≤ 0,03 com escorregamento; intermitente em αG ≤ 0,01 e a escorregamento alto. Regra de fração baixa + massa virtual (forma objetiva) tornam bolhas e a camada fina hiperbólicos; Bestion (δ = 1,2) torna anular e intermitente hiperbólicos. **Na trajetória do caso base** (padrão dinâmico) o modelo é mal posto em 0,9 % das células-instante (mediana de |Im λ| 0,001 m/s); sem a regra de fração baixa e sem massa virtual os mesmos estados dariam 23–30 % | secção "boa colocação": mapas de max|Im λ| por padrão (αG × escorregamento), com e sem regularização; fração mal posta ao longo das trajetórias simuladas; lição: "escolher métodos indiferentes à classificação do sistema não basta" |
| **impossibilidade de convergir** sem as melhorias | issues 002–011, 020–021, 032–036: esquemas colocados oscilam nó a nó; "upwind" regressivo em todas as variáveis é instável no modo acústico; limitadores largos criam ciclos falsos; IPE da tese rasteja (9834 passos até 1,5 s); BDF1 catastrófico; arrasto espúrio da fase virtual | tabela de ablação no caso base: tira-se um ingrediente de cada vez e mede-se se a corrida chega ao fim, quanto custa, o erro de massa e a assinatura do ciclo |
| o grande impacto do **JAX** | issue 036: avaliação 0,43 → 0,08 ms, Jacobiana 10,7 → 0,9 ms; caso de 3000 s de 5 min para 45 s; malha fina de 58 min para 105 s; a Jacobiana exata não reduz as iterações de Newton (não-suavidade), o ganho é o custo por passo; paridade 1e−14 | secção de backends: tabela de tempos numpy × JAX no caso base com padrão dinâmico (rhs, Jacobiana, passo, corrida completa, mapa inteiro em paralelo) |
| caso principal: **golfada severa** e o **mapa** de estabilidade | issue 043 (padrão fixo): 252 corridas, fronteira = critério de Jansen (queda no choke ≈ 0,67 bar), 8 células de histerese, +7 % de produção possível a 15 bar | o mapa refeito com padrão dinâmico (mesma grelha, duas partidas), comparado com o critério de Jansen e com a fronteira do padrão fixo |
| crescendo: modelo → discretização → integrador → backend → aplicação | — | ordem das secções 2–7 |
| **perfis bonitos** com e sem golfada para várias aberturas | as corridas do mapa têm saída a 5 s e malha grossa | corridas dedicadas a 15 bar (aberturas estável, ciclo de pressão, golfada severa e aberto) com saída a 1 s: diagramas x–t de fração de líquido, pressão e padrão; séries de pressão e vazões |
| **estabilidade da célula na curva** (pescoço) | issue 042 + nb_estabilidade_mapa: modos instáveis rápidos e locais no ponto baixo (até +27 /s), crescem com a malha, saturam a 0,005–0,02 bar, não desenham a fronteira. **Achado novo (2026-10-16)**: o estacionário discreto tem um xadrez na fração de gás do riser e o modo de +41 /s na curva porque os fechamentos das faces usam a **média** das células e o transporte/gravidade a célula **doadora**; com os fechamentos pela doadora o perfil fica liso e o modo da curva desaparece; sobra um modo na célula da frente (fim do descendente) e modos fracos (≈ 4 /s, 40–45 rad/s) na linha estratificada | secção de estabilidade linear: espetro do estacionário, localização dos modos, efeito da malha e da coerência da fração nas faces; a não-linearidade (histerese) explica porque o mapa tem de ser dinâmico |
| **mapa de padrão dinâmico** (nenhum trecho fixo) e perfis de padrão no estacionário | identificação automática suave e local (021) com a regra de fração baixa: entrada horizontal intermitente, descendente estratificado, ponto baixo bolhas/intermitente, riser bolhas, anular no blowdown; custo ≈ 4× o padrão fixo | perfis de padrão ao longo do duto no estacionário para várias aberturas; diagrama x–t do padrão dominante ao longo de um ciclo |
| **evolução da golfada** ao longo do perfil e do tempo | o notebook mostra o x–t da fração de gás perto do riser | rastreio das frentes: frente de líquido no descendente (penetração do tampão na linha), nível de líquido no riser, frente de gás no blowout; comprimento do tampão no tempo; as quatro fases do ciclo (formação, produção, blowout, queda do líquido) marcadas no x–t |

## Lições nas entrelinhas (sem apontar a tese)

1. Métodos "indiferentes à classificação" das EDP não escapam da má-posição: ela aparece como crescimento
   dependente da malha; regularizar com termos físicos (massa virtual, regra de fração baixa) e verificar
   a hiperbolicidade **nos estados visitados**.
2. Esquemas colocados sem dissipação nas ondas de fração e "upwind" por diferenças regressivas em todas as
   variáveis (pressão incluída) não transportam frentes de golfada; malha escalonada com doador por fase.
3. Integrar a forma primitiva perde massa num degrau por blowdown; o balanço verifica-se com a mesma
   recorrência do integrador, não com quadratura das saídas.
4. A fração nas faces tem de ser a mesma em todos os termos da equação de momento (achado 044).
5. Mapas de padrão de regime permanente não servem em dinâmica sem avaliar os critérios no estado local,
   com chaveamentos de largura relativa e pesos que formam partição da unidade.
6. Fechamentos dimensionalmente inconsistentes compensam-se num caso e falham noutra escala: testar a
   escalas diferentes (laboratório e campo).
7. Uma válvula com densidade ponderada pela massa não resiste ao gás e fica singular em refluxo.
8. Um integrador de caixa fechada não deixa conservar, projetar em limites nem escolher tolerâncias por
   faixa; a Jacobiana exata não reduz as iterações quando os fechamentos não são suaves: o ganho está no
   custo por passo (compilação).
9. ~~Os autovalores do estacionário não desenham a fronteira da golfada~~ (era o espetro sem matriz de massa).
   Lição certa (048): numa formulação conservativa linearizar com a matriz de massa, J·v = λ·M·v; J sozinha
   mostra modos rápidos que não existem e esconde o par lento que marca o início dos ciclos (Hopf, com o
   período do ciclo). As simulações dinâmicas com duas partidas dão a amplitude e a histerese.
10. A contagem de ciclos por cruzamentos da média é enganada por oscilações curtas (coluna do riser contra o
   gás do bolsão, ~8 s): média móvel antes de contar (`cycle_class(smooth=)`).

## Decisões e achados durante a preparação (2026-10-16)

- **Espetro com a matriz de massa (issue 048).** Os espetros (artigo, notebook 043, issues 042/044) usavam J
  sozinha; no modo conservativo o rhs devolve dT/dt e a linearização é M·ẋ = J·x. Com M, a 15 bar o par lento
  cruza o eixo em a_H = 0,24 (período ~575 s; Hopf), os transitórios seguem o par (0,36: previsto 455 s e
  3,6e−3 /s, simulado ciclo de 450 s e 2,9e−3 /s; 0,28: 518 contra 495 s, crescimento mais lento); a ≥ 0,70
  um modo global real fica instável (relaxação). A fronteira de Hopf sobe com a pressão (~0,07 a 8 bar,
  ~0,40 a 19 bar). Secção 7.3 reescrita (figura `linear_stability`, tabela `tab_hopf_mesh`).
- **Problema-modelo (secção 2, pedido do dono).** Duas EDP lineares de coeficientes constantes (equações do
  telégrafo: a linha/bolsão) com fonte linear (IPR) e terminação de um estado com característica cúbica em N
  (riser + choke `a`): Nagumo e Shimura (1961). Mostra o método todo com o BDF do mfsim e T = M x: Hopf em
  a_H = 0,976 (período 6,75), colisão no eixo real em a_R = 0,756 (relaxação com refluxo, período 12,3 a 0,5),
  J sozinha «estável» em todo o a. Código em `toy/model_problem.py`; figura `model_problem`.
- **Geometria da base do riser (proposta ao dono, à espera de resposta).** Cotovelo (caso da ablação), arco
  R = 5 m (base), arco R ≈ 15 m, riser em catenária (tangente aos −3°, ~85° no topo), mesmo desnível de 20 m:
  estabilidade linear a 15 bar (barata) e transitórios a 1,0 depois da ablação.
- **Mapa linear (artigo, figura do mapa painel a).** Espetro com M nos 107 estacionários do mapa com padrão
  dinâmico: estável → par lento instável (Hopf) → modo global real instável ao abrir o choke. A fronteira linear
  tem a forma do critério de Jansen (queda no choke constante) mas com Δp crítico 0,34–0,40 bar (10–24 bar)
  em vez de 0,67; aberturas críticas 1,3–1,6× as de Jansen. Padrão fixo (043, verificado a 15 bar: a_H 0,217;
  Δp 0,37–0,48 bar segundo a sessão do 043). A 19–48 bar há um modo rápido global da linha (2–3 s) instável em
  vários pontos (linha hiperbólica: instabilidade linear, não má-posição). Paragem adaptativa (041) não vê
  crescimentos lentos (<~1,4e−3/s): a estabilidade do estacionário vem da análise linear; o mapa dinâmico só com
  a partida bloqueada (`runs/maps/well_auto/blocked/batch.json`).
- **Fechamentos das faces pela célula doadora (issue 044).** Com a média das células o estacionário discreto
  tinha um xadrez na fração de gás do riser e, a 0,13, um par instável local na curva (25 s; o transitório
  satura numa oscilação de 26 s das células da curva); com a doadora o perfil é liso e o estacionário estável. Na
  dinâmica a doadora muda o ciclo na malha grossa (intervalo 985 s contra 824 s em 1500 s de referência sem
  poço) e o custo por ciclo é parecido (as duas têm fases caras no blowdown); a convergência com a malha
  decide-se no lote `paper01_mesh`. Adotada no exemplo base.
- **Ponte do filme do anular com a fração local (issue 046).** O modo local ainda usava o filme de
  equilíbrio de Pots e marcava como anular a célula do ponto baixo com 91 % de líquido; corrigido com a
  opção `local_film` (estacionário igual, padrão intermitente nessa célula).
- **Orçamento de tempo de relógio (issue 045).** Um passo mínimo de 1e−6 s faz falhar a própria referência
  (blowdown da partida bloqueada aos 8 s); os lotes de comparação usam `integration.max_cpu = 5400`.
- **Continuação do mapa com padrão dinâmico.** As quinas da identificação de padrão (tabelas de equilíbrio,
  máximos suaves) fazem o Newton estagnar em alguns passos; com passagens de preenchimento a partir de
  vizinhos, 107 de 126 pontos têm estacionário (falham os de pressão alta e choke aberto, ≥ 30 bar); esses
  partem da linha bloqueada nas duas colunas do mapa.
- **Boa colocação medida**: mapas por padrão mal postos em 24 % (estratificado, camada de gás fina), 49 %
  (intermitente), 46 % (bolhas), 94 % (anular); massa virtual objetiva leva as bolhas a 0 % e a simples
  (λ = 2) a 99 %; na trajetória de referência 0,9 % das células-instante são mal postas (23–30 % sem as duas
  regularizações nos mesmos estados).
- **Desempenho (medida sob carga)**: lado direito 2,4 ms (numpy) contra 0,25 ms (JAX), Jacobiana 63 ms
  (diferenças coloridas) contra 3,3 ms (exata), compilação ≈ 10 s; repetir com a máquina livre.

## Mudança de exemplo base (decisão do dono, 2026-10-16, à noite)

O exemplo anterior (linha de 295 m, riser de 15 m, 2 pol.) é de escala piloto. O dono decidiu:

- **Um só caso base, offshore:** o sistema do caso offshore da dissertação do Gustavo (Gustavo Alves de Carvalho,
  UERJ 2022; reproduzida em `~/code/personal/research-factory/01-multiphase-compositional-flow`):
  - poço vertical de 1500 m (D 0,127 m) com IPR no fundo;
  - linha de 6000 m e riser de 300 m (D 0,1524 m);
  - um trecho final da linha a −2° perto do cotovelo do riser (comprimento a avaliar, mantendo a geometria);
  - todas as corridas do artigo refeitas com ele.
- **Fechamentos:**
  - frequência de golfada de Zabaras em todos os exemplos, sem estudo de sensibilidade no artigo (issue 049);
  - sigmoides mantidas;
  - Apêndice A com as expressões exatas e as atribuições certas: a fórmula da tese é a eq. 5.83, atribuída a
    Manolis 1995; o docstring dizia Zabaras.
- **Lotes antigos parados:** o de malha e o do mapa bloqueado foram parados; as corridas parciais ficam em `runs/`.

Achados ao montar o caso (gerador `generators/offshore.py`):

- **Unidade de golfada da tese.** A célula com mais gás do que a unidade guarda perdia o balanço de massa (corte do
  filme a 0,95). Opção `closures.intermittent.unit = "consistent"` (issue 050).
- **Limitação que fica.** A alta pressão (linha toda intermitente, 100–250 bar), o filme fino tem velocidade para
  trás não física. O atrito da mistura sai negativo e o quase-estacionário tem a pressão a subir ao longo da linha.
  Com o reservatório depletado (20–40 bar na linha) o atrito é positivo.
- **Fluido do §5.5 congelado.** Tabela 19, fração mássica de gás 0,29 a 20 bar e 40 °C:
  - com o choke aberto e P_res de 70–110 bar, a linha é anular ou intermitente com o gás a ~12 m/s;
  - os estacionários são estáveis;
  - não se encontrou intermitência severa.
- **Fluido do §5.7.** Tabela 6, mesma geometria, fração de gás 0,10. A estimativa de Bøe e Taitel dá φ ≈ 0,24 e um
  recuo do líquido de ~900 m no trecho a −2°, o que justifica L_down ≈ 1000 m. Em teste.
- **Choke.** Com o separador a 10 bar e a lei homogénea (densidades a jusante), CV 150 dá < 1 bar a 5 kg/s com o
  choke aberto. O ponto nominal do Gustavo (10 kg/s, 91,5 bar no topo do riser) fica perto da abertura 0,19.

### Noite de 16 para 17 de outubro: a unidade de golfada no caso offshore

- **Decisões do dono (noite de 16/10):** fechamento intermitente com o filme de equilíbrio (051), fluido da Tabela 6,
  poço em variáveis de mercado (IPR em Sm³/d/bar, RGO em Sm³/Sm³) com a RGO como variável a varrer; nenhum padrão
  fixado por troço (a identificação dinâmica tem de dar estratificado nos troços horizontais e descendentes por si).
- **Pergunta do dono: «como é possível atrito negativo?»** Não é atrito com a parede negativo: é a soma Γ_L + Γ_G
  (parede + interface) da unidade da tese, em que o filme a −25 m/s (cinemática da unidade com α_f cortado a 0,95)
  arrasta a mistura para trás. Era um artefacto da unidade, resolvido pela 051.
- **Três patologias encontradas e corrigidas na 051 (todas com teste):**
  1. bolsão quase parado no vertical: a raiz de Brotz dá α_f → 1 e velocidades ~1/R_f (1e12 N/m) → o filme nunca passa
     de 0,95 (acima é anular, pelos pesos); célula mais rica fica só filme no estado da célula;
  2. arranque violento: a estimativa inicial com frações de equilíbrio punha gás a 37 m/s na curva → a estimativa é sem
     escorregamento (forças interfaciais nulas) e o transitório rearranja o holdup;
  3. ponto triplo α_G ≈ α_s⁰ ≈ α_f^eq (linha horizontal a 0,5 de gás): r saltava de 0 a 1 e a célula passava de só filme
     (2,7 N/m) a só pistão com o escorregamento inteiro (11 m/s) no fechamento de bolhas (−1,7e5 N/m) → o pistão fica
     pelo menos 0,1 menos arejado do que o filme (r contínuo) e o escorregamento no corpo do pistão fica limitado a
     1 m/s na força de bolhas. Medido num estado de 120 bar com o choke aberto: 60 s passaram de 1272 passos/442
     rejeições para 34/6.
- **Mais três patologias corrigidas (01:00–01:30):** quinas dos `min`/`max` (Jacobiana de um só lado), sinal
  discreto das velocidades na unidade (riser com o líquido a oscilar em torno de zero) e, a decisiva, a perda da
  pressão interfacial de Stuhmiller no riser (má-posição: 18 ondas instáveis de 0,3–2,5 s). Com tudo isto, 100 s a
  partir de um estado de baixa vazão passaram de 3255 para 97 passos.
- **Procura do ponto base:** com o choke aberto e P_res 120–200 bar a linha é intermitente e o riser anular/churn,
  sem golfada severa (vazões de 10–20 kg/s). A golfada severa pede linha estratificada: P_res 100–110 bar com RGO
  50 (1,7–3,7 kg/s); a partida bloqueada (gerador `--ic blocked`) poupa o enchimento da linha de 6 km. Em curso:
  (110, RGO 50) e (110, RGO 122) com o choke aberto, (110, RGO 50) com 0,1 e 0,2.
- **Custo:** a fase com o topo do riser cheio de gás (fase líquida quase extinta) custa ~25 passos por segundo
  simulado; um ciclo de golfada severa numa linha de 6 km tem período da ordem de 1–2 h. Lotes com tf 8000 s e
  menos variantes (`tools/offshore_batches.py`).
- **Célula da cabeça do poço (02:30):** com o poço a entregar gás (sopro), a primeira célula da linha esvazia de
  líquido (α_L = 0,001) e a velocidade do líquido na face seguinte chega a −163 m/s: a relaxação da fase que
  desaparece olha para a célula doadora *da fase* (a jusante, com líquido) e não dispara, enquanto a fração dos
  fechamentos vem da doadora *da mistura* (a célula vazia). Força de −8900 N/m sobre 0,1 % de líquido; o Newton
  falha nas células 0–4. Pista para o futuro: fração doadora por fase nos fechamentos das faces (variante de 044).
  Subir alpha_min para 0,01 não mudou o número de passos: o custo vem sobretudo do riser (fase quase extinta no
  topo). Aceite para esta noite; ~16 passos por segundo simulado.
- **Sondas com riser de 8 células (02:45–02:55):** a 200 bar / PI 5, RGO 50 entra em fase bloqueada aos 1000 s
  (linha toda estratificada, base do riser cheia, topo com gás, q_L,out = 0, poço a produzir 4,4 kg/s) — golfada
  severa em formação, com o primeiro sopro previsto para ~5000 s (0,17 kg/s de gás a pressurizar 6 km); RGO 122
  pressuriza 3× mais depressa (ciclos de ~2000 s), ao preço do riser rico em gás (0,8) e da célula da cabeça do poço.
  Custo: ~17 passos por segundo simulado na fase bloqueada.
- **Resultados às 05:20:** aberturas 1,0 e 0,6 — golfada severa com período ~4300 s, 14–33 bar, surto de
  40 kg/s (10× a produção média de 1,9 kg/s); 0,3 — ciclo de pressão parcial (riser nunca esvazia, período 2400 s,
  23–34 bar); RGO 80 — período 2300 s com a mesma faixa de pressão. Em curso: 0,2, 0,1, RGO 30 e 122, ablação (12),
  malha (2), variante primitiva (relançada com `relax_target=raw`). Texto e tabelas atualizados à medida.
- **Ponto base definitivo (03:10): 200 bar, PI 5, RGO 50.** A RGO 122 (riser rico em gás a 0,8 e célula da
  cabeça do poço patológica) custa 90 passos por segundo simulado e o lote não acabaria de manhã; a RGO 50 entra em
  golfada severa (bloqueio aos 1000 s, pressão da linha de 18 → 32 bar, sopro a partir dos ~4000 s) com a fase
  bloqueada barata (~1 passo/s). Lotes relançados com `--GOR 50`; o lote de RGO cobre 30, 80 e 122.
- **Ponto base (01:45):** a 110 bar com RGO 50 o poço morre quando o riser bloqueia (coluna do riser, 21 bar, mais a
  do poço de 1500 m excedem a pressão do reservatório: P_wf 116 > 110 bar) — sem ciclo. Um poço produtivo a
  200 bar (PI 42,9) dá 10–20 kg/s e a linha fica intermitente. O ponto base passa a um **poço fraco a 200 bar**:
  PI 5 Sm³/d/bar, RGO 122 (o do fluido), ~4–6 kg/s. Lotes relançados (`tools/offshore_batches.py --P-res 200
  --GOR 122.13 --PI 5`); sonda `e_200_pi5` em paralelo.

### Manhã de 17 de outubro: integrador, estabilidade linear e o que falta

- **Quinas dos fechamentos (052):** versões C1 de todas as quinas C0 restantes como opção (`closures.kinks`,
  `scheme.options.relax_ramp`); nenhum ganho nos passos (423→425, 3987→4503, 1072→1260). O custo das fases rápidas
  é o Newton nas frentes, não a suavidade dos fechamentos.
- **Newton do BDF (053):** busca em linha de Armijo em todas as iterações (`integration.newton_damping=armijo`) corta
  10–41 % dos passos e das rejeições e menos Jacobianas nas três janelas (bloqueio, sopro, fim do sopro); políticas
  da Jacobiana (`jac_policy=step|rate`) trocam passos por Jacobianas sem ganho líquido sob carga; iterado inicial
  (`predictor=none|euler`) sem ganho, Euler catastrófico. Omissão intacta (goldens A). Os casos do artigo ficaram em
  `legacy` até os lotes acabarem (ablação consistente); o gerador tem o parâmetro `newton_damping` para os próximos.
  Para o artigo: lote `batches/paper01_offshore_newton.json` (3 janelas × 4 variantes, 150 s cada, a partir dos
  estados de `ab1.000` gravados em `cases/paper01/newton/`) → `tab_newton` e parágrafo na §5.1.
- **Estabilidade linear do caso offshore:** `tools/offshore_linear.py` (continuação em abertura a partir do fim de
  `ab0.100`, espetro com a matriz de massa, participação sobre os dois dutos; grava também P_topo, α_G da linha e
  holdup do riser) → figura `linear_offshore` (crescimentos, período, P_in e q_L do estacionário contra as corridas)
  e `tab_linear` (nota com a_H, coluna do riser e o critério de Jansen et al. 1996 avaliado no estacionário em a_H).
  Arranca sozinha quando `ab0.100.npz` aparecer.
- **jaxdae (arXiv 2607.23202), a pedido do dono:** não adotar como integrador (sem projeção nos limites, tolerâncias
  por faixa, passo máximo nem paragem adaptativa; GMRES acima de n = 100; Newton de corda sem busca em linha;
  benchmarks só até n = 50). Fica como referência para o adjunto em grelha congelada do plano
  `docs/future/sensibilidades_jax.md`; estudo barato possível depois do artigo.
- **Lotes parados às 07:40 por decisão do dono:** outra sessão avalia tornar o padrão estratificado 0 % mal-posto, o
  que mudaria o modelo e obrigaria a refazer tudo; não valia a pena gastar mais ~8 h de máquina. Perderam-se (sem
  gravação parcial: a interrupção não é `Exception`) `ab0.100` (4 h 20 de relógio), `gor122.13` (4 h 20),
  `bubble_drag_thesis` e `unit_consistent` (~1 h 30 cada), malha fina (2 h 16) e numpy (1 h 39). Ficam gravadas:
  aberturas 1,0/0,6/0,3/0,2, RGO 30/50/80, ablação base/primitiva/sem viscosidade acústica/faces pela média/sem regra de
  fração baixa/sem massa virtual, malha grossa, JAX. Faltam no texto: abertura 0,1, RGO 122, resto da ablação, malha
  fina, tempo de relógio dos backends, secção da estabilidade linear, resumo e conclusões — a refazer com o modelo
  final, já com o Newton melhor (ver abaixo). O lote do Newton (leve) continuou.
- **Linha encurtada para 2 km (decisão do dono, 17/10 ~07:50):** o período da golfada severa é proporcional ao volume
  de gás da linha (bloqueio ≈ 3000 s dos ~4300 s com 6 km); com 1500 m horizontais + 500 m a −2° esperam-se ciclos de
  ~1400–1600 s, ainda bem dentro do critério de Bøe (limiar U_SL ≥ 0,12 m/s contra 0,34 m/s), 288 → 208 estados e
  ~2× menos passos por ciclo resolvido; o retrato offshore (poço com IPR, tie-back de 2 km em 300 m de lâmina de água,
  riser de 317 m, choke e separador) mantém-se. Gerador: `L_line=2000, L_down=500, n_h1=8, n_down=5`,
  `newton_damping="armijo"` por omissão; malhas grossa (4+3) e fina (16+10) reescaladas. Sonda de confirmação:
  scratch `probe_L2000` (4000 s, Armijo). Os casos do Newton (`cases/paper01/newton/`) são da geometria de 6 km e
  refazem-se a partir da nova `ab1.000`.
- **Sonda da linha de 2 km (09:00, ponto base, Armijo, 4000 s):** golfada severa confirmada — período ~1000–1250 s
  (sopros aos 480, 2226 e 3284 s), P_in 24,6–33 bar, surto 13–18 kg/s (5–6× a média de 2,5–3,2), topo do riser
  esvazia (holdup mínimo 0,59), selo até 284 m; 200 estados; 95 198 passos em 1580 s de relógio (um processo, máquina
  partilhada). **Achado de custo:** o bloqueio era barato e o sopro custa o mesmo seja qual for a linha — ~32 mil
  passos por ciclo nas duas geometrias; a linha curta dá 3× mais ciclos por segundo simulado e 30 % menos estados,
  não menos passos por ciclo. Campanha a 4000 s (dinâmica, RGO) e 3000 s (ablação, malhas) ≈ 1,9 M passos ≈ 3–4 h com
  4 processos. Alavancas por tentar: rtol 3e-3 → 1e-2 (~−30 %), ordem do BDF acima de 2.
- **Estado às 09:10:** casos e lotes regenerados a 2 km (`--tf 4000 --tf-short 3000`); resultados de 6 km arquivados
  em `runs/paper01/L6000/` (dinâmica, RGO, ablação, malha, backend, Newton); `runs/paper01/cache` apagado para
  recalcular. **Campanha à espera da decisão do estratificado (outra sessão), por decisão do dono.** O PDF versionado
  ainda mostra os resultados de 6 km com o texto já a 2 km — refaz-se com a campanha. Os casos do Newton
  (`cases/paper01/newton/`) são da geometria de 6 km: regenerar a partir da nova `ab1.000` antes de relançar o lote.
- **Armadilha descoberta às 09:15: a omissão `choke_opening=0.2` do gerador.** A sonda de 2 km correu com o choke a
  0,2 (não aberto), e **as malhas grossa e fina da noite também** (o `tools/offshore_batches.py` não passava a
  abertura nas malhas): a comparação de malhas do artigo misturava abertura 0,2 na malha grossa com o choke aberto
  na base — conclusão inválida, parágrafo passado a `\tbd`. Corrigido: omissão do gerador `choke_opening=1.0`,
  abertura explícita nas malhas, casos regenerados. A comparação justa da sonda é com a corrida de 6 km a 0,2
  (189 mil passos, ~3,5 ciclos parciais de 2100 s ≈ 54 mil por ciclo): a 2 km ~32 mil por ciclo (−40 %), período
  1000–1250 s e topo do riser a esvaziar.
- **BDF 100 % em JAX no problema-modelo (issue 056, 11:35, a pedido do dono):** protótipo
  `toy/model_problem_jaxbdf.py` reproduz passo a passo o `ImplicitBDF` (626/5285/11 640/43 587 passos, |Δy| ≤ 1,5e−4),
  3–5× mais rápido só porque o problema é pequeno (no offshore o laço em Python é < 5 % do custo); `jvp` através do
  laço adaptativo concorda com diferenças finitas (1 % Hopf, 0,004 % relaxação) a 4× uma integração; `grad` por
  reposição em grelha congelada desvia 16 % na relaxação. Não muda o simulador; o degrau 1 do plano de sensibilidades
  é o caminho.
- **Variantes do BDF no problema-modelo (issue 058, 12:10, a pedido do dono):** diagramas trabalho–precisão contra
  Radau 1e−10. A estimativa de erro de ordem consistente (Milne, preditor quadrático) poupa 20–50 % de tentativas a
  erro igual (o simulador controla o BDF2 como se fosse de 1.ª ordem: custo ∝ rtol^(−1/2)); ordens 3–5 ficam presas
  pela estabilidade (modos acústicos a 89,8°, fora de A(α)) — argumento para o BDF2 no artigo; H211b, crescimento e
  tolerância do Newton não mudam nada. Implementado em ficheiro à parte (`mfsim/integrators/milne.py`, 12:40) e medido nas janelas do Newton: com o
  Newton a partir do preditor linear, −45 % de passos e −55 % de Jacobianas no bloqueio, +5 % no sopro (limitado pelo
  Newton), ~−13 % por ciclo; o tampão do passo depois de falhas do Newton não compensa. Entra na campanha quando
  for opção JSON do `ImplicitBDF` (`implicit.py`/`options.py` em edição na outra sessão).
- **Campanha a 2 km lançada às 13:30 (decisões do dono):** ponto base com 054 (`regime.local_bubble`), 055
  (`newton_damping = armijo_tol`), limiar da regra de fração baixa 0,025 (zera o estratificado mal-posto, mesma física
  das outras regularizações segundo a outra sessão) e a estimativa de Milne (058, dobrada no `ImplicitBDF` como
  `integration.error_estimate`); tf 16 000 s na dinâmica e na RGO (ciclo limite de período 2, ~1150 + ~1800 s, depois
  de ~8000 s de transitório), 8000 s na ablação (16 variantes: + `no_local_bubble`, `newton_legacy`,
  `estimate_thesis`) e nas malhas, 2000 s nos backends. Cadeia `/tmp/campaign_L2000.sh`: dinâmica (5 processos) →
  RGO (3; gor50 copiado de ab1.000) → ablação (5) → malhas (2) → backends (1); logs em `runs/paper01/offshore_*.log`,
  `offshore_campaign.log` no fim. Depois: `tools/newton_windows_cases.py` (janelas do último ciclo da nova ab1.000) +
  lote do Newton, `tools/offshore_linear.py` a partir de ab0.100, figuras, texto, PDF.
- **Pendência de validação (outra sessão, 13:30):** o caso de laboratório de Fabre 2 com o padrão automático não
  chega ao primeiro ciclo — sem a 054 pára no passo mínimo aos 91 s, com a 054 rasteja até 96,5 s em 2 h (1,39 M
  passos); ambos emperram perto do primeiro sopro do riser a 1 atm. A 054 entra no ponto base do artigo por decisão
  do dono, mas **ainda não está validada com laboratório**; `batches/paper01_lab.json` fica fora da campanha até ao
  diagnóstico (registado na 057 e nas pendências do INDEX).
- **Passagem de texto a fazer com os resultados (lista para não esquecer):** §3.3 (l. 175, identificação de padrão:
  critério de bolhas com a fração local, 054) e Apêndice B (l. 488); §4.2 (l. 202–209, regra de fração baixa a 0,025 e a
  fração mal-posta por padrão: estratificado a 0 %); §5.1 (l. 245–247: a configuração passa a ter a busca em linha até
  à convergência e a estimativa de Milne; parágrafo e `tab_newton` refeitos com as janelas do novo ciclo e as variantes
  referência / sem busca / Armijo em todas / estimativa da tese / Jacobiana pela taxa / estado anterior); §5.3 (l. 262:
  16 variantes); §8 inteiro (ciclo de período 2, aberturas, RGO, estabilidade linear, malhas); conclusões (l. 402) e
  resumo; `tab_parameters`/`tab_analysis` conferir que leem as opções novas do caso.
- **14:25 — resultados da campanha a 2 km no texto (dinâmica, RGO, estabilidade linear, Newton):** ciclo limite de
  período 2 com o choke aberto (sopros de 33 e 45 kg/s alternados, 14,7–31,7 bar, duplo período ~3000 s, riser a
  esvaziar), 0,6 regular (1240 s), 0,3 e 0,2 ciclo de pressão com pulsos (730 s), 0,1 estável a 33 bar; estacionários
  por continuação (3,5 s de Newton por abertura): par «coluna» (~70 s) cruza em a_H = 0,13, par «troca de inventário»
  (545–965 s) em 0,19, modo real acima de 0,4 (+0,046 s⁻¹ aberto); queda no choke em a_H 3,65 bar = Jansen 3,5 bar.
  RGO: 50/80/122 iguais (riser esvazia, 15–31 bar, 40–45 kg/s; período 1530/1540/1180 s), RGO 30 ciclo de pressão com
  pulsos. Janelas do Newton (400 s): busca em linha até à convergência −28 a −43 % de passos; Armijo em todas as
  iterações falha no sopro (055); Milne neutro aqui (−1 a +7 %); `jac_rate` −46 % de passos no sopro por 2,4× as
  Jacobianas. Tempos: compilação JAX cronometrada à parte (`info["compile"]`), `tab_cost` nova. Transitório da
  partida bloqueada: 5 ciclos de período decrescente até ~8600 s; figuras desde t = 0 e último ciclo completo.
  Faltam: ablação (16, a correr), malhas, backends, §4 (hiperbolicidade com low_void 0,025), resumo e conclusões.
- **16:15 — campanha completa e artigo sem `\tbd`:** ablação com 19 variantes (unidade de golfada da tese falha antes do
  primeiro ciclo nas duas versões; critério de bolhas pela vazão 7,7× os passos e outro ciclo; faces pela média 7,6×
  e outro ciclo; primitiva cria 0,5 % do líquido; Armijo em todas as iterações falha aos 2040 s; Milne +19 % de
  passos no ciclo completo — neutro nas janelas, ganho só no problema-modelo; Bestion −16 % sem base física;
  low_void 0,01 = 0,025 em custo; tolerância uniforme 5,8×; BDF1 14×), malhas (faixa de pressão igual, período e surto
  ±15 % entre grossa e fina: não convergidos na malha base), backends (JAX 7× mais rápido; compilação 14–31 s fria,
  2 s com cache). Overrides com `null` removem a chave (uniform_atol); `compile` gravado na info das corridas.
  Rotina 410 passed. **Decisão pendente do dono:** manter `error_estimate = milne` por omissão no gerador (custa
  +19 % no ciclo offshore; ganha 20–50 % no problema-modelo) ou voltar a `thesis`.
- **Pendência de texto (16:25):** as afirmações da §5.2 sobre os esquemas colocados (Galerkin, diferenças) vêm do
  estudo de discretização do caso à escala piloto (`runs/paper01/discretization`, 15 bar) e não foram refeitas a 2 km;
  ficam como observações qualitativas sem números. Refazer exigiria um lote de discretização para o caso offshore.
- **Órbitas periódicas por shooting (17:00, pedido do dono: figura do ramo de ciclos pelo método do artigo 03):**
  implementada a sensibilidade ao estado inicial no BDF (059, testada no problema-modelo e no hello_world) e a
  ferramenta `tools/periodic_orbit.py`. **No caso offshore a monodromia não é fiável:** o mapa de um período é áspero
  a todas as escalas (frentes a atravessar células, chaveamentos, projeções) — derivadas direcionais por diferenças
  finitas inconsistentes entre ε de 0,1 % a 10 %, multiplicador trivial e M·v errados por 50 % a 65×, sem padrão com a
  tolerância; o primeiro passo do Newton produz um estado que o integrador não arranca. Decisão do dono: **varrimento
  denso em abertura** (lote `paper01_offshore_sweep`, 12 corridas de 16 000 s) e a figura `cycle_branch` (§8.6) com o
  ramo de ciclos observado; os ciclos instáveis ficam para o artigo 03 (colocação da órbita ou mapa suavizado).
- **Quando se refizer a campanha:** `newton_damping=armijo` como omissão do gerador (o estado anterior como iterado
  inicial só ganha no bloqueio, −63 %, e não soma à busca em linha: 295 contra 249 passos); a variante `closure_mean` falhou por passo mínimo aos 3774 s (31 % de rejeições) — rerodar com o
  Newton melhor antes de afirmar que "as faces pela média falham no sopro".

## Exemplo base (anterior, escala piloto; substituído)

Caso de referência como poço (`cases/severe_slug/well_choke.json`, issue 043) com **padrão de escoamento
dinâmico**: linha horizontal de 20 m, descendente de 300 m a −3°, curva em arco (R = 5 m, 4 elementos),
riser de 20 m e 5 m horizontais até ao choke; D = 50 mm; IPR linear na entrada (0,056 kg/s/bar, razão
gás–líquido da referência), choke homogéneo CV 15 a descarregar no separador a 4 bar; líquido 900 kg/m³
(γL = 100), gás politrópico (γG = 1,05). Modelo a dois fluidos com identificação automática suave e
local, regra de fração baixa (bolhas abaixo de 1 %), massa virtual (C_vm = 0,5, forma objetiva), arrasto
físico de bolhas; esquema escalonado conservativo com alvo `noslip`, viscosidade artificial 0,05;
BDF próprio com `tol_range` 0,01 e lado direito em JAX; fechamentos das faces pela célula doadora (044) e ponte do filme local (046). Pressão do reservatório de referência 15 bar. Ficheiro: `cases/severe_slug/well_choke_auto.json`.

## Casos de simulação (lotes)

Todos no ponto base offshore (`cases/paper01/offshore.json`: 200 bar, PI 5 Sm³/d/bar, RGO 50, partida bloqueada),
gerados por `tools/offshore_batches.py`; os lotes anteriores (escala piloto, 15 bar) ficaram em `runs/paper01/{dynamics,
discretization,ablation,mesh,backend}` e já não entram no artigo.

| lote | o quê | saída |
|---|---|---|
| `batches/paper01_offshore_dynamics.json` | aberturas 1,0, 0,6, 0,3, 0,2 e 0,1; 8000 s, saída a 2 s | `runs/paper01/offshore_dynamics/` |
| `batches/paper01_offshore_gor.json` | RGO 30, 50, 80 e 122,13 com o choke aberto; 8000 s | `runs/paper01/offshore_gor/` |
| `batches/paper01_offshore_ablation.json` | caso base menos um ingrediente de cada vez; 4000 s (primitiva e regra de fração baixa relançadas em mini-lotes) | `runs/paper01/offshore_ablation/` |
| `batches/paper01_offshore_mesh.json` | malha grossa e fina; 4000 s | `runs/paper01/offshore_mesh/` |
| `batches/paper01_offshore_backend.json` | JAX × numpy; 2000 s | `runs/paper01/offshore_backend/` |
| `batches/paper01_offshore_newton.json` | Newton do BDF em três janelas de 150 s (053) | `runs/paper01/offshore_newton/` |
| `tools/offshore_linear.py` | estacionários e espetro ao longo da abertura (não integra no tempo) | `runs/paper01/offshore_linear/hopf.npz` |
| `toy/model_problem.py run` | problema-modelo (secção 2): linha de transmissão + terminação de resistência negativa, 4 valores de a | `runs/paper01/toy/` |

## Decisões em aberto para o dono

- **Compressibilidade do líquido (γL) — resolvido pela mudança de caso:** o caso offshore usa o fluido da Tabela 6 com γL = 1000. (Histórico:) O exemplo anterior usava γL = 100 (c ≈ 258 m/s, ~40× mais compressível que um
  líquido real). A ablação com γL = 1000 dá o mesmo ciclo dentro da variabilidade (5 blowdowns, P_in 4,56–5,85 bar)
  com 45 % menos passos (145 mil contra 264 mil): o BDF amortece os acústicos mais rápidos em vez de os resolver.
  O rascunho dizia o contrário e foi corrigido. Mudar o caso base para um γL realista obriga a refazer todas as
  corridas (dias de CPU); alternativa: manter 100 e reportar a ablação.

- Revista (proposta: *International Journal of Multiphase Flow*; ver `../00-escolha-da-revista.md`).
- Autores: Jaime Neiva Miranda de Souza (Miradouro Digital) e André Luiz Hemerly Costa (UERJ, autor correspondente), decididos pelo dono em 2026-10-16 (Medeiros e Nunes saíram). E-mail do correspondente: andrehc@uerj.br. Falta a morada da Miradouro Digital.
- Fechamentos das faces pela célula doadora (044) como padrão do caso do artigo.
- Limiar da regra de fração baixa (1 % mantido; 10 % é a alternativa estudada em 042).
