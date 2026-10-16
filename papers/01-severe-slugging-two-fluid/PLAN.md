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
9. Os autovalores do estacionário não desenham a fronteira da golfada (histerese, modos locais rápidos):
   o mapa faz-se por simulação dinâmica com duas partidas.

## Exemplo base

Caso de referência como poço (`cases/severe_slug/well_choke.json`, issue 043) com **padrão de escoamento
dinâmico**: linha horizontal de 20 m, descendente de 300 m a −3°, curva em arco (R = 5 m, 4 elementos),
riser de 20 m e 5 m horizontais até ao choke; D = 50 mm; IPR linear na entrada (0,056 kg/s/bar, razão
gás–líquido da referência), choke homogéneo CV 15 a descarregar no separador a 4 bar; líquido 900 kg/m³
(γL = 100), gás politrópico (γG = 1,05). Modelo a dois fluidos com identificação automática suave e
local, regra de fração baixa (bolhas abaixo de 1 %), massa virtual (C_vm = 0,5, forma objetiva), arrasto
físico de bolhas; esquema escalonado conservativo com alvo `noslip`, viscosidade artificial 0,05;
BDF próprio com `tol_range` 0,01 e lado direito em JAX. Pressão do reservatório de referência 15 bar.

## Casos de simulação (lotes)

| lote | o quê | saída |
|---|---|---|
| mapa (`tools/slug_map.py build --base ...`) | 14 aberturas × 9 pressões × 2 partidas com padrão dinâmico | `runs/maps/well_auto/` |
| `batches/paper01_dynamics.json` | 15 bar: aberturas estável, ciclo de pressão, golfada severa e aberta, saída a 1 s | `runs/paper01/dynamics/` |
| `batches/paper01_discretization.json` | o mesmo caso com Galerkin, Petrov-Galerkin, diferenças centrais, regressivas, escalonado primitivo e conservativo | `runs/paper01/discretization/` |
| `batches/paper01_ablation.json` | caso base menos um ingrediente de cada vez | `runs/paper01/ablation/` |
| `batches/paper01_mesh.json` | malha grossa, graduada e fina | `runs/paper01/mesh/` |
| `batches/paper01_backend.json` | numpy × JAX | `runs/paper01/backend/` |

## Decisões em aberto para o dono

- Revista (proposta: *International Journal of Multiphase Flow*; ver `../00-escolha-da-revista.md`).
- Autores e afiliações (no esqueleto: o autor da tese e os orientadores, a confirmar).
- Fechamentos das faces pela célula doadora (044) como padrão do caso do artigo.
- Limiar da regra de fração baixa (1 % mantido; 10 % é a alternativa estudada em 042).
