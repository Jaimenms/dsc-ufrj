# Índice de alterações ao modelo e ao solver da tese

*Documentação do mfsim (índice geral no [README](../../README.md)).*

Cada entrada é uma alteração atómica em relação ao MFSIM da tese, com um ficheiro próprio
nesta pasta que diz **como era antes, como ficou depois e que benefício se mediu**, com os
dados e as opções do caso JSON que a ativam. Tipos:

- **feature**: capacidade nova ou opção nova (a da tese continua disponível quando faz sentido);
- **bugfix**: correção de algo errado na tese, nos ficheiros originais ou numa versão anterior daqui;
- **estudo**: sensibilidade ou verificação sem alteração do modelo, registada porque muda a
  leitura dos resultados.

Os números de cada entrada estão em [severe_slug.md](../severe_slug.md),
[lab_validation.md](../lab_validation.md), [jax_backend.md](../jax_backend.md) e
[thesis_reproduction.md](../thesis_reproduction.md); a infraestrutura (casos JSON, corridas,
referências de regressão) está em [data_formats.md](../data_formats.md) e no README.

A síntese do que impedia a tese de simular a intermitência severa está em
[why_no_severe_slug_in_thesis.md](../why_no_severe_slug_in_thesis.md).

## Notebooks

Cada grupo tem um notebook em `notebooks_src/issues/` que lê as corridas gravadas e desenha as
figuras (`uv run mfsim nb notebooks_src/issues/<nome>.py --execute`; os `.ipynb` executados ficam
em `notebooks/issues/`): `nb_reproducao_capitulo10` (001–002), `nb_esquema_e_conservacao`
(002–005, 027), `nb_joelho_malha_acustica` (006–011), `nb_fechamentos_hiperbolicidade` (015–019),
`nb_identificacao_de_padrao` (020–021), `nb_fronteira_riser_gas_gotas` (022–024, 026, 028–029),
`nb_laboratorio` (030–031), `nb_integrador_e_jax` (032–038), `nb_rede_refluxo_deriva` (012–014,
025) e `nb_gas_lift_intermitencia` (033, 034, 036, caso além da tese). Os casos da tese tal e
qual estão em `notebooks_src/thesis/`.

## Reprodução da tese

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [001](001-reproducao-capitulo-10.md) | estudo | O que os ficheiros e o código originais escondiam | CI gravada = estado final; "upwind" = regressivas em tudo; estacionário discreto não único → casos reexecutáveis, fechamentos a 1e−14 |

## Esquema numérico do duto

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [002](002-fv-upwind.md) | feature | Volumes finitos escalonados com upwind pela velocidade de fase | esquemas colocados sem severe slug → primeiro ciclo-limite (18 ciclos em 8000 s) |
| [003](003-fase-que-desaparece.md) | feature | Tratamento da fase que desaparece | limitadores destruíam massa (ciclos falsos) → balanço verificável, fase rápida converge |
| [004](004-formulacao-conservativa.md) | feature | Formulação conservativa (BDF nas massas por volume) | −43 kg (2.7 %) de líquido por corrida → erro < 1e−9, mesma dinâmica |
| [005](005-acumulador-bdf.md) | feature | Acumulador de balanço consistente com o BDF | trapézios nas saídas (−1.7 kg de erro de quadratura) → medida exata |
| [006](006-arrasto-espurio-fase-virtual.md) | bugfix | Arrasto espúrio da fase virtual (`relax_target = noslip`) | coluna de líquido travava a 0.49 kg/s → 3.65 (= monofásico); picos 7.9 → 7.1 kg/s |
| [007](007-viscosidade-artificial.md) | feature | Viscosidade artificial acústica | modo do tampão a 16–20 Hz, malha fina inviável → −24–32 % de passos, fina viável |
| [008](008-joelho-em-arco.md) | feature | Joelho em arco e mistura por ângulo | ciclo amortecido (5.7–6.4 bar, 240 s) → blowdown "de livro" (4.2–6.1 bar, 660–700 s) |
| [009](009-malha-graduada.md) | feature | Malha graduada por trecho | só uniforme → resolução da fina no pescoço por metade do custo |
| [010](010-reconstrucao-muscl.md) | feature | Reconstrução MUSCL nas faces | hipótese de frente difusa por α → sem efeito (difusão vem do momento) |
| [011](011-compressibilidade-artificial.md) | feature | Compressibilidade artificial do líquido (γL = 100) | rigidez acústica do líquido → corridas longas viáveis, dinâmica lenta igual |
| [012](012-vf-escalonado-deriva.md) | feature | Volumes finitos escalonados para o modelo de deriva | só esquema nodal → 10.3.1 escalonado e conservativo; severe slug com B&B não é simulável (físico) |
| [013](013-filtro-anti-refluxo.md) | bugfix | Filtro anti-refluxo removido | remendo nas saídas marcadas `check` → equações puras; 10.2.2 inalterado |
| [014](014-inversao-de-sentido.md) | feature | Inversão de sentido com fração imposta nas extremidades | fração livre em refluxo → fantasma doador; inversão enche o duto com o líquido do separador |

## Fechamentos e modelo físico

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [015](015-fechamento-de-bolhas.md) | bugfix | Fechamento de bolhas da tese (vb sem g e σ; força ∝ ρG αG²) | nenhum caso de laboratório cicla → Fabre 2: 135.8 s contra 137 medidos |
| [016](016-fracao-alta-anular.md) | feature | Fração alta → anular (contracorrente no riser, item 1) | filme nunca drena (bolhas a 95 %) → filme drena 56–66 % do tempo, intervalo +10 %, mais barato |
| [017](017-fracao-baixa-bolhas-bestion.md) | feature | Bolhas abaixo de 1 % e pressão interfacial de Bestion | estratificado mal posto em αG ≤ 3e−3 → mapa hiperbólico |
| [018](018-massa-virtual.md) | feature | Massa virtual de Drew & Lahey (item 3) | sem inércia da fase dispersa → mapa hiperbólico e intervalo +7–10 % (efeito físico) |
| [019](019-atrito-interfacial-estratificado.md) | feature | Atrito interfacial do estratificado parametrizado (item 2) | fixo → ×0.5 / ×2 / Taitel & Dukler: ciclo insensível (±2 %) |
| [020](020-ipe-da-tese-defeitos.md) | bugfix | Identificação de padrão da tese: quatro defeitos | 9834 passos até 1.5 s → partição da unidade, graus, NaN tratados |
| [021](021-ipe-suave-e-local.md) | feature | Identificação automática suave e local | inutilizável em dinâmica → ciclo reproduzido a 35k passos / 207 s |
| [022](022-riser-intermitente-vs-bolhas.md) | estudo | Riser intermitente contra bolhas no blowdown (item 5) | caso base só cicla com bolhas; linha longa cicla com ambos (±10 %) |
| [023](023-lei-do-gas.md) | estudo | Lei do gás γG (item 4) | 1.05 → 1.3 encurta o intervalo 9–18 %; amplitude igual |
| [024](024-arraste-de-gotas.md) | estudo | Arraste de gotas no anular (item 7) | sem modelo → desprezível (1–5 % do líquido em picos de segundos) |

## Rede, condições de contorno e validação

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [025](025-vertices-conservativos.md) | bugfix | Vértice dinâmico da tese é sumidouro de massa | 741 de 1590 kg absorvidos → balanço exato, passos 8× menores |
| [026](026-volume-de-gas-a-montante.md) | feature | Volume de gás a montante (item 6) | vazões constantes → tanque de mistura (não representa gasoduto); linha de entrada longa sim (+29 % / +35 %) |
| [027](027-condicoes-iniciais-por-receita.md) | feature | Condições iniciais por receita (bloqueada, linha fechada) | primeiro blowdown aos 2500 s → aos 12 s; CI reprodutível no JSON |
| [028](028-criterio-de-boe.md) | bugfix | Critério de Bøe corrigido | L/sin|θ| (19× maior) → forma das fontes; caso base dentro da região |
| [029](029-fronteira-da-intermitencia-severa.md) | estudo | Fronteira da intermitência severa | varredura: ciclo em toda a região estratificada; permanente quando a linha deixa de o ser |
| [030](030-validacao-laboratorio.md) | feature | Validação com laboratório (item 8) | sem dados medidos → Fabre 2, Tengesdal, Jansen como ciclo clássico com o arrasto físico |
| [031](031-numerica-a-1-atm.md) | bugfix | Numérica a 1 atm | banda de undershoot cruzava a singularidade; Newton estagnado → `state_bounds` com γL, `slow_exit` |
| [039](039-lei-de-valvula-homogenea.md) | bugfix | Lei de válvula homogénea | densidade ponderada por massa (choke não resiste ao gás; singular em refluxo, 7 falhas) → por volume, padrão; tese só nos seus casos |
| [040](040-partida-do-estacionario.md) | feature | Partida a partir do estado estacionário | só condições iniciais ou estimativa (Newton não converge dela) → `initial_state.source = "steady"` a partir de um estado gravado, com perturbação |
| [041](041-paragem-adaptativa.md) | feature | Tempo simulado adaptativo | `tf` fixo (curto para os lentos, longo para os que assentam) → janelas com paragem quando o sinal fica permanente ou o ciclo se repete |
| [042](042-modos-locais-da-curva.md) | estudo | Modos locais instáveis na curva | hipótese: troca de fechamentos → não; instabilidade de ondas curtas no ponto baixo, cresce com a malha; limiar de bolhas a 10 % reduz-na (resultado misto na validação) |
| [043](043-mapa-de-intermitencia-severa.md) | feature | Mapa de intermitência severa (abertura × pressão do reservatório) | 10 pontos sem choke nem poço → 2 × 126 corridas em paralelo, sem falhas; fronteira = critério de Jansen; histerese em 8 células |

## Integrador e desempenho

| n.º | tipo | título | antes → depois |
|---|---|---|---|
| [032](032-bdf-implicito-proprio.md) | feature | BDF implícito próprio | `ode15s` fechado → ordem 1–2, projeção, breakpoints, Newton com orçamento, parcial em falha |
| [033](033-slow-exit-regressao.md) | bugfix | `slow_exit` desligado quebrava o gas lift | evento da retenção falhava → padrão ligado; severe slug de campo desliga explicitamente |
| [034](034-tolerancia-por-faixa.md) | feature | Tolerância por faixa (`tol_range`, `state_ref`) | atol único ou ajustado à mão → 25.9k passos / 50 s sem ajuste; gas lift 13.7 → 6 s |
| [035](035-config-rapida.md) | feature | Tolerâncias absolutas por variável e orçamento de Newton | 85.7k passos / 20 min → 23.2k / 5 min, mesmo ciclo |
| [036](036-lado-direito-jax.md) | feature | Lado direito compilado e Jacobiana exata (JAX) | 5 min → 45 s (grossa), 58 min → 105 s (fina); paridade 1e−14 |
| [037](037-tangentes-nan.md) | bugfix | Tangentes NaN na diferenciação automática | "Factor is exactly singular" → Jacobiana finita em repouso e no corte das sigmoides |
| [038](038-derivadas-beggs-brill.md) | feature | Derivadas do Beggs & Brill exatas (jvp) | diferenças finitas a 1e−3 → exatas; concordam a 1e−4 fora das quinas |

## Pendências abertas

- Gas lift na base como mitigação (item 9; o choke no topo está no mapa, 043); fecha a comparação com Jansen (030).
- Vértice do tipo separador (fases separadas, fase certa em refluxo) (014, 026).
- Fechamento de bolhas entre 10 e 30 % de gás (022).
