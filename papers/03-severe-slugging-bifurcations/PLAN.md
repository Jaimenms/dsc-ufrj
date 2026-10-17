# Artigo 03 — plano: bifurcações da golfada severa por continuação de órbitas periódicas

*Documento de trabalho do artigo (português, como o resto da documentação; o manuscrito é em inglês).
Escrito em 2026-10-17 a partir da discussão com o dono sobre o que o estacionário e os autovalores dizem (e não
dizem) da golfada severa; atualizar à medida que as decisões e as corridas chegam.*

## O pedido, em uma frase

Calcular diretamente os ciclos limite do modelo a dois fluidos do artigo 01 (órbitas periódicas: um estado x₀ e um
período T com φ_T(x₀) = x₀), com Newton–Krylov e derivação automática, e segui-los em função dos parâmetros
(abertura do choke, pressão do reservatório, RGO) para obter o diagrama de bifurcação completo da golfada severa:
ramos estáveis e instáveis, natureza do Hopf, transição para a relaxação, duplicação de período e histerese.

## Porque é um artigo próprio

A literatura de golfada severa tem sobretudo integrações no tempo, critérios de estabilidade de regime permanente
(Bøe, Taitel, Jansen) e, mais recentemente, análises lineares do estacionário. Um diagrama de bifurcação de um
modelo a dois fluidos completo, com os ciclos instáveis e os multiplicadores de Floquet, responde a perguntas que
nem a integração longa nem os autovalores do estacionário respondem.

## O que já se sabe (caso de 2 km do artigo 01, 2026-10-17)

Espetro com a matriz de massa (048) ao longo da abertura e dinâmicas de 16 000 s da campanha do artigo 01:

| abertura | espetro do estacionário | período linear 2π/Im λ | dinâmica: período | dinâmica: amplitude de P |
|---|---|---|---|---|
| 0,10–0,12 | estável | — | estável | 0 |
| 0,15–0,20 | Hopf, par rápido (68–71 s) à frente; par lento (546 s) quase neutro a 0,20 | 68 / 546 s | ~720 s (0,20) | 0,8 bar |
| 0,25–0,30 | Hopf, par lento | 509–569 s | ~730 s (0,30) | 2,8 bar |
| 0,40 | Hopf, par lento quase a colidir no eixo real | 1434 s | — | — |
| 0,60–1,00 | real (relaxação, sem frequência linear) | — | 1238 s (0,60); período 2 a 1,00 (1394 / 1732 / 3012 s) | 8,9 / 16,9 bar |

O mapa da issue 043 (padrão fixo) teve histerese real em 2 células, o que sugere um Hopf subcrítico em parte da
fronteira.

## Perguntas do artigo

1. **Natureza do Hopf** ao longo da fronteira: supercrítico (o ciclo nasce pequeno) ou subcrítico (nasce grande,
   com um ciclo instável a separar o estacionário da golfada: histerese).
2. **Da oscilação de pressão à relaxação:** como o ciclo de ~720 s e amplitude ~1 bar a 0,2–0,3 vira a golfada de
   ~1200–1500 s e 9–17 bar quando o par colide no eixo real; continuamente ou por uma dobra.
3. **Duplicação de período** no choke aberto (alternância de ciclos de ~1150 e ~1800 s): onde acontece, e se há
   uma cascata (caminho para dinâmica caótica).
4. **Interação dos dois pares** (o rápido de ~70 s e o lento) perto de 0,20: ponto de Hopf duplo, toros.
5. **Diagrama a dois parâmetros:** curvas de Hopf, de dobra e de duplicação de período no plano abertura × P_res (ou
   abertura × RGO), comparadas com o mapa dinâmico do artigo 01 e com os critérios de Bøe e de Jansen.
6. **Sensibilidades do ciclo:** derivadas do período, da amplitude, do pico de líquido e da produção média em
   relação aos parâmetros, pelo teorema da função implícita sobre a órbita.

## Método

### Formulação

- Incógnitas: x₀ (os estados do caso; 200 no de 2 km) e T.
- Equações: φ_T(x₀) − x₀ = 0 e uma condição de fase (por exemplo, a pressão na base do riser em x₀ igual à média do
  ciclo anterior e a subir, ou a condição integral de Doedel).
- Newton:
  `[M − I, f(φ_T(x₀)); cᵀ, 0] [δx₀; δT] = −[φ_T(x₀) − x₀; 0]`, com M = ∂φ_T/∂x₀ a matriz de monodromia.

### Produtos M·v sem montar M

- **Sensibilidade ao estado inicial propagada dentro do BDF** (degrau 1 de
  [docs/future/sensibilidades_jax.md](../../docs/future/sensibilidades_jax.md)): em cada passo aceite, a perturbação
  resolve (J_T − γJ) δx_{n+1} = (recorrência BDF das perturbações anteriores), com a LU que o Newton já tem, ou seja,
  uma substituição por direção e por passo. Reproduz a derivada da trajetória discreta que o integrador calcula, sem
  diferenciar o Newton nem o controlo de passo. É uma opção nova do `ImplicitBDF` (issue própria).
- **GMRES** sobre M − I: o sistema é muito dissipativo e quase todos os multiplicadores de Floquet são ~0, logo
  M − I ≈ −I + posto baixo e bastam ~10–30 produtos por iteração (ideia do método de Newton–Picard de Lust et al.,
  1998, a verificar na bibliografia). Com várias direções por integração (bloco), cada integração dá vários
  produtos.
- Alternativa estudada na 056 (BDF inteiro em JAX, jvp através do laço adaptativo): concorda com diferenças finitas
  a 1 % no problema-modelo, mas exigiria reescrever o integrador do simulador; fica como referência.

### *Shooting* múltiplo

Para as fases de sopro, em que perturbações crescem muito num ciclo, partir o período em 5–20 troços, cada um com o
seu estado inicial como incógnita e continuidade entre troços.

### Continuação

- Partida no ponto de Hopf: x₀ ≈ estacionário + ε·Re(v), T ≈ 2π/Im λ, com v o vetor próprio do par crítico (048).
- Partida alternativa: um estado do ciclo de uma corrida dinâmica no mesmo ponto, com o período medido (é a melhor
  estimativa no regime de relaxação).
- Continuação por comprimento de arco (Keller) no parâmetro, para passar as dobras.
- Multiplicadores de Floquet por Arnoldi sobre M (os mesmos produtos M·v): dobra (multiplicador +1), duplicação de
  período (−1), Neimark–Sacker (par complexo no círculo unitário). Na duplicação, troca para o ramo de período 2T.
- Continuação a dois parâmetros dos pontos de Hopf (sobre o estacionário, barata), de dobra e de duplicação.

## Verificação, por esta ordem

1. **Problema-modelo do artigo 01** (`papers/01-severe-slugging-two-fluid/toy/model_problem.py`, com casos de Hopf e
   de relaxação): órbita por *shooting* contra a integração longa; multiplicadores contra diferenças finitas.
2. **Caso de 2 km em pontos estáveis do ciclo** (0,30, 0,60 e 1,00): período, amplitude e pico de líquido contra as
   corridas de 16 000 s.
3. **Convergência numérica:** tolerância do BDF, malha (grossa e fina do artigo 01), número de troços.

## Custo, em ordem de grandeza

| | custo |
|---|---|
| um ciclo de ~1500 s com o modelo do artigo 01 | ~1 min |
| uma iteração de Newton (1 resíduo + 10–30 produtos M·v) | ~10–30 min |
| uma órbita (3–6 iterações, a partir de uma dinâmica) | ~1–3 h |
| um diagrama a um parâmetro (~20 pontos) | ~1 dia de máquina |

## Riscos e planos alternativos

- **Mapa pouco suave** (frentes, chaveamentos de padrão): bacia de convergência estreita. Partir sempre de corridas
  dinâmicas, *shooting* múltiplo e passos de continuação curtos.
- **Derivada através do passo adaptativo:** a monodromia da trajetória discreta pode afastar-se da do contínuo nas
  fases rápidas (na 056, a reposição em grelha congelada desviou 16 % na relaxação). O Newton tolera uma Jacobiana
  aproximada; os multiplicadores verificam-se contra diferenças finitas em pontos escolhidos.
- **Relaxação com período 2 ou caótica no choke aberto:** se não houver órbita periódica simples, o artigo mostra a
  cascata até onde o método a segue e caracteriza o resto com a integração longa (secções de Poincaré).
- **Plano B:** diagrama de bifurcação a um parâmetro com os ramos estáveis por integração longa e os instáveis por
  *shooting* só onde a histerese o exige.

## Ligações aos outros artigos

- Depende do modelo do artigo 01 congelado (054, 055, 058, vazio baixo 0,025).
- A propagação da sensibilidade no BDF serve também ao artigo 02: gradientes exatos das grandezas do ciclo em
  relação às decisões (choke, gas lift), para validar o substituto e o ótimo junto da fronteira.

## Etapas

0. Artigo 01 fechado e modelo congelado.
1. Sensibilidade ao estado inicial no `ImplicitBDF` (opção nova; issue), verificada no problema-modelo.
2. Ferramenta de órbitas (`tools/periodic_orbit.py` ou `mfsim.analysis`): *shooting* simples e múltiplo, GMRES,
   condição de fase, Floquet por Arnoldi; testes no problema-modelo.
3. Continuação a um parâmetro (abertura) no caso de 2 km, a partir do Hopf e das dinâmicas da campanha.
4. Deteção e troca de ramo nas duplicações de período; continuação a dois parâmetros dos pontos especiais.
5. Sensibilidades do ciclo; figuras (diagramas, multiplicadores, retratos de fase) só a partir de corridas e caches;
   manuscrito.

## Decisões em aberto para o dono

- Segundo parâmetro do diagrama a dois parâmetros: P_res, RGO ou pressão do separador.
- Se o artigo inclui o caso de laboratório (Fabre, Jansen) como segundo sistema, depois de resolvida a pendência
  do padrão automático a 1 atm (057).
- Revista alvo (dinâmica não linear aplicada ou escoamento multifásico) e autores.

## Bibliografia a levantar

Sem chaves inventadas: cada referência entra em `refs.bib` (ou em `../_common/references.bib`) só depois de o DOI
estar verificado. Temas: continuação de órbitas periódicas (Keller; Doedel e o AUTO; MatCont); Newton–Picard e
*shooting* para sistemas grandes e dissipativos (Lust et al., 1998, a verificar); sensibilidades em integradores BDF
(sensibilidade direta interna); bifurcações em escoamento multifásico e golfada severa (critérios de Bøe, Taitel e
Jansen, já na bibliografia comum).
