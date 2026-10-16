# Escolha da revista para os artigos do mfsim

*Pesquisa feita em 2026-10-16 para o artigo 01 (intermitência severa com o modelo a dois fluidos a quatro
equações e identificação dinâmica do padrão de escoamento: boa colocação do modelo, discretizações
testadas, integrador implícito, backend compilado, perfis e mapa de estabilidade com choke). O método e o
formato vêm do `research-factory` (`01-multiphase-compositional-flow/papers/00-escolha-da-revista.md`, que
escolheu a Geoenergy Science and Engineering para o tema composicional).*

## Decisão proposta (a confirmar pelo dono)

**International Journal of Multiphase Flow** (Elsevier; ISSN 0301-9322). Razões:

1. **É onde está a conversa que o artigo continua.** Os trabalhos de referência do tema saíram aí: o critério
   de estabilidade com choke de Jansen, Shoham e Taitel (1996), os modelos de intermitência severa de Taitel
   (1990) e Sarica e Shoham (1991), a captura de golfadas com o modelo a dois fluidos de Issa e Kempf (2003)
   e boa parte da literatura sobre hiperbolicidade e regularização do modelo a dois fluidos (Stuhmiller,
   Bestion, Holmås, Lopez de Bertodano e colaboradores). A contribuição — o que é preciso para o modelo a
   quatro equações com mapa de padrão dinâmico simular o ciclo e um mapa de estabilidade — é de modelação e
   numérica do escoamento multifásico, o núcleo do âmbito da revista.
2. **Os temas numéricos cabem.** Discretização, conservação, boa colocação e integração implícita de modelos
   a dois fluidos são assunto recorrente da revista; o desempenho com compilação e diferenciação automática
   entra como meio, não como fim.
3. **Formato.** Elsevier, LaTeX aceite com as classes da Elsevier (o modelo CAS `cas-sc` do esqueleto),
   citações autor–ano, resumo ≤ 250 palavras, 3–5 *highlights* ≤ 85 caracteres em ficheiro à parte.
   O *Express Track* exige `elsarticle` e 6 páginas; não é o caso deste artigo (artigo completo).

## Candidatas comparadas

| revista | âmbito e semelhança com o artigo 01 | observações |
|---|---|---|
| **International Journal of Multiphase Flow** | modelação fundamental e numérica de escoamentos multifásicos; intermitência severa e boa colocação do modelo a dois fluidos | **proposta**; exigente na física do escoamento |
| Chemical Engineering Science | escoamento multifásico e dinâmica de processos; aceita mapas de estabilidade e não-linearidade | alternativa se o artigo puser o mapa e a histerese à frente |
| Computers & Chemical Engineering | métodos numéricos, diferenciação automática, sistemas de processo | encaixa se a contribuição for vendida como método numérico (discretização + integrador + JAX) |
| Geoenergy Science and Engineering (ex-JPSE) | produção de hidrocarbonetos, garantia de escoamento | escolhida no research-factory para o tema composicional; boa para uma versão aplicada (mitigação com choke) |
| SPE Journal | engenharia de petróleo, intermitência severa | sem modelo LaTeX oficial |

## Guia de autores (o que o esqueleto já cumpre)

Lido em 2026-10-16 (a página do ScienceDirect devolve 403 a leitura automática; os pontos abaixo vêm da
pesquisa sobre a própria página do guia):

- Resumo ≤ 250 palavras, autocontido, sem referências nem abreviaturas não definidas.
- *Highlights*: 3 a 5 pontos, ≤ 85 caracteres cada (com espaços), ficheiro editável à parte com
  «highlights» no nome.
- Palavras-chave, secções numeradas, SI, figuras vetoriais com fontes embebidas, cores seguras para
  daltonismo, tabelas editáveis sem linhas verticais.
- Declarações na submissão: interesses concorrentes, uso de IA generativa, disponibilidade dos dados,
  financiamento (no research-factory o dono tirou o CRediT dos manuscritos; segue-se a mesma prática).
- Confirmar no guia oficial, antes de submeter, o limite de páginas e se pede *graphical abstract*.

## Fontes

- Guia de autores: https://www.sciencedirect.com/journal/international-journal-of-multiphase-flow/publish/guide-for-authors
- Página da revista: https://www.sciencedirect.com/journal/international-journal-of-multiphase-flow
- *Express Track* (elsarticle, 6 páginas): https://api.journals.elsevier.com/media/wgsb1gsk/multiphaseflowextra.pdf
- Modelo CAS (CTAN): https://ctan.org/pkg/els-cas-templates ; instruções Elsevier: https://www.elsevier.com/latex
