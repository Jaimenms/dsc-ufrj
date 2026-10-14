# De uma tese e um código legado a uma versão moderna e melhorada: o método

*Documentação do mfsim (índice no [README](../README.md)). Consolidação do que funcionou nas
sessões de 2026-10-12 a 2026-10-14 entre o dono do projeto e o Claude, escrita para ser reutilizada
noutro projeto do mesmo tipo: um texto científico, um código antigo que o acompanha, e a vontade
de o reproduzir, modernizar e levar mais longe.*

## O que entra e o que sai

Entrada: o texto da tese (PDF), o código MATLAB original e os ficheiros de resultados gravados
(`.mf2`). Saída: um pacote Python com arquitetura limpa, os casos da tese reproduzidos bit a bit
contra os resultados originais, uma lista de alterações atómicas documentadas com antes, depois
e benefício medido, uma resposta à pergunta que a tese deixou em aberto, e uma rede de segurança
que garante que nada disto se perde quando o código muda.

## As seis frentes, na ordem em que compensa atacá-las

### 1. Rede de segurança antes de tocar em qualquer coisa

- `git init` no dia um; dados grandes fora do git com um manifesto de hashes
  (`tools/manifest.py`); tudo o que se arquiva fica com índice; nada se apaga sem aprovação.
- Referências de regressão em três níveis: **A** bit a bit (hash dos arrays) onde a ordem das
  operações não muda, **B** tolerância relativa documentada onde muda, **C** assinaturas de ciclo
  para dinâmica caótica. Geradas só por uma ferramenta com `--rebaseline --note`, com registo em
  `CHANGELOG.md`. Uma suíte de rotina de minutos e uma noturna de dezenas de minutos.
- Corridas longas são dados: a dinâmica é caótica e duas sequências de passos diferentes dão
  curvas diferentes ponto a ponto. A identidade do código prova-se pelo lado direito avaliado nos
  estados guardados e por corridas curtas, não pela trajetória inteira.

### 2. Reprodução fiel, componente a componente

- Portar fechamentos, esquemas e elementos um de cada vez, comparando com as grandezas
  intermédias que o código antigo gravou (aqui os coeficientes F1..F12 por nó: erro 1e−14).
- Descobrir o que os ficheiros escondem e registar em proveniência: condição inicial gravada
  que era o estado final, casos gravados com uma versão anterior de um fechamento, redes
  compostas de dois ficheiros, um "upwind" que não era upwind.
- Manter o legado disponível como opção (os esquemas da tese continuam a correr) e ler o
  legado só por uma ponte fora do simulador.

### 3. Arquitetura que separa o genérico do específico

- Um grafo de vértices e arestas com especificações no tempo; um só código para a física
  sobre um backend injetado (numpy e JAX), sem ramos em valores, para ter `jit` e Jacobiana por
  diferenciação automática de graça.
- Casos como ficheiros JSON autocontidos com schema, herança (`$extends`) para variantes e
  overrides por caminho (`--set`); lotes de corridas declarados em JSON; o simulador só lê JSON.
- Fora do simulador: a ponte com a tese, os geradores paramétricos de casos, as ferramentas de
  migração. Dentro: só o que qualquer caso precisa.

### 4. Investigar o problema crítico com método

- Formular a hipótese, construir a sonda mais barata que a testa (janelas de reinício de 20 s na
  fase cara, saída densa de milissegundos, mapas de hiperbolicidade, decomposição do balanço de
  momento), medir, e só então correr o caso completo.
- Uma alavanca só fica se o ciclo não muda e o balanço de massa fecha; registar também o que não
  funcionou (BDF1, MUSCL, Jacobiana a cada passo), porque evita repetir.
- Separar o que é numérico do que é físico com estudos de sensibilidade (joelho, malha,
  fechamento do riser, lei do gás, massa virtual), e fechar com dados de laboratório: foi a
  validação que mostrou o defeito do fechamento de bolhas da tese.
- Tudo como opção com bandeira e default igual à tese, até os resultados obrigarem a mudar a
  recomendação.

### 5. Documentar como alterações atómicas

- Um ficheiro por alteração, com tipo (feature, bugfix, estudo), área, estado, onde está no
  código e no JSON, dados de origem, **antes**, **depois**, **benefício medido** e limitações; um
  índice com uma linha por entrada; um teste que mantém índice e pasta em sincronia.
- Visões gerais curtas por tema apontam para as issues; o detalhe vive nas issues, não em
  documentos longos paralelos.
- Uma síntese de uma página responde à pergunta da tese a partir das issues.

### 6. Notebooks como relatórios, não como laboratórios

- Dois conjuntos: reprodução fiel (um por secção da tese, só a configuração original) e
  melhorias (um por grupo de issues, a apontar para elas). Nenhum simula: as corridas vêm de
  lotes reproduzíveis e os notebooks só leem e desenham.
- Os textos impressos são goldens com tempos normalizados; a noturna reexecuta lotes e
  notebooks.

## Como trabalhar com o Claude neste tipo de projeto

- Começar por um plano aprovado com fases e portas de saída; cada fase fecha com a suíte verde,
  documentação atualizada e um commit com tag.
- Dizer as regras no início e repeti-las quando forem violadas: solver próprio, melhorias só com
  bandeira, nada apagado sem aprovação, inglês no código e português nos textos, um só caminho
  de execução.
- Perguntar "para que serve isto?" sempre que um ficheiro não se explica: foi assim que saíram
  do simulador o registo da tese, os geradores, as ferramentas de migração e o arquivo.
- Pedir avaliação antes de implementação quando a pergunta é de desenho ("acha que seria
  complexo…"); dizer "não quero avançar" quando a resposta não convence.
- Exigir que cada resultado numérico apresentado venha de uma corrida gravada e nomeada, nunca de
  memória.

## Prompt consolidado (para reutilizar)

> Tenho o texto de uma tese (PDF), o código original em [linguagem] e os ficheiros de resultados
> gravados. Quero reimplementá-lo em Python com arquitetura limpa, reproduzir os casos do texto
> contra os resultados originais, e depois resolver [o problema que a tese deixou em aberto].
> Regras: solver próprio sem pacotes de EDP; toda a melhoria é uma opção com bandeira e o
> comportamento da tese continua disponível; nada é apagado sem a minha aprovação explícita;
> identificadores e chaves em inglês, comentários e documentação em português; cada passo fecha
> com testes verdes, formatação, documentação e commit. Primeiro monta a rede de segurança: git,
> manifesto dos dados, referências de regressão bit a bit e uma suíte de rotina curta e uma
> noturna. Depois reproduz componente a componente contra as grandezas intermédias gravadas e
> regista tudo o que os ficheiros originais escondem. A seguir organiza o código como um grafo
> genérico com uma só implementação da física sobre numpy e JAX, casos em JSON com schema e
> herança, lotes de corridas, e tudo o que é específico da tese fora do simulador. Para o
> problema em aberto, trabalha por hipóteses com sondas baratas, mantém só o que preserva o
> ciclo e fecha o balanço de massa, faz estudos de sensibilidade e valida com dados publicados.
> Documenta cada alteração como uma issue atómica com antes, depois e benefício medido, num
> índice, e escreve uma síntese de uma página que responda à pergunta da tese. Os notebooks
> dividem-se em reprodução fiel e melhorias e não simulam: leem corridas de lotes.
