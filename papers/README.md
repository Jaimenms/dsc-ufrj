# Artigos

*Documentação do mfsim (índice geral no [README](../README.md)).*

Um artigo por pasta `NN-<slug>/`, todos com a mesma construção e a partilhar bibliografia, estilo de
figuras e regras de `make`. O mecanismo veio do `research-factory`
(`01-multiphase-compositional-flow/papers/`) em 2026-10-16 e foi adaptado ao mfsim: as figuras leem
corridas pelo `mfsim.runs.load_run`, a paleta é a das entidades do escoamento bifásico (padrões,
classes do mapa, partidas, esquemas) e a revista alvo e o seu guia estão em
[00-escolha-da-revista.md](00-escolha-da-revista.md).

| pasta | conteúdo |
|---|---|
| `_common/` | `paper.mk` (regras de `make`), `figlib.py` (caminhos, estilo, paleta por entidade, gravação PDF+PNG, fragmentos de tabela), `mfsim-paper.mplstyle`, `references.bib` (bibliografia canónica do tema), `latex/` (ícones da classe CAS) |
| `_template/` | esqueleto de artigo novo (`tools/new_paper.py` copia-o) |
| `01-severe-slugging-two-fluid/` | **artigo 1**: intermitência severa com o modelo a dois fluidos e identificação dinâmica do padrão — boa colocação do modelo, discretizações, integrador, backends, perfis e mapa de estabilidade com choke |
| `02-gas-lift-surrogate-optimization/` | **artigo 2** (em planeamento): gas lift desde a pressão na plataforma sobre o modelo do artigo 1, substituto treinado com estacionários + autovalores e dinâmicas, alocação do gás entre vários poços com a mesma geometria ao longo da vida, com a estabilidade (golfada severa, *heading*) como restrição; água fora (limitação declarada). Ver o `PLAN.md` |
| `03-severe-slugging-bifurcations/` | **artigo 3** (em planeamento): ciclos limite do modelo do artigo 1 calculados diretamente (*shooting* com Newton–Krylov e a sensibilidade ao estado inicial propagada no BDF) e seguidos por continuação: natureza do Hopf, transição para a relaxação, duplicação de período, histerese e sensibilidades do ciclo. Ver o `PLAN.md` |

Dentro de cada artigo:

| ficheiro | papel |
|---|---|
| `main.tex` | o manuscrito (classe `cas-sc`; inglês; comentários em português) |
| `refs.bib` | referências só deste artigo; as comuns vêm de `../_common/references.bib` |
| `PLAN.md` | análise, casos de simulação, exemplo base e o que cada figura e tabela mostra (para quem retoma o artigo) |
| `figures/make_figures.py` | **um só guião** gera as figuras em `figures/generated/` (PDF vetorial + PNG) e os fragmentos de tabela em `tables/generated/` **só a partir de corridas gravadas** (`runs/`, `notebooks/data/`) e de caches calculados do modelo sem integrar no tempo; não simula. `figlib.run_cli(FIGURES, descrição, tables=TABLES)` trata a linha de comando (`--out`, `--tables`, `--only`) |
| `figures/static/` | figuras que não vêm de código (esquemas), no git |
| `build/` | saída do LaTeX, **no git** (o PDF de cada artigo fica versionado); recompilar com `make` antes de cada commit que mude o artigo |
| `figures/generated/`, `tables/generated/` | saídas reproduzíveis, fora do git |
| `Makefile` | três linhas: `PAPER = main` e `include ../_common/paper.mk` |

## Construir

```bash
cd papers/01-severe-slugging-two-fluid
make check        # que motor LaTeX vai usar e onde está o projeto
make              # figuras e tabelas (se o guião mudou) + PDF em build/main.pdf
make refigures    # força as figuras e as tabelas (depois de novas corridas em runs/)
make pdf-only     # só o LaTeX
make clean        # apaga build/ ; make distclean apaga também as figuras geradas
```

Motor: `latexmk` (TeX Live/MacTeX) se existir, senão `tectonic` (`brew install tectonic`; um binário que
descarrega os pacotes à primeira compilação e corre o BibTeX sozinho). As corridas de que cada artigo
precisa estão no `PLAN.md` dele e nos lotes `batches/paper<NN>_*.json`. Sem corridas, as figuras saem com
um aviso no lugar dos dados, as tabelas com «--», e o PDF compila na mesma.

## Artigo novo

```bash
uv run python tools/new_paper.py slug-mitigation --title "Gas lift against severe slugging"
cd papers/02-slug-mitigation && make
```

## Convenções

- As figuras do artigo são código: tudo o que aparece no PDF sai de `make_figures.py` + `figlib`, com a
  paleta por entidade de `figlib.COLORS` (a mesma entidade tem a mesma cor em todos os artigos; validada
  para daltonismo) e as larguras de coluna da Elsevier (`SINGLE` 90 mm, `MID` 140 mm, `DOUBLE` 190 mm).
- Os artigos não simulam: as corridas vêm de `mfsim batch` (casos JSON, caminho único de execução) e das
  ferramentas de `tools/`; o guião de figuras só lê.
- A bibliografia canónica do tema fica em `_common/references.bib` (chave `apelido+ano`, DOI sempre que
  exista); `refs.bib` só para o que é específico do artigo.
- O guia da revista, resumido em `00-escolha-da-revista.md`, já está embutido no esqueleto (resumo ≤ 250
  palavras, 3–5 *highlights* ≤ 85 caracteres, 1–7 palavras-chave, SI, declarações). Na submissão, além do
  `.tex`, vão: cada figura como ficheiro separado (`figures/generated/*.pdf`, renomeadas Figure_1…), os
  *highlights* num ficheiro à parte e as declarações.
- Os artigos não comparam com os resultados da tese: a tese é citada como origem do modelo; as escolhas
  que não funcionaram aparecem como lições gerais, medidas com o código atual.
