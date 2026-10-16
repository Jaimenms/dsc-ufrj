# mfsim — guia para sessões do Claude

Leia por esta ordem: este ficheiro, [README.md](README.md), [docs/issues/INDEX.md](docs/issues/INDEX.md),
[docs/why_no_severe_slug_in_thesis.md](docs/why_no_severe_slug_in_thesis.md), [docs/data_formats.md](docs/data_formats.md).
O método que nos trouxe até aqui está em [docs/method_thesis_to_modern_code.md](docs/method_thesis_to_modern_code.md).

## O que é

Reimplementação em Python do MFSIM da tese de doutorado *Modelagem e Simulação de Escoamento
Multifásico em Dutos de Produção de Óleo e Gás Natural* (Jaime Neiva Miranda de Souza, UFRJ/EQ,
TPQB, 2010; orientadores José Luiz de Medeiros, André Luiz Hemerly Costa e Giovani Cavalcanti
Nunes; código original em MATLAB): simulador
dinâmico de escoamento bifásico em redes de dutos (modelo a dois fluidos a 4 equações, modelo de
mistura, monofásicos; Galerkin, diferenças finitas, volumes finitos escalonados; BDF implícito
próprio, numpy ou JAX). Reproduz o capítulo 10 da tese bit a bit contra os ficheiros MATLAB e
resolve o que a tese não conseguiu: a intermitência severa (severe slugging), validada com dados
de laboratório. As 43 alterações em relação à tese estão em `docs/issues/` (feature, bugfix, estudo).

## Mapa do repositório

| pasta | papel |
|---|---|
| `mfsim/` | o simulador, genérico: só lê casos JSON (`mfsim-case/1`). Não contém casos, geradores nem nada da tese. |
| `thesis/` | ponte com a tese, fora do simulador: `.mf2` originais em `thesis/data`, leitor, registo legado, exportador para JSON, apoio aos notebooks fiéis. |
| `generators/` | geradores de casos (parâmetros físicos → JSON): `severe_slug.py`, `column.py`. |
| `cases/` | casos JSON: `examples/hello_world`, `chapter10/` (11 da tese), `severe_slug/` (referência, variantes `$extends`, laboratório), `studies/`. |
| `batches/` | lotes de corridas (`mfsim batch`): `thesis.json` → `runs/chapter10`, `issues.json` → `runs/issues`. |
| `notebooks_src/` | fontes dos notebooks (`# %%`): `thesis/` reprodução fiel (um por secção), `issues/` melhorias (um por grupo de issues). Nenhum simula: leem `runs/` e `notebooks/data`. |
| `notebooks/data/` | 80 corridas históricas e os dados do mapa de intermitência severa (`severe_slug_map_well*.npz`, de `tools/slug_map.py export`; 043), fora do git, fixados em `MANIFEST.sha256`; não são reproduzíveis bit a bit (dinâmica caótica), são dados. |
| `tests/` | rotina (`uv run pytest`, ≈3 min, 335 testes) e noturna (`-m nightly`, ≈35 min: lotes, corridas longas, laboratório, notebooks contra `tests/golden/notebook_text`). `tests/golden/` só por `tools/golden_generate.py --rebaseline --note`. |
| `tools/` | `golden_generate`, `manifest`, `make_nb`, `slug_map` (mapa de intermitência severa, 043), migração do dialeto antigo (`legacy_runs`, `convert_runs`, `severe_run`, `save_run_data`, `export_severe_cases`). |
| `docs/` | documentação; `docs/issues/NNN-*.md` + `INDEX.md`; `docs/future/` (planos: criogenia, CO₂, composicional, trifásico, assinaturas de golfada e alocação a separadores, este com o degrau 1 feito em 043); `docs/thesis.pdf`. |

## Comandos

```bash
uv sync
uv run mfsim validate cases/examples/hello_world.json && uv run mfsim run cases/examples/hello_world.json
uv run mfsim run caso.json --backend jax --set integration.rtol=1e-4      # overrides por caminho
uv run mfsim batch batches/thesis.json                                     # corridas dos notebooks fiéis
uv run mfsim batch lote.json --jobs 12                                     # lote em paralelo (processos)
uv run python tools/slug_map.py build && uv run python tools/slug_map.py plot   # mapa de intermitência severa (043)
uv run mfsim nb notebooks_src/thesis/nb_10_1_1.py --execute                # notebook a partir da fonte
uv run pytest                      # rotina; uv run pytest -m nightly      # noturna
uv run black --check mfsim tests tools notebooks_src generators thesis     # antes de cada commit
uv run python tools/manifest.py --check                                    # dados fora do git
```

## Regras acordadas com o dono (não negociar sem perguntar)

1. **Solver próprio.** Nada de pacotes de EDP; melhorias só como opções com bandeira, mantendo
   a fidelidade à tese disponível e os goldens de nível A (bit a bit) verdes. "Nada mais de gambiarras."
2. **Nada é apagado sem aprovação explícita.** Mover ou arquivar; ferramentas e dados só saem com um sim.
3. **Identificadores e chaves JSON em inglês; comentários, docs e mensagens em português.**
4. **Um só caminho de execução:** tudo o que `simulate` precisa está no JSON (`integration`,
   `initial_state`); os notebooks não simulam, leem corridas dos lotes.
5. **Cada passo fecha com** rotina verde, black limpo, docs/README/memória atualizados, commit com
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`, e tag quando é um marco.
6. **Cada alteração atómica ao modelo ou ao solver** ganha um ficheiro em `docs/issues/` (tipo,
   área, estado, onde, antes, depois, benefício medido) e uma linha no `INDEX.md`;
   `tests/test_docs_issues.py` exige a sincronia.
7. **Notebooks** são regenerados das fontes; os textos são goldens (tempos normalizados). Para
   rebaselinar um texto, re-executar o notebook primeiro.
8. **Decisões tomadas:** não adotar DWSIM nem tabelas de fluido real; `archive/` e `legacy/`
   foram removidos em 2026-10-14 (histórico até ao commit 22ecbc9).

## Armadilhas conhecidas

- Dataclasses do grafo aceitam atribuição de atributos desconhecidos em silêncio: usar
  `pipe.ports_in = ports(...)`, `net.set_options(...)`, nunca `pipe.bc_in = ...`.
- JAX: sequências de passos mudam com 1 ulp na Jacobiana; corridas longas comparam-se por
  assinaturas de ciclo, não ponto a ponto. O período do caso de Jansen não é reprodutível (58–90 s).
- `tf` menor que `dt_out`: `simulate` inclui sempre `tf` nos instantes de saída.
- O choke usa por omissão a lei homogénea (039); os casos da tese marcam `"law": "thesis"`, e corridas antigas
  sem a chave leem-se com a da tese. Um estado guardado como condição inicial nos nós da malha perde exatidão no
  esquema escalonado: use `state_to_case_ic(X, exact=True)` (040).
- zsh: `$VAR` não faz word-split; usar `${=VAR}`.
