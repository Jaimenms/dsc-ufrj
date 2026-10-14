# Dados da tese para comparação

Os doze ficheiros `CASO10-*.mf2` (MAT v5, 36 MB) gravados pelo MFSIM original em MATLAB, com a
entrada (`dados`), as saídas pós-processadas (`saidas`, incluindo os coeficientes F1..F12 por nó e
instante) e a trajetória (`estados`: T, X) de cada caso do capítulo 10. São lidos por
`thesis.mf2.load_mf2` (nomes resolvidos nesta pasta), exportados para `cases/chapter10/*.json` por
`python -m thesis.export --chapter10` (cada JSON guarda a origem em `meta.provenance.source`) e usados
pelos notebooks de `notebooks_src/thesis/` e pelos testes como referência do MATLAB. Versionados no
git como binários e fixados em `MANIFEST.sha256`; o código MATLAB original (`legacy/mfsim23`) ficou no histórico do git até ao commit 22ecbc9.
