# 001 · Reprodução do capítulo 10: o que os ficheiros e o código originais escondiam

| | |
|---|---|
| tipo | estudo |
| área | reprodução da tese |
| estado | feito (2026-10-12/13) |
| onde | `mfsim/io_legacy.py`, `mfsim/cases/chapter10.py`, `cases/chapter10/*.json` |
| dados | `legacy/mfsim23/Files/*.mf2`; notebooks `nb_10_*` |

## Antes

Os 12 casos do capítulo 10 existiam como ficheiros `.mf2` (MAT v5) com `dados`, `saidas`
(incluindo os coeficientes F1..F12 por nó e por instante) e `estados` (T, X), gerados pelo
MFSIM em MATLAB com o `ode15s`. Não havia forma independente de os reexecutar nem de saber
que partes do resultado dependiam de detalhes de implementação.

## Depois

Os fechamentos, os esquemas nodais (Galerkin, Petrov-Galerkin, diferenças centrais e
regressivas), os elementos de rede e o layout de estados foram reimplementados e comparados
diretamente com o que está gravado. Cinco factos que não estão no texto da tese:

1. **A condição inicial gravada é o estado final.** `dados.tubo.condicaoInicial*` é
   sobrescrito pelo simulador no fim da corrida; o estado inicial da trajetória está em
   `estados.X[:, 0]`. O leitor dos `.mf2` usa a coluna 0 e os casos JSON exportados levam
   essa condição em `*.ic.npz`.
2. **O "upwind" do código original é diferenças regressivas em todas as variáveis, pressão
   incluída.** Isso explica o insucesso relatado na tese com esse esquema: regressivas na
   pressão não têm nada de *upwind* e produzem um esquema instável para o modo acústico.
3. **O estado estacionário discreto dos esquemas centrados não é único** quando a fração
   de gás na entrada é livre (a Jacobiana tem um autovalor nulo). Com gradiente nulo na
   entrada, ou com o esquema de volumes finitos (002), ele é único, independente da malha e
   igual ao da tese (αG = 0.4636 no 10.1.1).
4. `CASO10-1-4` foi gravado com uma versão anterior do atrito de Beggs & Brill, e
   `CASO10-3-1` com padrão intermitente (como no texto). O 10.3.2a não existe como ficheiro
   completo: é a rede do 10.3.2b com as válvulas do ficheiro "a" (composto no leitor, com a
   proveniência registada no JSON).
5. O caso 10.3.2b (gas lift intermitente) só roda em tempo razoável com `rtol = 1e-3`:
   com `1e-4` o esquema centrado resolve oscilações nó a nó no tubing com passos de
   milissegundos.

## Benefício

- Fechamentos validados contra F1..F12 do MATLAB com erro relativo 1e−14; os casos 10.1.1,
  10.1.2, 10.1.3, 10.2.1 (A/B), 10.2.2 e 10.3.3 reproduzidos com erro de 1e−4 a 1e−2 da
  faixa das variáveis, em segundos (`tests/test_chapter10_regression.py`, goldens de nível A).
- Cada caso é agora um ficheiro JSON autocontido (`cases/chapter10/`), reexecutável com
  qualquer integrador e com o backend JAX.

## Referências

[docs/thesis_reproduction.md](../thesis_reproduction.md); notebooks `notebooks_src/nb_10_*.py`.
