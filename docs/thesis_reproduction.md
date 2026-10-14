# Reprodução do capítulo 10 da tese

*Documentação do mfsim (índice no [README](../README.md)).*

## Modelo e layout de estados

Variáveis nodais `[alphaG, PT = P/PR, qL, qG]`; condições de contorno Dirichlet
(valor do vértice) ou von Neumann (gradiente nulo) removem estados nas
extremidades; vértices dinâmicos acrescentam `[HG, PT]`. O layout coincide com o
do código original para os esquemas nodais, o que permite comparar diretamente com
`estados.X` gravado nos `.mf2`.

## Notebooks de reprodução fiel (`notebooks_src/thesis/`)

Um notebook por secção do capítulo 10 (`nb_10_1_1` ... `nb_10_3_3`), **sem simulação dentro
do notebook**: cada um lê as corridas do lote `batches/thesis.json` (`uv run mfsim batch
batches/thesis.json` grava `runs/chapter10/*.npz`), que integram os casos JSON exportados dos
`.mf2` com a configuração original (BDF do SciPy, tolerâncias e passo máximo do ficheiro), e
sobrepõe os resultados do MATLAB (`thesis.mf2.legacy_results`). As sensibilidades de malha das
figuras 10.4–10.6 e 10.12–10.13 são entradas do lote com o estado estacionário recalculado em
cada malha. As correções necessárias para carregar os ficheiros (estado inicial de
`estados.X[:, 0]`, padrão do 10.3.1, composição do 10.3.2a) estão nos JSON e descritas na
[issue 001](issues/001-reproducao-capitulo-10.md); tudo o que é melhoria fica nos notebooks de
`notebooks_src/issues/` (um por grupo de issues, com as corridas de `batches/issues.json` e de
`notebooks/data`) e no [índice de issues](issues/INDEX.md). A suíte
noturna reexecuta o lote e os notebooks e compara os textos.

## Principais resultados da reprodução

* Fechamentos validados contra os coeficientes F1..F12 gravados pelo MATLAB
  (erro relativo 1e-14).
* Casos 10.1.1, 10.1.2, 10.1.3, 10.2.1 (A/B), 10.2.2 e 10.3.3 reproduzidos com
  erro de 1e-4 a 1e-2 da faixa das variáveis, em segundos.
* O estado estacionário discreto dos esquemas centrados (Galerkin, diferenças
  centrais) não é único quando a fração de gás de entrada é livre (autovalor
  nulo); com gradiente nulo na entrada, ou com o esquema `fv_upwind`, é único e
  independente da malha.

## Gas lift com intermitência forte (`notebooks/nb_gas_lift_intermitencia.ipynb`)

Rede do caso 10.3.2 com a choke de gás a 12 % e 10 %, simulada por 16 000 s:
ciclos de *heading* com período de ~4000 s, produção de líquido oscilando entre
0.2 e 23 kg/s, injeção de gás intermitente pela válvula de retenção e
pressurização lenta do anular entre os ciclos (cerca de 1 min de CPU por caso).

## Observações sobre os arquivos originais

Em [issues/001](issues/001-reproducao-capitulo-10.md): condição inicial gravada = estado
final, "upwind" original = diferenças regressivas em todas as variáveis, estado estacionário
discreto não único, ficheiros 10.1.4/10.3.1/10.3.2a.
