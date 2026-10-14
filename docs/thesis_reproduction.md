# Reprodução do capítulo 10 da tese

*Documentação do mfsim (índice no [README](../README.md)).*

## Modelo e layout de estados

Variáveis nodais `[alphaG, PT = P/PR, qL, qG]`; condições de contorno Dirichlet
(valor do vértice) ou von Neumann (gradiente nulo) removem estados nas
extremidades; vértices dinâmicos acrescentam `[HG, PT]`. O layout coincide com o
do código original para os esquemas nodais, o que permite comparar diretamente com
`estados.X` gravado nos `.mf2`.

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

* `dados.tubo.condicaoInicial*` contém o estado **final** da simulação; o estado
  inicial está em `estados.X[:,0]`.
* `CASO10-1-4`: o resultado gravado corresponde a uma versão anterior do atrito
  de Beggs & Brill; `CASO10-3-1`: padrão intermitente (como no texto).
