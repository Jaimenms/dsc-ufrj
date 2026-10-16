# 041 · Tempo simulado adaptativo (paragem quando o sinal assenta)

| | |
|---|---|
| tipo | feature |
| área | integração |
| estado | feito (2026-10-16) |
| onde | `time.settle = {window, t_min, amp_tol, rel_tol, signal}` no JSON; `mfsim.simulation._integrate_settled`, `window_signature`, `settled` |
| dados | testes em `tests/test_settle.py`; uso no mapa de intermitência severa (043) |

## Antes

Cada corrida integrava até `tf`. Numa varredura isso obriga a escolher um `tf` único: curto
demais para os casos lentos (15 células do mapa ainda derivavam aos 8000 s, com o choke muito
fechado a linha leva milhares de segundos a pressurizar) e longo demais para os casos que
assentam depressa (a maioria das células estáveis está parada muito antes dos 8000 s, e as
corridas de baixa vazão chegaram a 50 mil passos).

## Depois

Com `time.settle` a integração faz-se em janelas de `window` segundos, cada uma a partir do
estado final da anterior. No fim de cada janela calcula-se a assinatura de um sinal (por
omissão a pressão no nó 1 do primeiro duto, em bar; `signal = {edge, node, var}` com `var` em
`P`, `alphaG`, `qL`, `qG`): amplitude, média, cruzamentos da média e período entre o primeiro e
o último cruzamento ascendente. Depois de `t_min` a corrida pára quando

- a amplitude da janela é menor do que `amp_tol` (`"steady"`), ou
- as duas últimas janelas têm pelo menos 4 cruzamentos e amplitude e período iguais a
  `rel_tol` (`"cycle"`).

Uma deriva sem ciclo não assenta e continua até `tf`, que passa a ser o tempo máximo. As
saídas mantêm-se nos instantes de `dt_out`, o acumulador dos fluxos de contorno é contínuo
entre janelas e `info["settle"]` guarda o motivo, o instante e as assinaturas de todas as
janelas. Valores por omissão: `window` 2000 s, `t_min` = `window`, `amp_tol` 0,05,
`rel_tol` 0,1. Instantes de saída explícitos (`t_eval`) desligam a paragem adaptativa.

## Benefício

- Cada caso de uma varredura corre o tempo de que precisa: os estáveis param ao fim de duas
  janelas, os cíclicos quando o ciclo se repete, os lentos continuam até assentar ou até `tf`.
- A assinatura de cada janela fica gravada, o que dá ao mapa a amplitude e o período sem
  pós-processamento.
- Medido nos testes: o caso 10.1.1 em regime permanente pára aos 40 s em vez de 200 s; sem
  assentar, o resultado em janelas coincide com a corrida contínua a 10⁻³; o acumulador do
  caso de referência em janelas fica a menos de 5 % do contínuo aos 60 s (as janelas reiniciam
  o BDF).

## Referências

Notebook: [nb_estabilidade_mapa](../../notebooks/issues/nb_estabilidade_mapa.ipynb) (fonte em `notebooks_src/issues/`).

Issues 005 (acumulador BDF), 040 (partida do estacionário) e 043 (mapa).
