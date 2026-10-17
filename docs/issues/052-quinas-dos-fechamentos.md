# 052 · Quinas dos fechamentos: versões C1 como opção, sem ganho medido

| | |
|---|---|
| tipo | estudo |
| área | fechamentos, identificação de padrão, esquema |
| estado | feito (2026-10-17); opções `closures.kinks = "smooth"` e `scheme.options.relax_ramp = "smooth"`, desligadas por omissão (bit a bit com a tese) |
| onde | `mfsim/closures/common.py` (`smooth_max`, `smooth_min`, `smooth_abs`), `bubble.py` (diâmetro de Hinze/Calderbank, corte do escorregamento de Stuhmiller), `stratified.py` (Andritsos & Hanratty), `annular.py` (Kowalski), `regime.py` (módulos das velocidades nos equilíbrios), `discretization/staggered.py` (rampa da relaxação da fase que desaparece), opções e schema; `tests/test_smooth_kinks.py` |
| dados | estados da corrida base do artigo 01 (`runs/paper01/offshore_dynamics/ab1.000.npz`, 200 bar, PI 5, RGO 50, choke aberto) aos 2500 s (bloqueio), 4100 s (sopro) e 4600 s (fim do sopro); fora do git |

## Pergunta

Depois de a issue 051 ter removido as descontinuidades verdadeiras da unidade de golfada (saltos de «só filme» a «só
pistão», sinal discreto das velocidades, filme a tender para a fase extinta), o BDF ainda rejeita ~30 % das tentativas
nas fases rápidas do ciclo offshore, todas por falha do Newton (nenhuma pelo controlo de erro). Restavam nos fechamentos
quinas C0 (derivada descontínua, Jacobiana de um só lado): `min(d_b, d_max)` e o corte a ±3 m/s do escorregamento de
Stuhmiller nas bolhas, o `max(0, ·)` de Andritsos & Hanratty em v_GS = 5,5 m/s, o `max` de Kowalski no anular, o
`|v|` dos equilíbrios da identificação e a rampa linear da relaxação da fase que desaparece. São elas que limitam o
passo?

## Método

Cada quina ganhou uma versão C1 (máximo, mínimo e módulo suaves, 0,5[a + b ± √((a − b)² + ε²)] e √(v² + ε²), com
larguras de 5 % do diâmetro de bolha, 0,02 em |v_GS|/v_GT − 1, 10 % do fator turbulento de Kowalski, 0,05 m/s nas
velocidades; `tanh` no corte do escorregamento; smoothstep na rampa da relaxação), atrás das opções acima. Os testes
conferem a continuidade da derivada, a coincidência com a tese longe das quinas, a omissão intacta e a paridade JAX.
A medida: 150 s de integração a partir de três estados da corrida base, com e sem as opções.

## Resultado

| estado | tese (passos / rejeições) | suave (passos / rejeições) |
|---|---|---|
| 2500 s, bloqueio | 423 / 136 | 425 / 145 |
| 4100 s, sopro | 3987 / 1416 | 4503 / 1555 |
| 4600 s, fim do sopro | 1072 / 337 | 1260 / 406 |

Nenhum ganho; ligeiro custo. As quinas não são o gargalo. A classificação das falhas no sopro (60 s, 3163 tentativas
de Newton): 694 «lentas» (contração < 1/2 por iteração), 313 divergentes, 40 «enormes», 47 com lado direito não
finito; os Newtons que convergem precisam de 2 a 10 iterações (mediana 5), com a Jacobiana reavaliada em 2 de cada 3
passos. Isto é a não-linearidade forte de cada passo — frentes a atravessar células de 20 m a 10–20 m/s — e não a
suavidade dos fechamentos: o preditor erra por muito e o Newton modificado, a partir dele, estagna ou diverge.

## Conclusão

As opções ficam disponíveis (desligadas), documentadas e testadas; o caminho para baixar o custo das fases rápidas
é do integrador — preditor melhor nas frentes, busca em linha no Newton, política de reavaliação da Jacobiana — e
não dos fechamentos. O localizador de saltos (forças por célula no último estado aceite contra o preditor da tentativa
falhada) continua a ser o diagnóstico certo para distinguir os dois casos.

## Referências

- Issue 051 (unidade de golfada com o filme de equilíbrio: as descontinuidades verdadeiras).
- Andritsos, N., Hanratty, T.J., 1987. Influence of interfacial waves in stratified gas-liquid flows. AIChE Journal
  33(3), 444–454. Kowalski, J.E., 1987. Wall and interfacial shear stress in stratified flow in a horizontal pipe.
  AIChE Journal 33(2), 274–281. Hinze, J.O., 1955. Fundamentals of the hydrodynamic mechanism of splitting in
  dispersion processes. AIChE Journal 1(3), 289–295.
