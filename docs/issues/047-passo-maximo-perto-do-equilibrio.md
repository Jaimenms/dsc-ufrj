# 047 · Passo máximo nas corridas que partem de um equilíbrio (o BDF amortece a instabilidade)

| | |
|---|---|
| tipo | bugfix |
| área | integração / análise |
| estado | feito (2026-10-16) nas ferramentas e lotes do artigo 01; o mapa 043 (padrão fixo) e o seu notebook ainda têm a coluna «partida do estacionário» sem passo máximo |
| onde | `tools/slug_map.py` (`MAX_STEP = 1 s` em `point_case`), `batches/paper01_dynamics.json` (`integration.max_step = 1`); `integration.max_step` já existia |
| dados | `runs/maps/well/summary.json` (043): passos das corridas da partida do estacionário; sondas do artigo 01 a 15 bar |

## Antes

As corridas do mapa (043) e as partidas do estacionário (040) usavam `integration.max_step = null`. Junto a um
equilíbrio a solução quase não muda, o controlo de erro deixa o passo crescer para dezenas ou centenas de
segundos, e um BDF com |λh| grande amortece também os modos instáveis (a região de estabilidade do BDF2
inclui boa parte do semiplano direito longe da origem). A trajetória fica então presa a um equilíbrio
instável e a corrida é classificada como estável:

- no mapa com padrão fixo, 7 das 8 células de «histerese» (estáveis a partir do estacionário, a ciclar a
  partir da linha bloqueada) têm partidas do estacionário com 22 a 97 passos em 2000 s (19 bar/0,36;
  24 bar/0,46, 0,6 e 0,8; 30 bar/0,8 e 1,0; 15 bar/0,22). Só a de 8 bar/0,06 (11 275 passos) foi integrada
  de facto;
- com o padrão dinâmico, a 15 bar, as aberturas 0,28 e 0,36 ficaram no estacionário perturbado durante 6000 s
  com 37 e 696 passos.

## Depois

As corridas que partem de um equilíbrio (ou que assentam num) usam um passo máximo de 1 s, abaixo das escalas
de tempo das instabilidades lentas que interessam (períodos de dezenas de segundos a horas): o BDF integra
então com precisão o crescimento da perturbação e a corrida só sai estável se o equilíbrio o for. O custo das
corridas estáveis sobe de dezenas para alguns milhares de passos, sem efeito nas corridas cíclicas (passos
de milissegundos nos blowdowns).

## Benefício

Verificação com a sonda do ponto estável (15 bar, abertura 0,13, padrão dinâmico): com passo máximo de 1, 0,3 e
0,1 s a perturbação de 0,1 % decai (amplitude da pressão de entrada de 0,007 bar nos primeiros 300 s para
0,0001 bar aos 1200 s), como prevê a linearização do estacionário com a matriz de massa (048: estável, par lento
−0,0033 ± 0,0067j /s): a classificação do ponto não depende do passo a partir de 1 s. A 0,28, onde o par lento é
instável (+0,00125 ± 0,0121j /s), a corrida com passo máximo de 1 s oscila com o período do par (~495 s).

Verificação nas 8 células de histerese do mapa 043 (feita pela sessão que refaz o 043): com passo máximo de 1 s
e de 0,1 s as classes, amplitudes e períodos são os mesmos (só o número de passos muda, ~5×); com o passo
limitado, 5 das 8 passam a ciclar também a partir do estacionário (19 bar/0,36, 365 s; 24 bar/0,6, 256 s;
24 bar/0,8, 291 s; 30 bar/0,8, 170 s; 30 bar/1,0, 159 s) e 3 mantêm histerese real (8 bar/0,06; 15 bar/0,22;
24 bar/0,46). 1 s chega para classificar.

A partida do estacionário volta a ser um teste de estabilidade linear do equilíbrio; a comparação com a
partida bloqueada separa de facto instabilidade de histerese. Números do mapa com padrão dinâmico no
artigo 01; o mapa 043 com padrão fixo precisa de refazer a coluna da partida do estacionário para as
contagens de histerese serem válidas.

## Referências

Issues 040 (partida do estacionário), 041 (paragem adaptativa), 043 (mapa) e o artigo 01. Hairer e Wanner
(1996), regiões de estabilidade das fórmulas BDF.
