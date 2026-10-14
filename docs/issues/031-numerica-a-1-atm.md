# 031 · Numérica à escala de laboratório (1 atm): singularidade do mapa de massas e Newton estagnado

| | |
|---|---|
| tipo | bugfix |
| área | integrador / esquema numérico |
| estado | feito (2026-10-13) |
| onde | `state_bounds` (banda de *undershoot* −min(0.005, 0.5·γG/γL)); `integration.slow_exit = true`; diagnóstico do passo rastejante no `ImplicitBDF` (verbose) |
| dados | `severe_slug_lab_{teng1,jansen}_fisico.npz` |

## Antes

Três problemas só apareciam a 1 atm, e travavam as corridas de Jansen e Tengesdal:

1. a banda de *undershoot* de αG do modo conservativo (−0.005) **cruzava a singularidade do
   mapa de massas** em αG = −γG/γL quando γL = 1000 (singularidade em −0.001; o campo usa
   γL = 100, singularidade em −0.0105): a pressão da célula colapsava para o piso de 100 Pa
   com o passo a rastejar;
2. na fase quase estática de pressurização (vazões de gás de 1e−5 kg/s a trocar de sinal
   nas faces junto ao bolsão; num tubo de 1" a 1 atm há 1e−7 kg de gás por célula) a
   Jacobiana reutilizada ficava desatualizada e o Newton estagnava perto da tolerância
   (17–20 iterações, 35–40 % de rejeições, passo de 1e−5 s sem nada a mudar);
3. com γL = 1000 e arrasto físico apareceu um tabuleiro de xadrez em αG no topo do riser
   (células alternadas anular/bolhas a 0.9 de fração) antes do colapso de pressão.

## Depois

1. A banda de *undershoot* passa a −min(0.005, 0.5·γG/γL) (sem efeito com γL = 100), e os
   casos de laboratório usam γL = 100 como o de campo (011);
2. recálculo da Jacobiana dentro do Newton quando a convergência é lenta (`slow_exit`, que
   já existia para os eventos de válvula das redes, 033);
3. o tabuleiro de xadrez não voltou a aparecer com γL = 100; o integrador ganhou um
   diagnóstico do passo rastejante (componentes dominantes do erro local e da última
   atualização de Newton quando h < 1e−3 s) e `mfsim run` grava a trajetória parcial em falha.

## Benefício

Tengesdal passa de rastejo em 408 s para 800 s simulados em 50.4k passos (período 63.7 s);
Jansen completa 600 s em 98k passos. Alargar a banda de chaveamento do gás (`q_switch_G`)
também passa, com o dobro dos passos e período 7 % diferente, o que fixa a incerteza
numérica do período de laboratório em ±7–10 %.

## Detalhe

*Texto movido de `docs/lab_validation.md` (secção «Validação com dados de laboratório (item 8) — Achado 3») em 2026-10-14.*

**Achado 3 — numérica a 1 atm.** Três problemas só aparecem a escala de
laboratório: (i) a banda de undershoot de αG do modo conservativo (−0.005)
cruzava a singularidade do mapa de massas em αG = −γG/γL quando γL = 1000
(singularidade em −0.001; o caso de campo usa γL = 100, singularidade em −0.0105)
e a pressão da célula colapsava para o piso de 100 Pa com o passo a rastejar
(`state_bounds` agora usa −min(0.005, 0.5 γG/γL); os casos de laboratório
passaram a γL = 100 como o de campo); (ii) na fase quase estática de
pressurização da linha (vazões de gás de 10⁻⁵ kg/s a trocar de sinal nas faces
junto ao bolsão, num tubo de 1" a 1 atm o gás tem 10⁻⁷ kg por célula) a
Jacobiana reutilizada fica desatualizada e o Newton estagna perto da tolerância
(17–20 iterações, 35–40 % de rejeições, passo de 10⁻⁵ s sem nada a mudar no
estado): o recálculo da Jacobiana dentro do Newton quando a convergência é lenta
(`ImplicitBDF.slow_exit`, opção `slow` do `severe_D.py`, já usada nos casos de
rede com eventos) resolve-o — Tengesdal passa de rastejo em 408 s para 800 s em
50k passos; alargar a banda de chaveamento do gás (`fsw_G` 0.3) também passa,
com o dobro dos passos e período 7 % diferente; (iii) numa corrida com γL = 1000
(Tengesdal, arrasto físico) apareceu um tabuleiro de xadrez em αG no topo do
riser (células alternadas anular/bolhas a 0.9 de fração) antes do colapso de
pressão — não voltou a aparecer com γL = 100. O integrador ganhou um diagnóstico
do passo rastejante (componentes dominantes do erro local e da última atualização
de Newton quando h < 10⁻³ s) e o `severe_D.py` grava a trajetória parcial em falha.

## Referências

[docs/lab_validation.md](../lab_validation.md) (visão geral). O texto integral está na secção «Detalhe» acima.
