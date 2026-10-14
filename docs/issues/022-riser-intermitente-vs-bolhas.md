# 022 · Fechamento do riser no blowdown: intermitente contra bolhas (item 5)

| | |
|---|---|
| tipo | estudo |
| área | fechamentos |
| estado | feito (2026-10-13) |
| onde | `model.regime.vertical.regime = "intermittent"` (em vez de `"bubble"`) |
| dados | `..._tol_int.npz`, `..._tol_int_hv_vm_lv.npz`, `..._boe_{bub,bub_dp01,bub_rules,int_rules}.npz` |

## Pergunta

O ciclo de severe slug obtido com bolhas prescritas no riser é uma previsão do modelo ou
uma consequência do fechamento escolhido?

## O que se encontrou

Forças sobre o líquido no riser nos estados típicos do blowdown (αG 0.3–0.85, vG 1.5–6 m/s):
bolhas dão GammaL de −19 a −2400 N/m; o intermitente (unidade de slug da tese) de −0.03 a
−128 N/m; o anular quase nada. As bolhas carregam o líquido pelo gás com muito mais
eficiência (picos de 7–8 kg/s contra 4–6 com identificação automática).

- **Caso base** (300 m a −3°, 4 bar, vSL 0.30): com intermitente no riser **não há ciclo**.
  Depois do primeiro blowdown o sistema vai para golfadas contínuas (αG no riser 0.01–0.3,
  P_in ≈ 5.8 bar quase constante, produção 0.3–1.2 kg/s; 101–112k passos). O caso está
  dentro da região de Bøe (vSL_mín = 0.06, com a forma correta do critério, 028), logo é uma
  limitação do fechamento intermitente a baixa fração de gás no riser, não uma previsão do
  critério: a 1e−2–0.3 de fração as bolhas retêm o gás na base do riser e permitem o bloqueio;
  o intermitente deixa-o subir como bolhas de Taylor.
- **Varredura em vSL** (1.2 e 1.6 m/s): os dois fechamentos dão escoamento permanente —
  a linha deixa de ser estratificada e Taitel (1986) exclui o bloqueio.
- **Caso de linha longa** (L = 1000 m, 2 bar, 6000 s): os dois fechamentos dão o ciclo
  clássico, período 3850–4280 s (±10 %), picos 7.8–8.9 kg/s; o intermitente sem as regras
  físicas falha na partida.

## Consequência

Os resultados do caso base devem ser apresentados como dependentes do fechamento a baixa
fração de gás no riser; o caso de linha longa é o que demonstra o mecanismo de forma
robusta; a verificação natural é a comparação com dados de laboratório (030), que
confirmou o ciclo com o arrasto físico de bolhas (015).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Fechamento do riser no blowdown: intermitente contra bolhas (item 5)») em 2026-10-14.*

Forças sobre o líquido no riser vertical nos estados típicos do blowdown
(αG 0.3–0.85, vG 1.5–6 m/s): o fechamento de **bolhas** dá GammaL de −19 a
−2400 N/m (arrasto do gás sobre o líquido, crescendo com αG e com o quadrado do
escorregamento), o **intermitente** (unidade de slug da tese) de −0.03 a −128 N/m
e o anular quase nada; só o intermitente tem correções de pressão grandes
(dPL até 1.5 kPa a αG = 0.3). As bolhas carregam o líquido pelo gás com muito
mais eficiência, o que explica os picos de 7–8 kg/s com bolhas contra 4–6 kg/s
nas corridas com identificação automática (intermitente → anular no riser).

Corridas com `regime_vertical=3` (intermitente acima de 45°, mistura de 20°),
`noslip`, JAX, 3000 s, com e sem as regras físicas (anular > 90 %, massa virtual,
bolhas < 1 %): **não há ciclo de severe slug**. Depois do primeiro blowdown
(42 s) e do reenchimento, o sistema vai para escoamento em golfadas contínuo pelo
riser: αG no riser 0.01–0.3, pressão de entrada ≈ 5.8 bar quase constante,
produção a oscilar entre 0.3 e 1.2 kg/s, riser com mais de 90 % de gás em 5 % do
tempo (19 % com bolhas); custo 101–112k passos (o escoamento em golfadas contínuo
é caro de resolver). O critério de Bøe na forma das fontes (Taitel 1986, Jansen
et al. 1996; sem fator de inclinação, ver a correção abaixo) dá vSL_mín = 0.06 m/s
para o caso, contra 0.30 m/s de alimentação: o caso base está **dentro** da
região de Bøe, e a ausência de ciclo com o intermitente é uma sensibilidade do
modelo ao fechamento do riser a baixa fração de gás, não uma previsão do
critério. O ciclo de severe slug obtido com bolhas prescritas no riser é
sustentado pelo fechamento: a 1e-2–0.3 de fração de gás as bolhas retêm o gás na
base do riser e permitem o bloqueio; o intermitente deixa-o subir como bolhas de
Taylor e o bloqueio não se forma. Consequência para os resultados anteriores: o
ciclo com bolhas é uma solução do modelo com aquele fechamento, não uma previsão
independente do padrão; a verificação natural é a comparação com dados de
laboratório (item 8).

**Varredura em vSL** (1.2 e 1.6 m/s, mesma geometria,
bolhas e intermitente no riser): os dois fechamentos dão **escoamento
permanente** (pressão de entrada constante, 5.9–6.1 bar; 356–3325 passos em
3000 s). Coerente com Taitel (1986): o critério de Bøe é condição necessária
para o bloqueio só quando a linha está estratificada, e a 1.2 m/s num tubo de
5 cm a linha já escoa em golfadas, com o gás arrastado pelo líquido. Com a forma
correta de Bøe todos estes pontos estão dentro da região (vSL_mín = 0.06 m/s em
300 m a 4 bar, 0.02 m/s em 1000 m a 2 bar); a passagem a escoamento permanente a
1.2 m/s vem da linha deixar de ser estratificada, não de Bøe. O caso de linha
longa abaixo (L_down = 1000 m, P_out = 2 bar) foi montado quando ainda se usava a
forma com sin θ (que dava vSL_mín = 1.03 no caso base e 0.24 na linha longa);
continua a ser o caso em que o ciclo é robusto ao fechamento do riser.

**Caso de linha longa** (antes "Bøe-consistente"; L_down = 1000 m, P_out = 2 bar, 50 células na linha,
6000 s, `noslip`, JAX; "regras" = anular > 90 %, massa virtual 0.5, bolhas < 1 %):

| riser | passos | CPU | blowdowns (s) | intervalo | pico | P_in (bar) |
|---|---|---|---|---|---|---|
| bolhas, sobrepressão inicial 0.3 bar | 105k | 590 s | 136, 4052 | 3916 s | 7.8 kg/s | 2.12–3.81 |
| bolhas, sobrepressão inicial 0.1 bar | 64k | 409 s | 134, 4062 | 3928 s | 8.2 kg/s | 2.12–3.81 |
| bolhas + regras | 57k | 341 s | 122, 3970 | 3848 s | 8.8 kg/s | 2.17–3.81 |
| intermitente + regras | 59k | 458 s | 110, 4386 | 4276 s | 8.9 kg/s | 1.99–3.81 |
| intermitente sem regras | — | — | não passa da partida (blowdown inicial violento com 0.3 bar; rasteja no riser cheio de líquido com 0.1 bar) | | | |

**Conclusão do item 5.** Onde o critério de Bøe prevê intermitência severa com
linha estratificada, o modelo a produz com qualquer fechamento do riser: ciclo
clássico (pressurização lenta da linha de 2.9 a 3.8 bar com o riser cheio de
líquido, blowdown, reenchimento), período 3850–4280 s (±10 % entre fechamentos),
picos 7.8–8.9 kg/s. No caso base (vSL = 0.30, L = 300 m, 4 bar; dentro da região
de Bøe, vSL_mín = 0.06) só o fechamento de bolhas sustenta um ciclo; com
intermitente o modelo dá golfadas contínuas, o que, com Bøe satisfeito, aponta
uma limitação do fechamento intermitente a baixa fração de gás no riser (a
comparação com dados de laboratório do item 8 é o teste directo). Para o artigo:
os resultados do caso base devem ser apresentados como dependentes do fechamento
a baixa fração de gás no riser; o caso de linha longa é o que demonstra o
mecanismo de forma robusta. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_boe_{bub,bub_dp01,bub_rules,int_rules}.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
