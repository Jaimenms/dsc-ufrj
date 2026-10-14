# 035 · Tolerâncias absolutas por variável e orçamento de Newton: a configuração rápida

| | |
|---|---|
| tipo | feature |
| área | integrador |
| estado | feito (2026-10-13); substituída na prática por `tol_range` (034) com o mesmo efeito |
| onde | `integration.atol_P`, `atol_q`, `max_newton`; `scheme.options.art_visc` |
| dados | `..._arc_av05.npz`, `..._arc_fast1.npz`, `..._arc_fast2.npz`; sondas `restart_ring.py`/`restart_probe.py` (histórico do git até ao commit 22ecbc9) |

## Antes

Malha grossa de referência: 14 ms por passo, 23.6 avaliações do lado direito por passo
(metade em Jacobianas coloridas), 92 % dos passos nas fases com gás no riser (toque acústico
do tampão e do bolsão a ±0.03–0.05 kg/s), Newton com ≈ 6 iterações e 19 % de falhas por
orçamento (12 iterações) na vazão de gás do bolsão; 85.7 mil passos e 20 min para 3000 s.

## Depois

Medido em janelas de reinício na fase cara (800–830 s) e confirmado em corridas completas:

| configuração (malha grossa, arco, conservativa, 3000 s) | passos | CPU |
|---|---|---|
| referência: ν = 0.01·c·h | 85.7 mil | 20 min |
| ν = 0.05·c·h + orçamento de Newton 20 iterações | 59.2 mil | 14 min |
| + tolerância absoluta nas vazões 0.03 kg/s e em P 0.1 bar | 23.2 mil | 5 min |
| + tolerância absoluta 0.05 kg/s e 0.2 bar | 21.8 mil | 4.8 min |

## Benefício

4× mais rápido com o mesmo ciclo (P_in 4.14–5.92 contra 4.19–5.86 bar; picos 6.3–6.8 contra
6.7 kg/s) e balanço exato. A tolerância absoluta nas vazões é a alavanca decisiva: o
controlador deixa de resolver o toque de ±0.03 kg/s (6 % da alimentação) e o BDF2 o amortece
sozinho. Sem ganho ou piores: BDF1, relaxação da fase ausente mais longa ou mais curta,
γL = 30, tolerância relativa só em P, Jacobiana a cada passo, passos diferentes nas
diferenças finitas. GPU não ajudaria (100–200 estados; custo dominado por latência de
chamadas, não por vazão); paralelismo só entre simulações — o caminho foi compilar o lado
direito (036).

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Intermitência severa — Aceleração») em 2026-10-14.*

**Aceleração sem perder o ciclo** (seção "Aceleração" do notebook). Onde vai o
tempo na malha grossa de referência: 14 ms por passo, 23.6 avaliações do lado
direito por passo (metade em Jacobianas coloridas, 25 cores — o padrão real só
permitiria 22), 92 % dos passos nas fases com gás no riser (h ≈ 14 ms, toque
acústico do tampão e do bolsão), Newton com ≈ 6 iterações por não-suavidade dos
fechamentos (Jacobiana nova a cada passo não muda nada). GPU não ajuda (100–200
estados, custo dominado por latência de chamadas numpy, não por vazão);
paralelismo só entre simulações. Alavancas que funcionaram, medidas em janelas
de reinício na fase cara (`restart_ring.py`, `restart_probe.py`; histórico do git até ao commit 22ecbc9)
e confirmadas em simulações completas de 3000 s (mesmo ciclo, balanço exato):

| configuração (malha grossa, arco, conservativa) | passos | CPU |
|---|---|---|
| referência: ν = 0.01·c·h | 85.7 mil | 20 min |
| ν = 0.05·c·h + orçamento de Newton 20 iterações (`max_newton`) | 59.2 mil | 14 min |
| + tolerância absoluta nas vazões 0.03 kg/s e em P 0.1 bar (`atol_q`, `atol_P`) | 23.2 mil | 5 min |
| + tolerância absoluta 0.05 kg/s e 0.2 bar | 21.8 mil | 4.8 min |

A tolerância absoluta nas vazões é a alavanca decisiva: o controlador deixa de
resolver o toque de ±0.03 kg/s (6 % da alimentação) e o BDF2 o amortece
sozinho; a pressão de entrada fica em 4.14–5.92 bar (referência 4.19–5.86) e os
picos em 6.3–6.8 kg/s (referência 6.7). Sem ganho ou piores: BDF1, relaxação da
fase ausente mais longa ou mais curta, γL = 30, tolerância relativa só em P,
Jacobiana a cada passo, passos diferentes nas diferenças finitas.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
