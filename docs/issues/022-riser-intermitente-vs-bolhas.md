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

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Fechamento do riser no blowdown (item 5)".
