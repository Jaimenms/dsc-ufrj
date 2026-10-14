# 020 · Identificação de padrão da tese: quatro defeitos que a tornavam inutilizável em dinâmica

| | |
|---|---|
| tipo | bugfix |
| área | fechamentos / identificação de padrão |
| estado | feito (2026-10-13); correções sempre ativas (sem efeito quando os indicadores são 0/1) |
| onde | `mfsim/closures/regime.py` (`identify_regime`, `equilibrium_fractions`) |
| dados | diagnóstico denso em `archive/scratch_2026-10/tools/diag_ipe.py`; `tests/test_jax_regime_auto.py` |

## Antes

`identificaRegimeEstavel.m` (equilíbrios de Taitel–Dukler e Pots, critérios de
Kelvin–Helmholtz, Jeffreys, Froude e turbulência com chaveamento sigmoidal de inclinação
`k_switch` = 1e6) fazia o integrador rastejar no severe slug: 9834 passos até t = 1.5 s,
contra 101 até 2 s com padrão fixo. Causas, encontradas com saída densa de 1 ms:

1. células cheias de líquido: FG* = FG*c = 0 → sigmoide(0) = 0.5 em todo instante, e a
   fórmula Int = (1−Est)(1−Anu)(1−Bub) **não é uma partição da unidade** com indicadores
   fracionários (os pesos somavam 1.44);
2. com vGS < 0 (o gás da entrada inverte o sentido no toque acústico da partida) os modelos
   de equilíbrio dão NaN e o original cai em hD = 1e−6 → FG*c salta de 0.02 para 19.8;
   sign(vM) e aG_bol saltam também → pesos de 0 a 1 em 1 ms, independentemente de k;
3. a tabela "mais próxima" muda hD_est em degraus de 2–5 %, logo FG*c também;
4. `flag_vertical` comparava |θ| em **radianos com 90**: nunca era vertical, e o critério de
   Froude do anular nunca se aplicava a risers (riser cheio de líquido = "anular" via o
   equilíbrio de Pots degenerado).

## Depois

`flag_vertical` em graus; partição da unidade Int = (1−Est)(1−Anu)(1−p_bub); a regra de
fração baixa (017) vale também no modo automático (sem gás não há padrão a identificar).
O modo da tese (k = 1e6, chaveamento absoluto) continua disponível e é o usado nos testes
de equivalência com o MATLAB.

## Benefício

Sozinhas, as correções não bastam (o blowdown continuava caro); com as opções de 021 a
identificação automática reproduz o ciclo. Porte para JAX com as tabelas como constantes,
busca na tabela sem gradiente e raízes seguras (`tests/test_jax_regime_auto.py`, paridade 1e−14).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Identificação automática de padrão (regime 6)".
