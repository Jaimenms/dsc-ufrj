# 018 · Massa virtual (Drew & Lahey, forma objetiva) como regularização física (item 3)

| | |
|---|---|
| tipo | feature |
| área | modelo |
| estado | feito (2026-10-13); C_vm = 0.5, λ = 1 na configuração recomendada |
| onde | `model.options.virtual_mass = {"C_vm": 0.5, "lambda": 1.0}` (pesada pelo indicador de bolhas); `TwoFluidModel.eigenvalues(..., virtual_mass, vm_lambda)` |
| dados | `..._tol_{vm05,vm05_lv,vm05_lam2,bestion}.npz`; `..._fine_graded_{noslip,vm}.npz` |

## Antes

O modelo a dois fluidos da tese não tem massa virtual; a única forma de o tornar
hiperbólico era um termo de regularização (pressão interfacial, 017) sem significado
físico e com custo.

## Depois

Termo de Drew & Lahey nas equações de momento, IG·aG = F_G − K(aG − aL) − K(vG − vL)[(λ−2)∂vG/∂x
+ (1−λ)∂vL/∂x], K = C_vm·w_bolhas·αG·ρL·A, com as acelerações materiais resolvidas face a
face como um sistema 2×2 no esquema escalonado; a análise de características refaz as
linhas de momento com o mesmo acoplamento. A forma simples (λ = 2, só a diferença de
acelerações) **piora** a má-posição (97.6 % do mapa); a forma objetiva (λ = 1 ou 0) com
C_vm = 0.5 torna o modelo hiperbólico em todo o mapa.

## Benefício

| severe slug (padrão fixo, `noslip`, JAX, 3000 s) | passos | CPU | intervalo | pico | P_in (bar) |
|---|---|---|---|---|---|
| referência | 25.2k | 47 s | 749 s | 7.1 kg/s | 4.14–5.92 |
| massa virtual 0.5 (λ = 1) | 23.6k | 50 s | 826 s | 7.6 kg/s | 4.02–6.02 |
| massa virtual 0.5 + bolhas < 1 % | 21.9k | 46 s | 801 s | 6.3 kg/s | 4.03–5.85 |
| bolhas < 1 % + Bestion δ = 1.2 | 41.4k | 86 s | 748 s | 5.9 kg/s | 4.16–5.85 |

Ao contrário da pressão interfacial, a massa virtual **muda o ciclo** (intervalo +7–10 %,
pressão mínima 0.1 bar mais baixa) ao mesmo custo, e o efeito é igual nas duas formas:
vem do acoplamento das inércias no riser (o gás acelera com o líquido, o blowdown é mais
lento), não da regularização. É uma sensibilidade de primeira ordem no período. O efeito
persiste com o refinamento (malha fina graduada: 778 s com, 710 s sem).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Massa virtual como regularização física (item 3)".
