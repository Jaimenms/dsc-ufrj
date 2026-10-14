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

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Massa virtual como regularização física (item 3)») em 2026-10-14.*

Termo de massa virtual de Drew & Lahey nas equações de momento
(`PipeParams.virtual_mass` = C_vm, pesado pelo indicador de bolhas, e
`vm_lambda` = parâmetro da forma objetiva; numpy e JAX, `tests/test_jax_rhs.py`):

    IG·aG = F_G − K (aG − aL) − K (vG − vL)[(λ−2) ∂vG/∂x + (1−λ) ∂vL/∂x]
    IL·aL = F_L + K (aG − aL) + (reação),   K = C_vm·w_bolhas·αG·ρL·A

com a = Dv/Dt as acelerações materiais (no esquema escalonado, (dq/dt +
d(qv)/dx)/I, resolvido face a face como um sistema 2×2 e devolvido na forma
dq/dt) e as derivadas espaciais das velocidades por diferenças centrais nas
faces. A análise de características (`TwoFluidModel.eigenvalues(...,
virtual_mass, vm_lambda)`) refaz as linhas de momento da matriz quase-linear com
o mesmo acoplamento.

| mapa (αG 1e−3–0.6 × escorregamento 0–5 m/s) | fração mal posta |
|---|---|
| bolhas, riser 90° | 50.5 % |
| bolhas + massa virtual 0.5, forma simples (λ = 2) | 97.6 % |
| bolhas + massa virtual 0.5, forma objetiva (λ = 1 ou 0) | 0 % |
| estratificado 8.6° + bolhas abaixo de 1 % | 37.6 % |
| idem + massa virtual 0.5 (λ = 1) | 0 % |
| idem + pressão interfacial de Bestion δ = 1.2 | 0 % |

A forma simples (só a diferença de acelerações) piora a má-posição; a forma
objetiva de Drew & Lahey com C_vm = 0.5 torna o modelo hiperbólico em todo o
mapa, tal como a pressão interfacial de Bestion, mas com um termo físico (a
inércia da fase dispersa) em vez de um termo de regularização.

| severe slug (padrão fixo, `noslip`, JAX, 3000 s) | passos | CPU | blowdowns (s) | intervalo | pico | P_in (bar) | produção |
|---|---|---|---|---|---|---|---|
| referência | 25.2k | 47 s | 38, 704, 1484, 2284 | 749 s | 7.1 kg/s | 4.14–5.92 | 0.515 |
| massa virtual 0.5 (λ = 1) | 23.6k | 50 s | 44, 822, 1678, 2522 | 826 s | 7.6 kg/s | 4.02–6.02 | 0.504 |
| massa virtual 0.5 (λ = 1) + bolhas < 1 % | 21.9k | 46 s | 46, 796, 1624, 2450 | 801 s | 6.3 kg/s | 4.03–5.85 | 0.507 |
| massa virtual 0.5 (λ = 2) | 20.0k | 47 s | 44, 832, 1684, 2514 | 823 s | 7.8 kg/s | 4.02–5.93 | 0.505 |
| bolhas < 1 % + Bestion δ = 1.2 | 41.4k | 86 s | 36, 718, 1532, 2280 | 748 s | 5.9 kg/s | 4.16–5.85 | 0.515 |

Ao contrário da pressão interfacial (que não muda o ciclo mas custa 1.6× mais
passos), a massa virtual **muda o ciclo**: intervalo entre blowdowns 7–10 %
maior e pressão mínima de entrada 0.1 bar mais baixa, ao mesmo custo. O efeito
é o mesmo nas duas formas (λ = 1 e 2), portanto vem do acoplamento das inércias
no riser em bolhas (o gás acelera com o líquido, o blowdown é mais lento), não
da regularização. É um efeito físico de primeira ordem no período e entra na
lista de sensibilidades do artigo. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{vm05,vm05_lv,vm05_lam2,bestion}.npz`.

**Consequência prática: a malha fina graduada no pescoço voltou a ser viável.**
A configuração 30/10 com razões 0.25/3 (células de 0.8 m na base do riser), que
com o alvo de fração real e ν = 0.01 custava 302.6k passos e 83 min (*chattering*
nas células cheias de líquido), com `noslip` e a configuração rápida custa 77.9k
passos / 240 s, e com massa virtual 0.5 + bolhas abaixo de 1 % 68.9k passos /
189 s. Ciclos: 5 blowdowns em 3000 s, intervalo 710 s, picos 10.3 kg/s (igual ao
antigo 10.2) sem massa virtual; 4 blowdowns, intervalo 778 s, picos 9.7 kg/s com
massa virtual, ou seja, o alongamento do intervalo pela massa virtual (+7–10 %)
mantém-se com o refinamento. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_fine_graded_{noslip,vm}.npz`.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
