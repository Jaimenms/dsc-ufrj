# 017 · Hiperbolicidade: bolhas abaixo de 1 % de gás e pressão interfacial de Bestion

| | |
|---|---|
| tipo | feature |
| área | fechamentos / modelo |
| estado | feito (2026-10-13); `low_void` na configuração recomendada, Bestion disponível |
| onde | `model.regime.low_void = {"regime": "bubble", "alpha": 0.01}`; `model.options.interfacial_pressure` (δ), `ip_slip_max`; `TwoFluidModel.eigenvalues` |
| dados | secção "Hiperbolicidade" do `nb_severe_slug`; `..._arc_av_reg.npz` |

## Antes

Os autovalores da matriz quase-linear do modelo a dois fluidos mostram que o fechamento
**estratificado é mal posto para αG ≲ 0.003 mesmo sem escorregamento** (limite de camada de
gás muito fina, |Im λ| ≈ 2–4 m/s), e é exatamente aí que toda célula cheia de líquido é
avaliada (αG cortado em 1e−3); o de bolhas é mal posto em baixa fração com escorregamento
(caso clássico). A tese não trata a hiperbolicidade; tensão superficial não resolve (termo
dispersivo de 3.ª ordem, corte capilar milimétrico).

## Depois

Duas regras: padrão de bolhas abaixo de 1 % de gás (rampa C1) e, opcionalmente, o termo de
pressão interfacial de Bestion/CATHARE com δ ≥ 1 (derivadas em αG e P incluídas). O mapa
(αG × escorregamento) fica inteiramente hiperbólico com as duas. A regra de fração baixa
vale também no modo de identificação automática (021).

## Benefício

| mapa (αG 1e−3–0.6 × escorregamento 0–5 m/s) | fração mal posta |
|---|---|
| estratificado 8.6° + bolhas abaixo de 1 % | 37.6 % |
| idem + Bestion δ = 1.2 | 0 % |
| idem + massa virtual 0.5 (018) | 0 % |

Na malha fina graduada a regularização reduziu o *chattering* à metade (amplitude e
frequência) sem o eliminar — a causa residual era acústica (007) e o arrasto espúrio (006).
Na malha grossa a regularização não muda o custo nem o ciclo (85.4 mil passos contra
85.7); com Bestion o custo sobe 1.6× (41.4k passos) sem mudar o ciclo, por isso a
recomendação passou a ser a massa virtual (018), que é física.

## Detalhe

*Texto movido de `docs/severe_slug.md` (secção «Intermitência severa — Hiperbolicidade») em 2026-10-14.*

**Hiperbolicidade dos fechamentos** (seção própria no notebook). Autovalores da
matriz quase-linear (`TwoFluidModel.eigenvalues`): o fechamento estratificado é
mal posto para αG ≲ 0.003 **mesmo sem escorregamento** (limite de camada de gás
muito fina, |Im λ| ≈ 2–4 m/s), e é exatamente aí que toda célula cheia de
líquido é avaliada (αG cortado em 1e−3); o de bolhas é mal posto em baixa fração
com escorregamento (caso clássico). Tensão superficial não resolve: é dispersiva
(3ª ordem), não altera as características, e com σ = 0.05 N/m o corte capilar é
milimétrico. O que torna o mapa inteiro hiperbólico: padrão de bolhas abaixo de
αG = 1 % (`low_void_regime=2`, `low_void_alpha`) **mais** o termo de pressão
interfacial de Bestion/CATHARE com δ ≥ 1 (`interfacial_pressure`, derivadas em αG
e P incluídas). Nos testes de 80 s na malha fina graduada essa regularização
reduziu o *chattering* à metade (amplitude e frequência) mas não o eliminou;
relaxação mais longa, bandas mais largas e chaveamentos de gás mais largos
pioram.

## Referências

[docs/severe_slug.md](../severe_slug.md) (visão geral). O texto integral está na secção «Detalhe» acima.
