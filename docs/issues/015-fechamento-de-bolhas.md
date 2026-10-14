# 015 · Fechamento de bolhas da tese: velocidade terminal e força interfacial

| | |
|---|---|
| tipo | bugfix |
| área | fechamentos |
| estado | feito (2026-10-13); `drag = "physical"` recomendado (a tese continua disponível como `"thesis"`) |
| onde | `model.closures.bubble.drag = "physical"`; `mfsim/closures/bubble.py`; `tests/test_bubble_drag.py` |
| dados | `severe_slug_lab_{jansen,fabre2}_tese.npz` vs `..._fisico.npz`; `..._tol_hv_vm_lv_bd_{tese,phys}.npz` |

## Antes

O `modelo_bolhas.m` da tese (porte fiel) tem a velocidade terminal
`vb = sqrt(0.51 db + 2.14/db)`, que é a forma de Mendelson / Fan–Tsuchiya
`sqrt(2.14 σ/(ρL db) + 0.505 g db)` **sem g e sem σ/ρL** (vb ≈ 14 m/s, C_D ≈ 7e−4 em ar–água com
db = 11 mm), e a tensão interfacial `f_I ρG Δv|Δv|/2` com `f_I = 1.5 αG ρL C_D/db`, o que dá
uma força proporcional a ρG·αG² (dimensionalmente inconsistente; o arrasto de bolhas escala
com ρL e αG). No caso de campo (5 bar, 5 cm) os dois erros compensam-se (escorregamento de
equilíbrio 0.5–1 m/s a 10–30 % de gás); a 1 atm e 2.5 cm dão 4 m/s, e por isso nenhum caso
de laboratório ciclava: o gás atravessava o riser cheio de líquido a 4 m/s com 2–4 % de
fração, com força interfacial −0.1 N/m igual ao empuxo.

## Depois

Opção `physical`: vb de Mendelson, C_D = 4/3 g db Δρ/(ρL vb²), força por volume
3/4 C_D αG ρL |vr| vr/db — na velocidade terminal o arrasto iguala o empuxo (0.26 m/s contra
4 m/s da tese no teste). A bandeira por nó é propagada ao corpo do slug do intermitente.

## Benefício

| caso | tese | físico |
|---|---|---|
| Fabre caso 2 (medido 137 s) | sem ciclo clássico | **135.8 s** |
| Tengesdal teste 1 (medido 79 s) | — | 64–68 s |
| Jansen (medido 46.6 s, com choke) | sem blowout, 75 s irregular | 56–63 s |
| campo (referência) | intervalo 800 s, pico 6.07 kg/s | 799 s, 6.53 kg/s (+8 %), 7 % menos passos |

A correção não muda o severe slug de campo (os resultados anteriores mantêm-se) e passa a
ser a recomendada.

## Referências

[docs/lab_validation.md](../lab_validation.md) achado 1; 030.
