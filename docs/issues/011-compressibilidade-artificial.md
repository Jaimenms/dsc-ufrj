# 011 · Compressibilidade artificial do líquido (γL = 100)

| | |
|---|---|
| tipo | feature |
| área | modelo / integrador |
| estado | feito (2026-10-13); padrão dos casos de severe slug e de laboratório |
| onde | `fluid.gammaL` (tese: 1000; severe slug: 100) |
| dados | todas as corridas `severe_slug_*`; `..._tol_hv_vm_lv_bd_tese.npz` confirma γL = 100 na referência |

## Antes

Líquido quase incompressível (γL = 1000, c ≈ 800 m/s): a rigidez acústica do tampão de
líquido no riser impõe passos pequenos sem que nada mude na dinâmica lenta do ciclo.

## Depois

γL = 100 (c = 258 m/s) nos casos de intermitência severa. A dinâmica lenta (pressurização
da linha, bloqueio, blowdown) não depende da velocidade do som do líquido; o custo cai.
Nos casos da tese (capítulo 10) mantém-se γL = 1000.

## Benefício

Permitiu as corridas longas (3000–9000 s) com que foram feitos os estudos; o efeito na
dinâmica foi verificado por comparação (mesmo ciclo). Interage com o modo conservativo:
a singularidade do mapa de massas está em αG = −γG/γL (−0.0105 com γL = 100, −0.001 com
1000), o que determinou a banda de *undershoot* (004) e explicou os travamentos a 1 atm
(031). γL = 30 foi testado na fase cara e não dá ganho adicional.

## Referências

Detalhe em [003](003-fase-que-desaparece.md) e [031](031-numerica-a-1-atm.md).
