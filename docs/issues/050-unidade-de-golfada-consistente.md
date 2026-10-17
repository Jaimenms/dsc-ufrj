# 050 · Unidade de golfada consistente com a massa

| | |
|---|---|
| tipo | bugfix |
| área | fechamentos (intermitente) |
| estado | feito (2026-10-16); opção `closures.intermittent.unit` = `"thesis"` (omissão) ou `"consistent"`; o caso offshore do artigo 01 usa `"consistent"`. Limitação em aberto: filme fino com velocidade não física em células ricas em gás a alta pressão (ver Benefício) |
| onde | `mfsim/closures/intermittent.py` (`slug_unit`), `mfsim/closures/common.py` (`ClosureInputs.slug_unit`), `mfsim/models/base.py` e `mfsim/models/twofluid.py` (`PipeParams.slug_unit`), `mfsim/core/options.py` (`ModelSpec.slug_unit`), schema; `tests/test_slug_unit.py` |
| dados | estados do caso offshore (`generators/offshore.py`) a 200 bar e abertura 0,2, relaxados por um transitório de 3000 s; varredura e transitórios a 90–110 bar com o choke aberto (fora do git) |

## Antes

A unidade de golfada da tese (`modelo_intermitente.m`) parte da fração de gás da célula α_G, da do pistão α_s
(Barnea e Brauner), do comprimento do pistão l_s e do da unidade l_u = v_t/ω_s. Faz:

- l_s = min(l_s, l_u) e l_f = max(l_u − l_s, 10⁻³ l_s);
- α_f = (l_u α_G − l_s α_s)/l_f;
- corta α_f a [10⁻⁶, 0,95].

Quando a célula tem mais gás do que a unidade consegue guardar, o corte quebra o balanço de massa:
α_G l_u ≠ α_s l_s + α_f l_f. As velocidades locais (relações de massa de Taitel e Barnea a partir de v_t) deixam então
de reproduzir as vazões da célula. No caso offshore a 100–250 bar (linha toda intermitente, α_G ≈ 0,45–0,6,
α_s ≈ 0,02–0,06, l_s = 17,4 m), o α_f do balanço antes do corte chega a:

- 420–470 com a frequência da tese (l_u < l_s, unidade só pistão);
- 1,2–2,2 com a de Zabaras (049).

## Depois

`slug_unit(aG, alphas, ls, lu, consistent=True)` impõe α_G l_u = α_s l_s + α_f l_f:

- **Gás do pistão.** α_s ≤ α_G: o pistão não guarda mais gás do que a célula.
- **Comprimento do pistão.** l_s ≤ l_u (0,95 − α_G)/(0,95 − α_s): o pistão encurta até o filme, com α_f ≤ 0,95,
  guardar o resto.
- **Filme.** l_f = l_u − l_s e α_f = α_G + l_s (α_G − α_s)/l_f.

No limite rico em líquido (α_G ≤ α_s) a unidade tem o estado da célula. Acima de α_G = 0,95 é só filme com α_f = α_G.
Nos dois casos as velocidades locais reduzem-se às da célula.

A opção `"thesis"` executa exatamente as mesmas operações de antes (goldens de nível A iguais).

## Benefício

- Balanço de massa exato da unidade para qualquer α_G, e velocidades locais que devolvem as vazões da célula (testes).
- **Limitação em aberto.** Em células ricas em gás com pistão pouco arejado, o filme fica com α_f = 0,95. A relação
  de massa u_f = v_t − (v_t − u_L)(1 − α_s)/(1 − α_f) dá-lhe então uma velocidade de −20 m/s ou menos (limitada pelo
  corte de 5× as velocidades superficiais).
  - **Efeito.** O atrito com a parede da mistura sai negativo (−(Γ_L + Γ_G) ≈ +50 N/m) na linha do caso offshore a
    100–250 bar, e o estado quase estacionário tem a pressão a subir ao longo do escoamento. Acontece com as duas
    unidades e com as duas frequências: é um defeito do modelo da unidade da tese, em que a espessura do filme sai do
    balanço de massa e não do balanço de momento do filme.
  - **Onde não aparece.** A 20–40 bar (o mesmo caso com o reservatório depletado e o choke aberto) o atrito da
    mistura é positivo e o estacionário é físico.
  - **Pendente.** Usar este fechamento em linhas inteiras de escoamento intermitente a alta pressão exige o filme do
    balanço de momento.

## Referências

- Taitel, Y., Barnea, D., 1990. Two-phase slug flow. Advances in Heat Transfer 20, 83–132.
- Barnea, D., Brauner, N., 1985. Holdup of the liquid slug in two phase intermittent flow. International Journal of
  Multiphase Flow 11(1), 43–49.
- Tese, secção 5.12 (eqs. 5.80–5.85); `modelo_intermitente.m` (histórico até 22ecbc9).
- Issue 049 (frequência de golfada de Zabaras).
