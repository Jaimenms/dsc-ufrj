# 057 · Padrão automático e pistão de golfada no duto parado

| | |
|---|---|
| tipo | bugfix |
| área | identificação de padrão e fechamento intermitente |
| estado | feito (2026-10-17); sem opção: só muda os ramos que davam NaN (vazões nulas), o resto fica igual bit a bit |
| onde | `mfsim/closures/regime.py` (`identify_regime`: `dmax` do critério das bolhas dispersas), `mfsim/closures/intermittent.py` (`slug_liquid_fraction`) |
| dados | `cases/paper01/lab_fabre2_auto.json` (Fabre et al. 1990, caso 2, com o padrão automático do artigo 01), que parte do duto em repouso |

## Antes

O caso de laboratório com o padrão automático não dava um único passo: «Factor is exactly singular» no 1.º passo,
com JAX e com numpy. No estado inicial (vLS = vGS = 0) o lado direito tinha 162 componentes NaN, todas as equações
de momento, por duas razões:

- no critério das bolhas dispersas de Barnea, a mistura parada dá `dmax` = ∞ (escolhido assim para não formar
  0 · ∞ na diferenciação automática); com a sigmoide relativa do modo suave, o argumento é
  (ddef − ∞)/∞ = NaN, e os pesos de bolhas e intermitente saíam NaN em todas as células (a sigmoide absoluta da tese
  dá 0 nos dois casos);
- na fração de líquido do pistão de Barnea e Brauner, ReS = 0 era substituído por 1e-30: o atrito de Churchill
  dava fS = ∞ e fS·|uS|³ = ∞ · 0 = NaN.

## Depois

`dmax` vale 1e30 (e não ∞) no duto parado, e o número de Reynolds do pistão tem piso 1e-12. Os pesos e o pistão
ficam finitos em repouso; com escoamento, os valores são os mesmos de antes. O caso de laboratório com o padrão
automático arranca.

## Benefício

- `lab_fabre2_auto` passa de 0 passos a integrar. Sem a 054 pára no passo mínimo aos 91 s (41 332 passos);
  com a 054 a comparação com o ensaio (≈ 137 s, 126–235 kPa) está em curso.
- A rotina e os goldens de nível A ficam iguais (402 testes).

## Referências

Issues 021 (identificação automática suave e local), 030 (validação com laboratório), 037 (tangentes NaN) e 054;
Barnea (1986), *Transition from annular flow and from dispersed bubble flow*, Chem. Eng. Sci. 41; Barnea e Brauner
(1985), *Holdup of the liquid slug in two phase intermittent flow*, IJMF 11.
