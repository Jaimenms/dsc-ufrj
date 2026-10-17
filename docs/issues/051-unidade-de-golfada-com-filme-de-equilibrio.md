# 051 · Unidade de golfada com o filme de equilíbrio e a perda de aceleração na frente do pistão

| | |
|---|---|
| tipo | bugfix |
| área | fechamentos (intermitente) |
| estado | feito (2026-10-17); opção `closures.intermittent.unit = "film"`; o caso offshore do artigo 01 usa-a |
| onde | `mfsim/closures/intermittent.py` (`film_unit`, `falling_film_fraction`, perda de aceleração em `intermittent`), `mfsim/closures/common.py` (`ClosureInputs.slug_film`, `aG_film_eq`), `mfsim/models/twofluid.py` (passa o estratificado de equilíbrio das vazões da célula ao fechamento); `tests/test_slug_film.py` |
| dados | estados do caso offshore (`generators/offshore.py`) com o fluido da Tabela 6 a 100–120 bar e o choke aberto, e com o do §5.5 a 200 bar (fora do git) |

## Antes

A unidade de golfada da tese toma três coisas de correlações:

- o gás do pistão, de Barnea e Brauner;
- o comprimento do pistão (Norris para D > 38 mm);
- o da unidade, v_t/ω_s.

Daqui sai a fração de pistão, e o gás do filme sai do balanço de massa com a fração de gás da célula α_G. Mas α_G é um
estado do modelo a dois fluidos, por isso a unidade fica sobredeterminada. Com uma célula rica em gás, o filme tem de
guardar quase todo o gás: o corte da tese, ou a unidade consistente (050), deixa-o com α_f = 0,95.

A cinemática da unidade, (v_t − u_f)(1 − α_f) = (v_t − u_L)(1 − α_s), dá então ao filme velocidades de −20 a −30 m/s.
O atrito desse filme, negativo, vence o do pistão: o atrito da mistura com a parede sai negativo (−(Γ_L + Γ_G) até
+50 N/m) e o quase-estacionário tem a pressão a subir ao longo do escoamento. Linhas do caso offshore em escoamento
intermitente:

- §5.5 a 100–250 bar: pressão de 100 → 250 bar na linha;
- Tabela 6 a 100 bar com o choke aberto: entrada da linha a 5,4 bar, abaixo dos 10 bar do separador.

## Depois

`film_unit` toma dois dados da física e deixa o terceiro para o balanço de massa, como nos modelos mecanicistas de
Taitel e Barnea:

- **pistão:** gás de Barnea e Brauner, limitado ao da célula;
- **filme:**
  - nos troços horizontais e descendentes, o estratificado de equilíbrio de Taitel e Dukler às vazões da célula, com
    as tabelas da identificação de padrão;
  - nos ascendentes, a bolha de Taylor com o filme em queda livre de Brotz,
    u_f = −9,916 [g D sin θ (1 − α_f^½)]^½. O α_f sai da cinemática da unidade por bissecção, seguida de um passo de
    Newton que dá a derivada exata no JAX;
  - transição suave entre 0 e 10°, e nunca menos gás do que a célula;
- **fração de pistão** r = (α_f − α_G)/(α_f − α_s), entre 0 e 1.

Nos dois limites a unidade reduz-se às velocidades da célula. Rica em gás (α_G ≥ α_f) é só filme; rica em líquido
(α_G ≤ α_s) é só pistão.

**Perda de aceleração do filme na frente do pistão** (Taitel e Barnea 1990): uma vez por unidade,
Δp = ρ_L R_f (v_t − u_f)(u_L − u_f) = ρ_L R_s (α_f − α_s)(v_t − u_L)²/R_f ≥ 0. Soma-se ao líquido como
A Δp ω_s/v_t, com a frequência de Zabaras (049) ou a da tese. Anula-se suavemente quando a unidade degenera (r a menos
de 0,05 de 0 ou de 1), porque sem filme ou sem pistão não há frente a apanhar filme.

Neste fechamento a frequência só entra por esta perda. O comprimento do pistão de Norris deixa de entrar.

As opções `"thesis"` e `"consistent"` ficam como estavam (goldens de nível A iguais).

## Benefício

- **Estado da Tabela 6 a 120 bar com o choke aberto.** É uma célula da linha horizontal com α_G = 0,78,
  v_SL = 1,9 m/s e v_SG = 12 m/s.
  - A unidade da tese dá atrito da mistura negativo.
  - A unidade com o filme de equilíbrio dá filme a +8 a +10 m/s e atrito positivo, de +29 a +47 N/m ao longo da linha
    (teste).
- **Testes de consistência:**
  - massa e vazões da célula exatas para qualquer α_G e ângulo;
  - raiz do filme em queda e a sua derivada contra diferenças finitas;
  - limites degenerados;
  - paridade numpy × JAX.

## Referências

- Taitel, Y., Barnea, D., 1990. Two-phase slug flow. Advances in Heat Transfer 20, 83–132.
- Taitel, Y., Barnea, D., Dukler, A.E., 1980. Modelling flow pattern transitions for steady upward gas–liquid flow in
  vertical tubes. AIChE Journal 26(3), 345–354.
- Brotz, W., 1954. Über die Vorausberechnung der Absorptionsgeschwindigkeit von Gasen in strömenden
  Flüssigkeitsschichten. Chemie Ingenieur Technik 26, 470–478.
- Taitel, Y., Dukler, A.E., 1976. A model for predicting flow regime transitions in horizontal and near horizontal
  gas–liquid flow. AIChE Journal 22(1), 47–55.
- Barnea, D., Brauner, N., 1985. Holdup of the liquid slug in two phase intermittent flow. International Journal of
  Multiphase Flow 11(1), 43–49.
- Issues 049 (frequência de Zabaras) e 050 (unidade consistente com a massa).
