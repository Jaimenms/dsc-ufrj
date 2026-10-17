# 049 · Frequência de golfada de Zabaras (2000) como opção do fechamento intermitente

| | |
|---|---|
| tipo | feature |
| área | fechamentos (intermitente) |
| estado | feito (2026-10-16); opção `closures.intermittent.slug_frequency` = `"thesis"` (omissão) ou `"zabaras"`; o exemplo base do artigo 01 (caso offshore) usa `"zabaras"` |
| onde | `mfsim/closures/intermittent.py` (`slug_frequency_zabaras`; docstring de `slug_frequency` corrigido), `mfsim/closures/common.py` (`ClosureInputs.slug_frequency`), `mfsim/models/base.py` e `mfsim/models/twofluid.py` (`PipeParams.slug_frequency`), `mfsim/core/options.py` (`ModelSpec.slug_frequency`, JSON e legado), schema; `tests/test_slug_frequency.py` |
| dados | estados das corridas `runs/paper01/dynamics` (0,36 e 1,0 a 15 bar) e continuação de Hopf a 15 bar no exemplo base anterior do artigo 01 (`cases/severe_slug/well_choke_auto.json`); fora do git |

## Antes

A unidade de golfada (`intermittent`) usava só a frequência da tese, eq. 5.83, que a tese atribui a Manolis (1995):

w_s = (u_GS/D) [0,0323/2 · (u_LS/u_GS) (ρL/Δρ)^½ f_L^−½ Eo^−0,2]^n,  Eo = D (gΔρ/σ)^½,  N_f = D^1,5 (ρL Δρ g)^½/μL.

Duas notas sobre esta fórmula:

- **Expoente.** O texto da tese escreve n = 0,85 + 260/N_f; o código MATLAB (`frequenciaSlugs` em
  `modelo_intermitente.m`) e o Python usam o inverso, n = N_f/(260 + 0,85 N_f). Com N_f grande (líquidos pouco
  viscosos), o do código tende a 1/0,85 ≈ 1,18, perto do expoente 1,2 de Gregory e Scott, por isso a gralha deve
  estar no texto. Mantém-se o do código (fidelidade aos ficheiros MATLAB, goldens de nível A).
- **Atribuição.** O docstring do porte dizia «Zabaras 2000 / Gregory-Scott generalizada», e o rascunho do artigo 01
  seguiu-o. Não é a correlação de Zabaras, e as revisões publicadas não registam esta forma para Manolis et al.
  (1995), que dão como uma modificação de Gregory e Scott para alta pressão. A origem não é verificável.

## Depois

`slug_frequency_zabaras` implementa Zabaras (2000), SPE J. 5(3) 252–258:

f_s = 0,0226 [u_LS/(g D) · (19,75/u_M + u_M)]^1,2 · [0,836 + 2,75 (sin θ)^¼],  u_M = u_LS + u_GS (SI).

Isto é a correlação de Gregory e Scott (1969) com o fator de inclinação de Zabaras.

- **Dutos descendentes.** O fator foi ajustado em dutos de 0 a 11° ascendentes; nos descendentes usa-se o fator do
  horizontal (sin θ truncado a 0).
- **Duto parado.** u_M leva 10⁻⁶ m/s.
- **Comportamento por omissão.** A opção `"thesis"` não passa a bandeira ao fechamento (`ClosureInputs.slug_frequency
  = None`): o caminho do código e os resultados ficam bit a bit iguais. O JSON só escreve
  `closures.intermittent` quando a opção não é a da omissão.

## Benefício medido

- A fórmula fica verificável numa referência publicada, com o fator de inclinação para o riser (o da tese não
  depende do ângulo). Os testes conferem:
  - o horizontal e o vertical contra Gregory e Scott (×0,836 e ×3,586);
  - o expoente MATLAB da fórmula da tese;
  - a passagem JSON e legado;
  - JAX contra numpy.
- Exemplo base anterior do artigo 01 (2 pol., riser de 15 m, 15 bar), sem transitórios:
  - **Frequência.** Nas células com peso intermitente acima de 0,5, Zabaras dá frequências 2,5× (linha) a 8–9× (curva
    e riser) as da tese. A unidade (0,3–0,7 m) fica mais curta do que o pistão de Nicholson (1,63 m), e o filme
    desaparece (ramo `l_s = l_u`).
  - **Abertura de Hopf.** Fica igual: 0,2413 → 0,2410, com período 573 → 578 s.
  - **Aberturas grandes.** Com a fórmula da tese o par lento colide no eixo real a partir de 0,70. Com Zabaras
    continua complexo até 1,0 (Re 0,018 s⁻¹, período ~470 s).
  - **Comparação com a largura das sigmoides.** No mesmo estudo, k = 20 → 40 → 80 dá a_H = 0,2413 → 0,2428 → 0,2436
    e período 573 → 569 → 566 s. A regularização converge e a escolha da frequência de golfada pesa mais do que a
    largura das sigmoides acima do limiar de Hopf.

## Referências

- Zabaras, G.J., 2000. Prediction of slug frequency for gas/liquid flows. SPE Journal 5(3), 252–258.
  doi:10.2118/65093-PA.
- Gregory, G.A., Scott, D.S., 1969. Correlation of liquid slug velocity and frequency in horizontal cocurrent
  gas-liquid slug flow. AIChE Journal 15(6), 933–935. doi:10.1002/aic.690150623.
- Manolis, I.G., 1995. High pressure gas-liquid slug flow. Tese de doutoramento, Imperial College, Londres
  (citada pela tese para a eq. 5.83; não consultada).
- Tese, secção 5.12.6, eq. 5.83 (`docs/thesis.pdf`, p. 62); `frequenciaSlugs` em `modelo_intermitente.m`
  (histórico até 22ecbc9).
- Tabela de correlações de frequência (Gregory e Scott, Zabaras, Heywood e Richardson, ...): COBEM 2007,
  «A comparative study of closure equations for gas-liquid slug flow», artigo COBEM2007-2415.
