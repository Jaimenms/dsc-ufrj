# 030 · Validação com dados de laboratório (item 8)

| | |
|---|---|
| tipo | feature |
| área | validação |
| estado | feito (2026-10-13); choke no topo (item 9) pendente para fechar Jansen |
| onde | `cases/severe_slug/lab_{jansen,fabre1,fabre2,tengesdal1}.json`; `mfsim val <corrida> --plot` (`mfsim.analysis.lab_validation`) |
| dados | `severe_slug_lab_<caso>_<tese|fisico>.npz`; fontes: Jansen, Shoham & Taitel 1996; Fabre et al. 1990 via Tengesdal 2002; Tengesdal 2002 |

## Antes

Nenhuma comparação do modelo com dados medidos de intermitência severa: a tese não
simulou o fenómeno e os resultados de campo (caso de referência) não têm medição. Sem
isso, não se sabia se o ciclo obtido era uma previsão ou um artefacto do fechamento (022).

## Depois

Quatro instalações de laboratório (ar–água 1" e 2", ar–óleo 3", linhas de 9–85 m, risers de
3–15 m, saída atmosférica) montadas como casos JSON com partida de linha fechada (027),
padrão fixo com as regras físicas (016–018), `tol_range` 0.01, `slow_exit` (031) e JAX, e a
comparação automática de período e pressões com os valores publicados.

## Benefício

| caso | bolhas | período (medido) | P_in kPa (medido) | o que o modelo faz |
|---|---|---|---|---|
| Fabre caso 2 | tese | — (137) | 200–237 (126–235) | sem ciclo clássico |
| Fabre caso 2 | físico | **135.8 s (137)** | 105–238 (126–235) | ciclo clássico regular, 8 blowouts em 1200 s, picos 32× |
| Jansen 1" (sem o choke do ensaio) | físico | 56–63 s (46.6) | 99–133 (111–133) | ciclo clássico com refluxo; o choke do ensaio retém ≈ 1 m de líquido e alonga o ciclo |
| Tengesdal teste 1 (3", óleo) | físico | 64–68 s (79) | 123–233 (121–222) | ciclo clássico em dente de serra, picos 12–13× |
| Fabre caso 1 | físico | 104 s (≈ 70) | 121–167 (128–210) | penetração contínua de gás, como o tipo observado |

Leitura: com o fechamento de bolhas da tese o modelo **não reproduz** o ciclo em nenhum
caso (015); com o arrasto físico os três casos de ciclo clássico saem como ciclo clássico,
sempre com a pressão máxima certa (riser cheio) e o riser a esvaziar um pouco mais do que o
medido no mínimo. A escala de laboratório custa 100–200 passos por segundo simulado,
10–50× o do campo.

## Limitações

O período do caso de Jansen (irregular) não é reprodutível pelo mesmo modelo (58–90 s sob
perturbações de 3 % na tolerância): a comparação com os 46.6 s medidos pede o choke
modelado. A sensibilidade do período à banda de chaveamento do gás é de ±7–10 %.

## Detalhe

*Texto movido de `docs/lab_validation.md` (secção «Validação com dados de laboratório (item 8)») em 2026-10-14.*

Fontes com geometria, vazões e ciclo medidos (PDFs obtidos em 2026-10-13):

| fonte | instalação | condições medidas usadas |
|---|---|---|
| Jansen, Shoham & Taitel (1996), IJMF 22:1055 (mesma bancada de Taitel et al. 1990) | ar–água, 2.54 cm, linha 9.1 m a −1°, riser 3 m, saída atmosférica, +10 m equivalentes de gás a montante (tanque), choke no topo C = 120 kPa·s²/m² | tabela 1: U_LS 0.0959 / U_GSO 0.0753 m/s → ciclo 46.6 s, golfada 3.53 m, cíclico com refluxo; pressão na base do riser 10–32 kPa man. (fig. 8) |
| Fabre et al. (1990), via Tengesdal (2002, figs. 3-3 e 3-5) | ar–água, 2", linha 25 m a −0.57°, riser 13.5 m, saída atmosférica | caso 2: vSL 0.13 / vSG0 0.20 m/s → período ≈ 137 s, pressão na linha 126–235 kPa abs (riser cheio no máximo); caso 1: vSL 0.13 / vSG0 0.45 → ≈ 70 s, 128–210 kPa (gás penetra continuamente, tipo 3) |
| Tengesdal (2002), tese TUSSP, instalação grande | ar–óleo (SG 0.856, 18.9 cSt a 40 °C), 3", linha 19.8 m (+ tanques até 85 m efetivos) a −1°, riser 14.6 m | teste 1: vSL 0.32 / vSG 0.35 m/s → 79 s, ΔP 101 kPa (121–222); teste 2: vSL 0.2 / vSG 0.68 → 66 s, 101 kPa |
| Sancho (IST, 2015), tabela 5 de Malekzadeh et al. (2012) | ar–água, 2", linha 100 m a −2°, riser 15.5 m | 38 casos com período e tipo, mas sem as vazões na versão obtida (não usada) |

Casos montados com `severe_slug_case` (`archive/scratch_2026-10/sh/val_launch3.sh`; hoje `mfsim run cases/severe_slug/lab_*.json --backend jax`; opções novas do
`severe_D.py`: `rhoG0`, `P0`, `sigma`, `muG`, `P_gas_ref` = pressão em que vSG é
dada, `dt_out`, `L_h2`, `roughness`, `bubble_drag`, `t_ramp`), partida de linha
fechada (`start="shut_in"`, rampa de 20 s; a partida bloqueada com 0.05 bar de
sobrepressão dá um blowdown inicial violento a escala de laboratório), padrão
fixo, `noslip`, anular > 90 %, massa virtual 0.5, bolhas < 1 %, `tol_range` 0.01,
JAX. O volume de gás a montante é representado por linha horizontal de entrada
(`L_h1`; o estratificado guarda ≈ 35 % de gás nela: 15 m ≈ 5 m de gás no caso
Jansen, contra 10 m da bancada). Resumo e figuras por `mfsim val <corrida> --plot`
(`mfsim.analysis.lab_validation`, `mfsim.plotting.lab_validation_figure`).

*Texto movido de `docs/lab_validation.md` (secção «Validação com dados de laboratório (item 8) — Resultados») em 2026-10-14.*

**Resultados** (`mfsim val`; período = média entre mínimos de P_in abaixo
de 30 % da faixa; pressão na entrada da linha, kPa abs; "medido" = fonte acima).
Partida de linha fechada e γL = 100 salvo indicação; "tese"/"físico" = fechamento de bolhas.

| caso | bolhas | período (medido) | P_in (medido) | o que o modelo faz | passos |
|---|---|---|---|---|---|
| Jansen (partida bloqueada, γL = 1000) | tese | 75 s (46.6) | 125–133 (111–133) | sem blowout: o riser fica cheio de líquido e o gás atravessa-o a 4 m/s com 2–4 % de fração (`severe_slug_lab_jansen_tese_bloqueada`) | 26.3k / 400 s |
| Fabre caso 2 | tese | — (137) | 200–237 após 320 s (126–235) | dois blowouts na partida (≈ 90 e 215 s, o segundo completo, 4.3 kg/s) e depois riser quase cheio com fugas de gás a cada ≈ 65 s (quedas de 20–40 kPa, picos 0.5–1.1 kg/s): sem ciclo clássico | 43.7k / 1200 s |
| Fabre caso 2 | físico | **135.8 s (137)** | 105–238 (126–235) | ciclo clássico regular: 8 blowouts em 1200 s, riser cheio no máximo de pressão e esvaziado no mínimo (o modelo esvazia mais do que o medido: 105 contra 126 kPa), picos 8–8.5 kg/s (32×), gás a 5–15 m/s na base durante o blowout (`severe_slug_lab_fabre2_fisico`) | 117.5k / 1200 s |
| Jansen (linha fechada, γL = 100) | tese | irregular (46.6) | 122–133 (111–133) | sem blowout: oscilações pequenas e irregulares, gás a 4 m/s com ≤ 40 % de fração na base (`severe_slug_lab_jansen_tese`) | 63.3k / 600 s |
| Jansen (1"; sem o choke do topo) | físico | 56–63 s (46.6, com choke) | 99–133 (111–133) | ciclo clássico com refluxo: 7–8 blowouts em 600 s, picos 0.8 kg/s (16×), riser esvaziado por completo no mínimo (o choke do ensaio, 10–15 kPa no blowout, retém ≈ 1 m de líquido e alonga o ciclo); período 62.6 s com `slow`, 56.5 s com `fsw_G` 0.3 (`severe_slug_lab_jansen_fisico`) | 98k / 600 s (`slow`) |
| Tengesdal teste 1 (3", óleo; linha efetiva de 85 m) | físico | 63.7 s (79) | 123–233 (121–222) | ciclo clássico em dente de serra: 10 blowouts em 800 s, picos 15–17 kg/s (12–13×), riser cheio no máximo e esvaziado no mínimo; ciclos menores intercalados depois de 400 s; com a banda de chaveamento do gás alargada (`fsw_G` 0.3) o período é 68.2 s (±7 % entre numéricas) (`severe_slug_lab_teng1_fisico`) | 50.4k / 800 s (`slow`) |
| Fabre caso 1 | físico | 104 s (≈ 70) | 121–167 (128–210) | oscilação regular com penetração contínua de gás (riser nunca enche, αG na base 0.3–0.9), como o tipo observado; amplitude 46 kPa contra 82 medidos; picos 2–5× a vazão | 6.7k / 1000 s |
| caso de campo D (referência, `tol_hv_vm_lv`) | tese → físico | 800 → 799 s | 4.00–5.85 bar iguais | ciclo igual, pico 6.07 → 6.53 kg/s (+8 %), 16.8k → 15.6k passos: a correção do arrasto de bolhas não muda o severe slug de campo (`..._tol_hv_vm_lv_bd_{tese,phys}`) | |

Leitura: (i) com o fechamento de bolhas da tese o modelo **não reproduz** o ciclo
de laboratório em nenhum caso — o gás atravessa a coluna de líquido a 3–4 m/s em
vez de a expulsar; (ii) com o arrasto físico de bolhas os três casos de ciclo
clássico saem como ciclo clássico: Fabre 2 com **período 135.8 s contra 137 s
medidos**, Tengesdal 64–68 s contra 79, Jansen 56–63 s contra 46.6 (ensaio com
choke, que alonga o ciclo), sempre com a pressão máxima certa (riser cheio) e o
riser a esvaziar um pouco mais do que o medido no mínimo; o caso de penetração
contínua (Fabre 1) sai qualitativamente certo (riser nunca enche) com período
50 % longo e amplitude 45 % baixa; (iii) no caso de campo a correção não muda o
ciclo, portanto os resultados anteriores mantêm-se, mas o fechamento físico passa
a ser o recomendado (`bubble_drag="fisico"`); (iv) a escala de laboratório custa
≈ 100–200 passos por segundo simulado (células de 0.25–0.5 m, 35–40 % de
rejeições), 10–50× o do caso de campo, e exige o recálculo da Jacobiana dentro do
Newton (`slow`) na fase quase estática (achado 3); a sensibilidade do período à
banda de chaveamento do gás é de ±7–10 %. Dados em
`notebooks/data/severe_slug_lab_<caso>_<tese|fisico>.npz`; secção
correspondente no `nb_severe_slug`.

*Texto movido de `docs/lab_validation.md` (secção «Sensibilidade do caso de Jansen (nota da reorganização, 2026-10-14)») em 2026-10-14.*

Ao re-correr a validação com o lado direito compilado unificado (igual ao anterior a
1e−14), o caso de Jansen (1", arrasto físico, linha fechada, `slow`) manteve o ciclo
clássico com refluxo mas mudou o intervalo médio entre blowouts: 89.9 s com 7 blowouts
em 600 s contra 69.1 s com 9 na corrida gravada. Duas corridas perturbadas isolam a causa:
`rtol` 3e−3 → 2.9e−3 dá 57.9 s (8 blowouts) e o primeiro passo 1e−3 → 1.1e−3 dá 78.9 s
(8). Pressões (mín. 0.99–1.00 bar, máx. 1.326–1.327 bar) e pico de líquido (0.70–0.79 kg/s)
são reprodutíveis; o período deste caso irregular não é (58–90 s pelo mesmo modelo), o que
alarga a faixa "56–63 s" indicada acima e reforça que a comparação com os 46.6 s medidos
(com choke) pede o choke modelado (item 9). O teste noturno passa a exigir, para Jansen,
só o número de blowouts (±2), as pressões e o pico; Fabre caso 2 mantém ±5 % no período.

## Referências

[docs/lab_validation.md](../lab_validation.md) (visão geral). O texto integral está na secção «Detalhe» acima. Ver também 015, 031.
