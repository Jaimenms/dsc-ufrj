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

## Referências

[docs/lab_validation.md](../lab_validation.md); 015, 031.
