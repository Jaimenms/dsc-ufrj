# Validação com dados de laboratório: visão geral

*Documentação do mfsim (índice no [README](../README.md)). O detalhe está em
[issues/030](issues/030-validacao-laboratorio.md) (casos, resultados, sensibilidade do caso de
Jansen), [015](issues/015-fechamento-de-bolhas.md) (fechamento de bolhas da tese),
[028](issues/028-criterio-de-boe.md) (critério de Bøe) e [031](issues/031-numerica-a-1-atm.md)
(numérica a 1 atm).*

| caso JSON | instalação | fonte | medido |
|---|---|---|---|
| `lab_jansen` | ar–água 1", linha 9.1 m a −1°, riser 3 m, saída atmosférica, choke no topo (não modelado) | Jansen, Shoham & Taitel (1996) | 46.6 s |
| `lab_fabre2` | ar–água 2", linha 25 m a −0.57°, riser 13.5 m | Fabre et al. (1990) via Tengesdal (2002) | ≈ 137 s, 126–235 kPa |
| `lab_fabre1` | idem, vSG0 = 0.45 m/s (penetração contínua) | idem | ≈ 70 s, 128–210 kPa |
| `lab_tengesdal1` | ar–óleo 3", linha 19.8 m (+ tanques, 85 m efetivos) a −1°, riser 14.6 m | Tengesdal (2002) | 79 s, 121–222 kPa |

Receita dos casos: partida de linha fechada (`shut_in`, rampa de 20 s), padrão fixo com as
regras físicas (anular > 90 %, massa virtual 0.5, bolhas < 1 %), arrasto físico de bolhas,
γL = 100, `tol_range` 0.01, `slow_exit` e JAX.

```bash
uv run mfsim run cases/severe_slug/lab_fabre2.json --backend jax
uv run mfsim val severe_slug_lab_fabre2_fisico --plot      # período e pressões contra os valores publicados
```

Resultado em uma linha: com o fechamento de bolhas da tese nenhum caso cicla; com o
arrasto físico, Fabre 2 dá 135.8 s contra 137 medidos, Tengesdal 64–68 s contra 79 e Jansen
56–63 s contra 46.6 (ensaio com choke, que alonga o ciclo), sempre com a pressão máxima
certa; o caso de penetração contínua sai qualitativamente certo. Os ficheiros são
`notebooks/data/severe_slug_lab_<caso>_<tese|fisico>.npz` e a secção correspondente do
`nb_severe_slug`.
