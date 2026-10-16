# 042 · Modos locais instáveis na curva: não são troca de fechamentos

| | |
|---|---|
| tipo | estudo |
| área | fechamentos |
| estado | feito (2026-10-16); sem alteração do modelo; o limiar de bolhas a 10 % fica como opção a decidir |
| onde | `regime.low_void.alpha`, `regime.vertical` (opções já existentes); espetro com `mfsim.compiled.get_jax` |
| dados | `runs/maps/prototype/` (fora do git): `bend_modes.py`, `slow_modes.py`, `bend_refine.py`, `low_void_test.py`; `runs/lowvoid/` (validação com o limiar a 10 %) |

## Pergunta

O estado estacionário do caso de referência com choke e IPR tem autovalores instáveis
rápidos (períodos abaixo de 1 s, crescimento de 5 a 28 /s), todos presos às células do ponto
baixo da curva, a −3°, 9° e 32°. A hipótese era que viessem da troca de fechamentos nessa
zona: a regra de fração baixa (bolhas abaixo de 1 %, rampa até 2 %), onde as células do ponto
baixo estão (1,5 a 2,4 % de gás), e a mistura estratificado–bolhas por ângulo (25° a 65°). Se
fosse isso, suavizar as transições tiraria estes modos, baixaria o custo do integrador e
deixaria o mapa de estabilidade pelos autovalores (bifurcação de Hopf) funcionar.

## O que se encontrou

1. **Não é perda de hiperbolicidade.** As velocidades características do modelo são reais em
   todas as células do estacionário.
2. **Suavizar as transições não os tira.** Deslocar a rampa de fração baixa para 0,5–1 % ou
   3–6 %, alargar a mistura por ângulo e combinar as duas só muda a célula onde o modo vive,
   sempre no ponto baixo. Alargar só a mistura por ângulo nem converge no Newton.
3. **Não há modo lento que marque a fronteira.** Na continuação da abertura a 15 bar
   (0,10 a 1,0) o único modo lento e global é amortecido (−0,42 ± 0,12–0,20j, período de
   30 a 50 s) e não cruza o eixo perto da fronteira do mapa dinâmico (entre 0,17 e 0,22). O
   modo local é instável em todas as aberturas, incluindo as que a dinâmica dá estáveis, onde
   satura com amplitude desprezável (0,005 bar). Passar a golfada ou não é um efeito não
   linear: os autovalores do estacionário não desenham este mapa.
4. **Cresce com a resolução.** Refinamento uniforme da malha, no ponto estável 15 bar /
   abertura 0,13:

| malha | estados | passos em 3000 s | crescimento máximo |
|---|---|---|---|
| referência | 112 | 9 162 | 27 /s |
| 2× mais fina | 224 | 64 727 | 37 /s (58 rad/s) |
| 4× mais fina | — | falha aos 821 s | — |

   É uma instabilidade de ondas curtas do estratificado num ponto baixo quase cheio de líquido
   (o padrão físico ali seria disperso ou intermitente). A malha grossa de referência
   amortece-a; malhas mais finas pagam-na em passos e acabam por não integrar.
5. **Subir o limiar de bolhas reduz o efeito mas muda a física.** No mesmo ponto:

| limiar de bolhas | gás na curva | crescimento máximo | passos em 3000 s |
|---|---|---|---|
| 1 % (atual) | 1,5–2,4 % | 27 /s | 9 162 |
| 5 % | 8–11 % | 5,4 /s | 16 283 |
| 10 % | 13–18 % | 3,5 /s | 1 251 |

   Validação com o limiar a 10 %, nos mesmos tempos simulados:

| caso | passos 1 % | passos 10 % | período 1 % | período 10 % | medido |
|---|---|---|---|---|---|
| referência, 3000 s | 15 602 | 19 477 | 799 s | 857 s (pico 5,4 contra 6,5 kg/s) | — |
| Fabre 2, 1200 s | 117 508 | 84 180 | 135,8 s | 139,4 s | 137 s |
| Jansen, 600 s | 98 214 | 90 948 | 56–63 s | 66,9 s | 46,6 s |
| Tengesdal 1, 800 s | 50 439 | 24 066 | 64–68 s | 78,4 s | 79 s |

   O Tengesdal melhora muito e com metade dos passos, o Fabre 2 quase não muda, o Jansen
   piora (o ensaio tem um choke não modelado) e o pico do caso de referência baixa 18 %.

## Leitura

A proposta de suavizar os fechamentos na curva partia de uma hipótese errada e não foi
aplicada. O limiar de bolhas já é uma opção do JSON (`regime.low_void.alpha`); passar o padrão
a 10 % é uma decisão de modelo, com o resultado misto acima, e fica para o dono. O mapa de
intermitência severa faz-se por simulação dinâmica (043); um mapa por autovalores exigiria
primeiro um fechamento fisicamente adequado para o ponto baixo quase cheio de líquido.

## Referências

Issues 017 (regra de fração baixa), 008 (mistura por ângulo no joelho), 018 (massa virtual),
030 (laboratório) e 043 (mapa). Storkaas e Skogestad (2007) para o mapa por bifurcação de Hopf
com o choke como parâmetro.
