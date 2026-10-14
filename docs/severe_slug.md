# Intermitência severa: modelo, numérica e estudos

*Documentação do mfsim (índice no [README](../README.md)).*

## Intermitência severa (`notebooks/nb_severe_slug.ipynb`)

Geometria linha–riser (`mfsim/cases/severe_slug.py`): 20 m horizontais, 300 m
descendentes a −3°, riser de 20 m e 5 m até o separador (4 bar), D = 50 mm,
vSL = 0.3 m/s, vSG ≈ 0.1 m/s. Um único duto com padrão estratificado na linha e
bolhas nos trechos quase verticais (`regime_vertical`), esquema `fv_upwind`,
partida transiente a partir de uma estimativa inicial.

Resultado: ciclo-limite de intermitência severa (18 ciclos em 8000 s com
período ≈ 312 s na malha fina; 24 ciclos com período ≈ 230 s na malha grossa) —
bloqueio do riser por líquido e pressurização da linha, *blowdown* (fração de gás
no riser de 0 a 0.9 em ~20 s) com picos de líquido na saída de 10–20× a
alimentação, reenchimento e repetição; balanço de massa de líquido fechado em
~2 % (verificado no notebook). A trajetória gravada está em
`notebooks/data/severe_slug_run.npz` (malha fina, 8000 s, ~80 min de CPU) e
`notebooks/data/severe_slug_run_long.npz` (malha grossa, 8000 s, ~50 min).

**Conservação de massa e escolha das variáveis de estado.** Com os estados
primitivos (αG, P) nas células, o estoque de líquido no duto descola da integral
(entrada − saída) em degraus, um por blowdown (≈ 5 kg por slug; 2.1 % do líquido
alimentado em 8000 s na malha fina; 5.6 % em 3000 s com a partida bloqueada, que
tem muito mais slugs). É erro numérico da integração temporal da forma
não-conservativa (dαG/dt, dP/dt) somado ao limitador/projeção nos limites αG = 0
e 1, não uma propriedade do modelo. Solução implementada (`{"conservative": True}`
em `pipe.extra`, esquema `fv_upwind`): o BDF é aplicado às massas por volume
T(x) = (ρG·αG, ρL·αL) — resíduo T(x_{n+1}) − Σ a_j T(x_{n−j}) − γh f(x_{n+1}) = 0 —
resolvido por Newton nas mesmas variáveis primitivas, com dT/dx analítica
(`NetworkModel.transform_jacobian`); os fluxos de massa nas faces são
multiplicados por um limitador da célula doadora (rampa C1 em αfase/`flux_band`,
fluxo nulo quando a célula a montante não tem a fase), que garante positividade
sem o limitador de fração de gás e mantém o balanço exato por célula a cada passo
(`tests/test_conservative_residual.py`). Como T é singular em αG = −γG/γL ≈ −0.01,
os limites admitidos para αG são [−0.005, 1.005] (um pequeno *undershoot* evita que
o Newton projete αG em 0 a cada iteração). Custo: a pressão passa a ser escrava da
massa de líquido pela compressibilidade (δP/P = γL·δmL/mL), de modo que o erro de
truncamento da massa aparece como "ruído" de pressão e o controle de passo
responde com passos menores. Usar as massas como *estados* (`fv_upwind_cons`)
foi descartado: P ∝ mL^γL é uma não-linearidade extrema para o Newton.

Resultado da comparação (partida bloqueada, malha grossa 15 + 5, 3000 s, rtol
3e-3; `notebooks/data/severe_slug_blocked_{primitive,conservative}.npz`): mesma
dinâmica (primeiro slug em ~12 s, 12–13 blowdowns, intervalo ≈ 240–250 s, picos de 8.5–8.9 kg/s);
erro de massa medido com a integral dos fluxos de contorno **consistente com o
BDF** (`ImplicitBDF.accum`, mesma recorrência dos estados — a regra dos
trapézios nas saídas de 2 s não é exata): primitivas −43 kg de líquido (2.7 % do
alimentado) e +0.28 kg de gás (8.5 %); conservativa 0 a menos de arredondamento
(< 1e-9 relativo, também em `tests/test_conservative_residual.py`). Custo:
132.7 mil passos contra 91.8 mil (1.45×; 25 min contra 18 min de CPU).

**Joelho curvo e transição suave dos fechamentos por ângulo.** Com o traçado
poligonal a inclinação salta de −3° para 90° entre duas células e o fechamento
muda de forma abrupta (estratificado → bolhas, chave dura em |θ| > 45°). Dois
ingredientes removem a descontinuidade espacial: `elbow_radius`/`n_elbow` em
`severe_slug_case` (arco tangente à linha e ao riser, elementos com ângulos
intermediários) e `vertical_blend` (graus) em `pipe.extra` (peso do padrão
vertical como rampa C1 entre 45° ± Δ; os fechamentos são somados com pesos
(1 − w) e w, `TwoFluidModel.vertical_weight`). A mistura sem o arco só atua na
face do joelho e piora a convergência (metade de cada fechamento entre duas
células "puras"); com o arco ela ajuda. Resultado (R = 5 m, 4 elementos, Δ = 20°,
partida bloqueada, 3000 s, `notebooks/data/severe_slug_blocked_*_arc.npz`): o
ciclo muda qualitativamente para o *blowdown* "de livro" — a pressão na entrada
cai quase até a do separador (4.2–6.1 bar contra 5.7–6.4 bar com o joelho
anguloso), o intervalo entre *blowdowns* quase triplica (≈ 660–700 s contra ≈ 240–250 s;
ciclos contados pelos mínimos da pressão na entrada) e os picos caem para
~6.8 kg/s (contra 8.5–8.9). As duas formulações concordam entre si (mesma faixa
de pressão e de período), ou seja, a sensibilidade é do tratamento do joelho, não
das variáveis de estado. Custo: 128 mil passos (primitivas, 24 min) e 112 mil
(conservativa, 22 min) — com o joelho suave a formulação conservativa fica mais
barata que a primitiva.

**Malha não uniforme.** `pipe_mesh(..., grading=...)` (razão último/primeiro
elemento por trecho, progressão geométrica; `Pipe.grading`,
`severe_slug_case(grading_down=, grading_riser=)`). O esquema de volumes finitos
já usa as distâncias entre centros de células, então não muda. Estudo
(conservativa, arco + mistura, partida bloqueada, 3000 s): uniforme 15 + 5
células (112 mil passos, 22 min), graduada 15 + 5 com razões 0.25/3 (células de
36 → 9 m na linha e 1.6 → 4.7 m no riser; 166 mil passos, 35 min) e uniforme
30 + 10 (262 mil passos, 72 min). As três dão o mesmo ciclo (blowdowns a cada
≈ 700 s, pressão 4.2–5.85 bar); os perfis de fração de gás e pressão junto ao
joelho da **graduada grossa coincidem com os da fina uniforme**, enquanto a grossa
uniforme difere (bolsão ruidoso, entrada de gás no riser mais difusa). Ou seja, a
graduação compra a resolução da malha fina no pescoço por metade do custo. A
malha fina graduada (0.8 m na base do riser) **não é utilizável** com o
tratamento atual da fase que desaparece: *chattering* da vazão de líquido no
joelho (período < 0.1 s, passos de 1 ms) — ver "Hiperbolicidade".

**Hiperbolicidade dos fechamentos** (seção própria no notebook). Autovalores da
matriz quase-linear (`TwoFluidModel.eigenvalues`): o fechamento estratificado é
mal posto para αG ≲ 0.003 **mesmo sem escorregamento** (limite de camada de gás
muito fina, |Im λ| ≈ 2–4 m/s), e é exatamente aí que toda célula cheia de
líquido é avaliada (αG cortado em 1e−3); o de bolhas é mal posto em baixa fração
com escorregamento (caso clássico). Tensão superficial não resolve: é dispersiva
(3ª ordem), não altera as características, e com σ = 0.05 N/m o corte capilar é
milimétrico. O que torna o mapa inteiro hiperbólico: padrão de bolhas abaixo de
αG = 1 % (`low_void_regime=2`, `low_void_alpha`) **mais** o termo de pressão
interfacial de Bestion/CATHARE com δ ≥ 1 (`interfacial_pressure`, derivadas em αG
e P incluídas). Nos testes de 80 s na malha fina graduada essa regularização
reduziu o *chattering* à metade (amplitude e frequência) mas não o eliminou;
relaxação mais longa, bandas mais largas e chaveamentos de gás mais largos
pioram.

**Modo acústico do tampão de líquido e viscosidade artificial.** A causa
residual do *chattering* (diagnóstico com saída a cada 5 ms) é um modo acústico
do tampão de líquido que se forma na base do riser após cada *blowdown* (≈ 8 m
de líquido entre o bolsão da linha e o gás do riser): vazão de líquido e pressão
oscilam a ≈ 16–20 Hz = c/2L (c = 258 m/s com o expoente 100). O esquema
escalonado com gradiente de pressão centrado não amortece acústica, o BDF2
quase não dissipa e o controlador de passo resolve a oscilação; o líquido
monofásico nas mesmas células é estável (a excitação vem das interfaces
gás–líquido). Remédio numérico: viscosidade artificial linear ν = C·c·h nas
equações de momento (`art_visc`, C = 0.01; some com h → 0 e a taxa ν·k² é
desprezível para ondas longas). Efeito (mesmo ciclo e balanço exato em todos):
grossa uniforme 112 → 85.7 mil passos (20 min), grossa graduada 166 → 145 mil
(35 min), fina uniforme 262 → 202 mil (58 min); a fina graduada, antes inviável,
passa a rodar (303 mil passos, 83 min — continua a mais cara: o modo do tampão
volta nas fases de reenchimento, com comprimento de onda ≫ h, fora do alcance da
viscosidade de malha). As cinco trajetórias dão 5 blowdowns em 3000 s com
intervalo de 680–730 s e pressão de entrada 4.15–5.86 bar; os picos de líquido
crescem com a resolução no pescoço (6.7, 8.6, 9.1 e 10.2 kg/s para grossa,
fina, grossa graduada e fina graduada). A regularização hiperbólica (`low_void_regime=2` + δ = 1.2) não altera o
custo nem o ciclo na malha grossa (85.4 mil passos) — fica como opção. BDF1
(Euler implícito) nas fases de toque foi testado e é igual ou pior.

**Condição inicial "bloqueada"** (`start="blocked"`): bolsão de gás estratificado
na linha na fração de equilíbrio (0.746, igual à observada no ciclo), riser cheio
de líquido, pressões hidrostáticas e bolsão a uma margem (0.3 bar) acima da
frente de líquido — o ciclo começa em poucos minutos em vez de ~2500 s.

Ingredientes numéricos que se mostraram necessários (todos em
`mfsim/discretization/staggered.py` e `mfsim/integrators/implicit.py`):
coeficientes de massa com a fração de gás real da célula (compressibilidade do
gás some quando a fase some; acumulação de líquido vai para a pressão);
relaxação da vazão da fase ausente para a condição sem escorregamento com base
na célula doadora (chaveamento abrupto — a mistura suave "vaza" fração da célula
vizinha e sustenta fluxo de fase inexistente); chaveamento *upwind* com largura
adaptativa às vazões locais; recuo do BDF2 para BDF1 quando estados são
projetados nos limites físicos; recálculo da Jacobiana quando o Newton converge
lentamente; compressibilidade artificial do líquido (expoente 100) para reduzir
a rigidez acústica. Limitadores "generosos" de fração de gás produzem ciclos
falsos por destruição de massa — o balanço de massa deve ser sempre verificado.

**Aceleração sem perder o ciclo** (seção "Aceleração" do notebook). Onde vai o
tempo na malha grossa de referência: 14 ms por passo, 23.6 avaliações do lado
direito por passo (metade em Jacobianas coloridas, 25 cores — o padrão real só
permitiria 22), 92 % dos passos nas fases com gás no riser (h ≈ 14 ms, toque
acústico do tampão e do bolsão), Newton com ≈ 6 iterações por não-suavidade dos
fechamentos (Jacobiana nova a cada passo não muda nada). GPU não ajuda (100–200
estados, custo dominado por latência de chamadas numpy, não por vazão);
paralelismo só entre simulações. Alavancas que funcionaram, medidas em janelas
de reinício na fase cara (`archive/scratch_2026-10/py/restart_ring.py`, `archive/scratch_2026-10/py/restart_probe.py`)
e confirmadas em simulações completas de 3000 s (mesmo ciclo, balanço exato):

| configuração (malha grossa, arco, conservativa) | passos | CPU |
|---|---|---|
| referência: ν = 0.01·c·h | 85.7 mil | 20 min |
| ν = 0.05·c·h + orçamento de Newton 20 iterações (`max_newton`) | 59.2 mil | 14 min |
| + tolerância absoluta nas vazões 0.03 kg/s e em P 0.1 bar (`atol_q`, `atol_P`) | 23.2 mil | 5 min |
| + tolerância absoluta 0.05 kg/s e 0.2 bar | 21.8 mil | 4.8 min |

A tolerância absoluta nas vazões é a alavanca decisiva: o controlador deixa de
resolver o toque de ±0.03 kg/s (6 % da alimentação) e o BDF2 o amortece
sozinho; a pressão de entrada fica em 4.14–5.92 bar (referência 4.19–5.86) e os
picos em 6.3–6.8 kg/s (referência 6.7). Sem ganho ou piores: BDF1, relaxação da
fase ausente mais longa ou mais curta, γL = 30, tolerância relativa só em P,
Jacobiana a cada passo, passos diferentes nas diferenças finitas.

**Lado direito em JAX** (`mfsim/compiled.py`, `build_jax_rhs(model)` → `(rhs, jac)`).
Na primeira versão foi uma reescrita em JAX do subconjunto usado nos casos de
intermitência severa; desde a reorganização (secção "Lado direito em JAX para
toda a tese") é o mesmo código numpy executado com o backend JAX injetado.
Reproduz `NetworkModel.rhs` a 1e−13 (`tests/test_jax_rhs.py`). `jit` leva a avaliação de 0.43 para 0.08 ms
e a Jacobiana exata (`jax.jacfwd`) custa 0.9 ms contra 10.7 ms das diferenças
finitas coloridas; o integrador a recebe por `ImplicitBDF(jac_fun=...)`. A
Jacobiana exata não reduz as iterações de Newton (a dificuldade é a
não-suavidade dos fechamentos), mas o custo por passo cai 4–6×: a configuração
rápida da malha grossa passa de 5 min para **45 s** (22.6 mil passos; a
referência original levava 20 min) e a malha fina (30 + 10) com a mesma
configuração roda em **105 s** (39 mil passos; 58 min na referência), com o mesmo
ciclo (pressão de entrada 4.14–5.86 bar) e balanço exato. Armadilhas da
diferenciação automática encontradas: raiz quadrada de quantidade que se anula
(0 × ∞ = NaN na derivada; *double where*) e derivadas em quinas (clips em
α = 0, chaveamentos em q = 0), onde diferenças finitas e AD divergem
legitimamente. O que resta em Python (≈ 2 ms por passo: mapa conservativo,
especificações, fatoração esparsa) é o próximo alvo se for preciso mais.

## Severe slug como rede de dutos: vértices dinâmicos conservativos

`severe_slug_case(single_pipe=False)` monta o mesmo caso como rede: linha
(horizontal + descendente), pescoço (o arco como duto próprio), riser e saída,
ligados por vértices dinâmicos (volume `vertex_volume`), com a mesma condição
inicial bloqueada interpolada pelos dutos. A primeira tentativa, com a
formulação de vértice da tese (hold-up e pressão como estados, limitador do
hold-up nos limites), falhou de forma instrutiva: cada duto conservava massa a
0.01 kg, mas o vértice entre a linha e o pescoço, saturado de líquido, absorveu
741 kg dos 1590 kg alimentados sem os armazenar (o limitador anula a derivada do
hold-up e o volume não retém o excesso) — um sumidouro de massa que também
tornava o sistema rígido (475 mil passos). Correções, todas em
`NetworkModel` (numpy e JAX, o mesmo código): (1) em modo conservativo os vértices entram no mapa de
massas, T = (ρG·HG, ρL·(1 − HG)) por vértice, com a EDO d(m)/dt = (QG, QL)/V
exata e sem limitador — é uma mudança de variáveis, não um DAE: a relação
mL/ρL(P) + mG/ρG(P) = V é invertível e dá P e HG explicitamente (o DAE só
apareceria no limite incompressível, que a compressibilidade artificial
regulariza); (2) os fluxos de contorno dos dutos são limitados pelo conteúdo do
vértice quando ele impõe a fração de gás (o vértice é a célula doadora), e os
dutos que chegam a um vértice recebem a sua fração no fantasma de saída; (3) o
balanço do vértice usa exatamente os fluxos limitados das faces vizinhas, e o
acumulador do integrador (`boundary_net_fluxes`) inclui as vazões líquidas dos
vértices. Resultado: balanço exato do sistema (dutos + vértices) e passos 8×
menores; a formulação vale para junções e bifurcações (N → 1, 1 → N), verificado
no caso 10.2.2 com o esquema conservativo (erro de massa 2e−10 kg de líquido e
1e−8 kg de gás em 3000 s; `tests/test_conservative_residual.py`).

Comparação com o duto único curvado (mesma malha, formulação conservativa,
ν = 0.05, JAX, `tol_range` 0.01, 3000 s; seção "Rede de dutos" do notebook,
dados `notebooks/data/severe_slug_blocked_network_v*.npz`):

| configuração | blowdowns / intervalo | pico de líquido | P na entrada | passos | erro de massa |
|---|---|---|---|---|---|
| duto único curvado | 4 / 765 s | 7.9 kg/s | 4.14–6.00 bar | 25.9 mil | 4e−5 kg |
| rede, V = 0.002 m³ | 4 / 748 s | 11.7 kg/s | 4.07–5.94 bar | 65 mil | 2e−2 kg |
| rede, V = 0.005 m³ | 5 / 565 s | 10.0 kg/s | 4.04–5.87 bar | 61 mil | 1e−4 kg |
| rede, V = 0.05 m³ | 6 / 582 s | 4.4 kg/s | 4.32–5.80 bar | 35.5 mil | 1e−5 kg |

Com volume de vértice da ordem de uma célula (0.002 m³ contra 0.004 m³ por
célula) a rede reproduz o intervalo entre blowdowns e a faixa de pressão do duto
único; volumes maiores acrescentam estoque nas junções e mudam o ciclo (mais
curto, picos menores com V = 0.05 m³). Os picos de vazão da rede são mais altos
(junções sem perda de momento e vértice como célula de mistura). A rede custa
2–3× mais passos que o duto único: os vértices pequenos são rígidos.

## Filtro anti-refluxo removido

O código original tinha um filtro nas saídas dos dutos marcados com `check`
(forçava as vazões de saída acima de 1e−6 kg/s e anulava derivadas negativas).
Era um remendo para contornar a falta de informação de fração nas extremidades
em refluxo. Foi removido do modelo (numpy e JAX); o campo `check` continua a ser
lido dos arquivos originais mas não tem efeito. O caso 10.2.2, que o usava em
quatro arestas, reproduz o MATLAB exatamente como antes (o filtro nunca atuava
nessa trajetória). O refluxo passa a ser resolvido pelas equações e pelo
tratamento das extremidades descrito a seguir.

## Inversão do sentido do escoamento

A onda de fração de gás viaja com o escoamento, então a fração tem de ser
informada em toda extremidade por onde a massa entra (a dificuldade relatada na
tese). No esquema escalonado isso é feito pela célula fantasma: a fração do
fantasma é imposta pelo vértice (hold-up do vértice dinâmico, ou `alphaG`
especificado num vértice estático, quando a condição de contorno de αG é
Dirichlet) ou copiada da célula vizinha quando é "ausente"; o fluxo de massa na
face de contorno usa o fantasma como doador quando ele é imposto, e as faces de
contorno com fração imposta usam o estado do lado doador (célula em saída,
vértice em entrada) para os fechamentos, em vez da média. Demonstração em
`tests/test_reversal.py` (`archive/scratch_2026-10/py/reversal_demo.py`): duto entre dois vértices
de pressão, diferença de pressão que troca de sinal em 60–100 s, separador com
αG = 0 — com a fração imposta, a inversão enche o duto com o líquido do separador
(frente nítida, balanço fechado); com a fração livre, o que volta tem a
composição do que saía (gradiente nulo), hipótese que só serve quando não se
sabe o que há do outro lado. Nos esquemas nodais da tese a fração continua livre
nos nós de extremidade, como na tese.

## Tolerâncias por faixa de variável (`tol_range`) e transferência das lições do severe slug

Em vez de ajustar tolerâncias absolutas por tipo de variável (`atol_q`, `atol_P`),
cada problema declara as **faixas esperadas** dos estados e o integrador usa
uma única tolerância absoluta como fração dessas faixas:
`simulate(case, tol_range=0.01)` / `integrate_implicit(..., tol_range=...)`
usam `atol_i = tol_range · faixa_i` com `NetworkModel.state_scales()` (frações:
1; pressão e vazões: referências do caso). As faixas vêm de
`case.settings.extra["state_ref"] = {"PT": ..., "qL": ..., "qG": ...}`; sem
declaração, a estimativa automática usa as especificações, a IPR e o estado
inicial — que subestima vazões de pico (blowdown) ou superestima (IPR a pressão
nula), por isso os casos devem declará-las. `severe_slug_case` declara
qL = 20× e qG = 50× a alimentação (picos de blowdown) e a pressão máxima da
linha. Resultados: severe slug (JAX) `tol_range 0.01` → 25.9 mil passos, 50 s,
mesmo ciclo e balanço exato (equivalente à configuração ajustada à mão, sem
ajuste); gas lift intermitente 16 000 s `tol_range 0.005` → 1.5 mil passos,
6 s contra 13.7 s, ciclo intacto (`0.01` já distorce o mínimo de produção).

Quais lições do severe slug se transferem aos demais casos (medido):
- genéricas e já aplicadas a tudo: lado direito/Jacobiana em JAX, `slow_exit`
  (recálculo da Jacobiana dentro do Newton, indispensável nos eventos de
  válvula do gas lift), `tol_range`, orçamento de Newton, acumulador de balanço
  de massa consistente com o BDF (`ImplicitBDF.accum`);
- balanço de massa dos esquemas legados nos casos da tese: transientes suaves
  conservam massa (10.1.1, 10.1.2, 10.2.2: erro < 0.01 % do alimentado, Galerkin
  e diferenças centrais); o caso de bolhas 10.1.3 perde 2.6 % do gás em
  Galerkin — a formulação conservativa só importa onde há fases que desaparecem;
- específicas do modelo a dois fluidos com o esquema escalonado (viscosidade
  artificial acústica, limitador de fluxo da célula doadora, joelho em arco com
  mistura por ângulo, regularização hiperbólica): não se aplicam ao tubing em
  Beggs & Brill nem às linhas monofásicas, que são o grosso das redes 10.3.x;
- o orçamento de Newton e as tolerâncias folgadas quase nada rendem no gas
  lift (custo nos eventos de *heading*, não em toque acústico).

## Identificação automática de padrão (regime 6) no severe slug

Objetivo: rodar o severe slug com a identificação de padrão da tese
(`identificaRegimeEstavel.m`: equilíbrios de Taitel–Dukler e Pots, critérios de
Kelvin–Helmholtz, Jeffreys, Froude e turbulência com chaveamento sigmoidal de
inclinação `k_switch`) em vez de prescrever estratificado na linha e bolhas no riser.

**Porte para JAX.** `closures.regime.equilibrium_fractions`/`identify_regime` (com `bk=JAX`)
reproduzem o numpy a 1e−14 (`tests/test_jax_regime_auto.py`): as grandezas
geométricas das tabelas de hD (FI, Λ) são constantes pré-calculadas, a busca na
tabela (argmin, constante por partes no estado) é feita sem gradiente
(`stop_gradient`, o que também evita propagar tangentes por n × 700 avaliações) e
as raízes que se anulam com as vazões (NL, T) usam `_safe_sqrt`. Armadilha de AD
encontrada: a sigmoide `1/(1+exp(clip(−k(x1−x2), ±700)))` tem tangente NaN quando
o argumento encosta no corte (exp ≈ 1e304 vezes uma derivada grande dá inf/inf);
com k = 1e6 isso acontece sempre que um critério está a 7e−4 do limiar. Trocada
pela logística estável (`jax.nn.sigmoid`, mesmos valores a 1e−300), também no
Beggs & Brill.

**O que a identificação da tese faz no severe slug** (estados gravados da corrida
de referência): linha descendente → estratificado; trecho horizontal de entrada →
intermitente (o ponto vSL = 0.3, vSG ≈ 0.1–0.3 m/s está a ~4 % do limiar de
Kelvin–Helmholtz); riser e horizontal de saída cheios de líquido → "anular"
(equilíbrio de Pots degenerado com vGS → 0, aL_anu = 0.04) ou empate 0.5/0.5 com
vGS = 0 exato; riser com gás → intermitente ou bolhas conforme aG_bol.

**Por que o integrador rasteja com o IPE da tese** (9834 passos até t = 1.5 s,
contra 101 até 2 s com padrão fixo; diagnóstico com saída densa de 1 ms):

1. células cheias de líquido: FG* = FG*c = 0 → sigmoide(0) = 0.5 em todo
   instante, e a fórmula Int = (1−Est)(1−Anu)(1−Bub) do original não é uma
   partição da unidade com indicadores fracionários (os pesos somavam 1.44);
2. na partida bloqueada o gás da entrada inverte o sentido no toque acústico
   (vGS de +0.5 a −0.7 m/s em 0.5 s); com vGS < 0 os modelos de equilíbrio dão NaN
   e o original cai em hD = 1e−6 → FG*c salta de 0.02 para 19.8, sign(vM) e aG_bol
   saltam também → os pesos vão de 0 a 1 em 1 ms, independentemente de k;
3. a tabela "mais próxima" muda hD_est em degraus de 2–5 %, logo FG*c também;
4. `flag_vertical` comparava |θ| em radianos com 90: nunca era vertical, e o
   critério de Froude do anular nunca se aplicava a risers.

**Correções e opções** (numpy e JAX, mesma implementação):

* sempre ativas, sem efeito quando os indicadores são 0/1: `flag_vertical` em
  graus; partição da unidade (Int = (1−Est)(1−Anu)(1−p_bub));
* `low_void_regime` passa a valer também no modo automático (sem gás não há
  padrão a identificar: bolhas abaixo de `low_void_alpha` com a rampa C1);
* `PipeParams.ipe_smooth=True`: sigmoides com largura *relativa* ao valor
  crítico (`k_switch` ≈ 20 → transição em ±10 % do limiar, igual para critérios
  de escalas tão diferentes como FG* ≈ 0.02 e K ≈ 1–10), equilíbrios avaliados
  com |vLS|, |vGS| (os modelos são co-correntes), hD interpolado na tabela
  (contínuo no estado; em JAX o vLS de equilíbrio é reavaliado com gradiente nas
  duas entradas que enquadram vLS), sentido do escoamento por tanh(cos θ·vM/0.02)
  e aG_bol = |vGS|/(|1.25 vM + ub| + 1e−3);
* `PipeParams.ipe_local=True`: os critérios são avaliados no **estado local** em
  vez dos equilíbrios de regime permanente: nível e fração reais (hD, αG) no
  lugar de hD_est/aG_est (Kelvin–Helmholtz, Jeffreys, trajetória e turbulência
  julgam a interface que existe, e deixa de haver busca em tabela nesses
  critérios); fração de bolhas max(aG_bol, αG) (máximo suave); estratificado
  impossível em dutos verticais (os critérios degeneram com cos θ → 0 e o
  indicador de sentido oscila com vM ≈ 0); fração de gás acima de 0.9 é filme de
  parede (anular) qualquer que seja a velocidade do gás. Motivo: depois do
  *blowdown* fica um bolsão quase parado no joelho (αG = 0.8, vLS ≈ 0) e o mapa
  de regime permanente, que só olha as velocidades superficiais, chama-o de
  "bolhas" sempre que a velocidade do gás, que toca a ~30 Hz, cai (aG_bol entre
  0.02 e 0.6); o fechamento alterna bolhas ↔ intermitente a cada 16 ms e
  sustenta o toque (passos de 0.1–1 ms em toda a fase rápida).

Custo da partida bloqueada (JAX, mesma configuração da referência): até 10 s,
`ipe_smooth` com k = 20 e bolhas abaixo de 1 % gasta 170 passos (padrão fixo:
370; k = 50: 427); até 101 s (primeiro *blowdown* incluído), `ipe_smooth`
13458 passos e `ipe_smooth + ipe_local` 986. O modo da tese (k = 1e6, absoluto)
continua disponível e é o usado nos testes de equivalência.

**Resultado (3000 s, malha 15/5, arco, conservativa, ν = 0.05, JAX, `tol_range`
0.01; dados em `notebooks/data/severe_slug_blocked_regime6_*.npz`):**

| padrão | passos | CPU | blowdowns (s) | intervalo | pico de líquido | P_in (bar) | produção média | erro de massa |
|---|---|---|---|---|---|---|---|---|
| fixo (estratificado + bolhas no riser, mistura 20°) | 25.9k | 50 s | 30, 740, 1518, 2324 | 765 s | 7.9 kg/s | 4.14–6.00 | 0.515 kg/s | 0 |
| IPE, `ipe_smooth + ipe_local`, k = 20, bolhas < 1 % | 35.3k | 207 s | 34, 870, 1726, 2588 | 851 s | 4.1 kg/s | 4.01–5.86 | 0.498 kg/s | 0 |
| IPE, só `ipe_smooth`, k = 20, bolhas < 1 % | 175.9k | 2034 s | 46, 876, 1750, 2628 | 861 s | 5.7 kg/s | 4.01–5.88 | 0.495 kg/s | 6e−4 kg |
| IPE, só `ipe_smooth`, k = 50, bolhas < 1 % | 163.1k | 1892 s | 36, 876, 1760, 2684 | 883 s | 5.4 kg/s | 3.98–5.91 | 0.491 kg/s | 1e−4 kg |
| IPE da tese (k = 1e6), com ou sem bolhas < 1 % | — | — | não passa de t ≈ 2 s (27k passos em 4.5 min) | | | | | |

A identificação automática reproduz o ciclo de intermitência severa (mesma
alimentação 0.530 kg/s, mesma faixa de pressão, intervalo entre *blowdowns* 11–15 %
maior) com picos de líquido menores e mais largos: o padrão identificado no riser
durante o *blowdown* é intermitente/anular (não bolhas), com mais atrito
interfacial. As três variantes do IPE concordam entre si no ciclo; o modo local
custa 5× menos passos e 10× menos CPU do que o suave sozinho, que passa a fase
rápida a alternar bolhas ↔ intermitente no bolsão do joelho. O que ela identifica ao longo do ciclo (mapa no notebook): linha
descendente estratificada, bolsão do joelho estratificado, riser cheio de líquido
= bolhas (regra de fração baixa), riser com gás = intermitente → anular (filme) à
medida que esvazia, trecho horizontal de entrada estratificado/intermitente
conforme a fração de gás. O custo por passo é ~3× o do padrão fixo (tabelas de
equilíbrio de 2 × 700 entradas por nó em cada avaliação).

## Reconstrução de α nas faces e o arrasto espúrio da fase virtual

**MUSCL/TVD da fração de área nas faces** (`alpha_face="muscl"`, numpy e JAX:
declive minmod a partir da célula doadora; `fv_order=2` faz o mesmo para a
velocidade doadora). Não dá nitidez à frente: na malha grossa o ciclo é igual
(intervalo 775 s contra 765 s) e a largura mediana da transição 0.1–0.9 de αG no
riser fica em 2.9 m com e sem reconstrução (1.7 m na malha fina). No esquema
escalonado os fluxos de massa são as vazões das faces (estados); a fração nas
faces só entra no momento. A difusão da frente vem da dinâmica do momento
(velocidade doadora, relaxação da fase ausente, limitador), não da interpolação
de α.

**Arrasto espúrio nas células cheias de líquido** (experimento
`archive/scratch_2026-10/py/plug_ring.py`, seção correspondente no `nb_severe_slug`): uma coluna de
20 m do modelo a dois fluidos cheia de líquido, com fechamento estratificado e um
degrau de 0.2 bar entre pressões fixas, trava em 0.49 kg/s, enquanto o líquido
monofásico acelera a 3.6 kg/s em 3 s. O balanço de momento mostra GammaL =
1.96 N/m contra 0.05 N/m de atrito de parede: é a força interfacial com a camada
de gás virtual (αG cortado em `alpha_min` = 1e−3) que a relaxação da fase ausente
mantém parada, porque o alvo da relaxação usa a fração real (nula → vazão de gás
nula → vG = 0 enquanto vL = 0.27 m/s). O estratificado exerce então 40× o atrito
de parede a 0.27 m/s e 58× a 1 m/s (anular 8–10×; bolhas e intermitente nada);
com vG = vL todos os fechamentos reproduzem o monofásico.
`relax_target="noslip"` (alvo sem escorregamento com a fração da célula doadora
com piso em `alpha_min`, nas duas fases; só no modo conservativo, onde o
limitador de fluxo impede que a vazão virtual drene massa de uma célula sem a
fase) elimina o arrasto: a coluna volta a 3.65 kg/s e o decaimento do modo
acústico da coluna fechada (c/2L = 5.6 Hz) passa de 1.33 para 0.63 1/s
(monofásico: 0.32). Uma primeira versão usou a fração **média da face** em vez
da doadora e fazia o severe slug rastejar (h ≈ 1e−7 s): numa face entre uma
célula cheia de gás e outra com 13 % de gás o filme de 1 % era arrastado como se
fosse 44 % do tubo, à velocidade do gás do blowdown. Com o alvo correto, o
severe slug de referência custa o mesmo (25.2k passos, 47 s) e mantém o ciclo
(blowdowns em 38, 704, 1484, 2284 s, intervalo 749 s contra 765 s, produção média
igual), com pressão máxima de entrada 5.92 bar em vez de 6.00 e picos de líquido
7.1 em vez de 7.9 kg/s: o arrasto espúrio nas células cheias de líquido segurava
um pouco o líquido na linha e no riser. É a variante fisicamente consistente; os
resultados anteriores foram obtidos com o alvo de fração real e devem ser lidos
com essa ressalva. Resposta à pergunta da pendência: as células cheias de
líquido do modelo a dois fluidos não excitam o modo do tampão; são mais
amortecidas do que o monofásico, por um arrasto espúrio que liga e desliga onde a
fração cruza a banda da relaxação. O tampão entre dois bolsões de gás, fechado,
oscila no modo "mola de gás" (1.1 Hz medido, 0.94 Hz previsto por
k = 2A²γP/V_bolsão e m = ρL A L_tampão), não no acústico.

## Volumes finitos escalonados para o modelo de deriva (Beggs & Brill)

O modelo de deriva da tese tem 3 equações (massa de cada fase e momento da
mistura) com o hold-up algébrico, da correlação de Beggs & Brill em função de
(P, qL, qG), e no esquema nodal as vazões são estados e o hold-up é diferenciado
no tempo pela matriz MA. No arranjo escalonado isso não fecha a contagem: duas
equações de massa por célula pedem dois estados de célula, e a divisão da vazão
da mistura entre as fases numa face é dada por uma relação algébrica. A
formulação adotada (`StaggeredFVScheme._rhs_drift`, numpy e JAX):

* células: hold-up HG e pressão como estados, com as mesmas equações de massa
  do modelo a dois fluidos (modo conservativo com limitador de fluxo, mapa de
  massas e acumulador de balanço reutilizados); o hold-up inicial vem da
  correlação quando a condição inicial nodal não o traz (`ic_holdup`);
* faces: qL e qG como estados; a soma qM obedece à equação de momento da
  mistura da tese (fluxo ρS·vM² com velocidade doadora, gradiente de pressão,
  tensão de parede bifásica e gravidade, por `_momentum` com ρS e α = 1);
* divisão entre as fases: a relação de escorregamento HG_BB(P, qL, qG) = HG da
  face (hold-up doador das células) é imposta na forma diferenciada no tempo com
  estabilização de Baumgarte, d/dt[HG_BB − HG_f] = −(HG_BB − HG_f)/τ
  (`drift_relax`, 0.05 s), o que dá uma EDO explícita para qG com o coeficiente
  ∂HG/∂qG − ∂HG/∂qL (> 0; piso `drift_den_min`). É a mesma ideia do modelo nodal
  (hold-up algébrico diferenciado no tempo), sem DAE, sem matriz de massa e sem
  inverter a correlação. As derivadas do hold-up são as diferenças finitas da
  tese (`bb_derivatives="fd"`) ou jvp exatas avaliadas a 1e−6 kg/s do lado do
  escoamento (em vazão de gás exatamente nula a correlação está na região
  cortada e a derivada automática é zero, o que derrubava o coeficiente no piso
  e fazia o lado direito explodir).

Verificação no poço 10.3.1 (`tests/test_fv_drift.py`, `archive/scratch_2026-10/py/well_fv.py`,
figura `notebooks/figs_well_fv.png`): estacionário com resíduo 1e−11 e todos os
autovalores da Jacobiana negativos (a primeira versão, com o sinal do coeficiente
trocado, tinha autovalores +4400 e parava em 3 ms); transiente de fechamento e
reabertura do choke em 232 passos / 1.4 s (numpy), líquido no topo 5.8–25.8 kg/s
contra 5.9–23.4 no esquema nodal, pressão de fundo 143.8–158.1 contra
146.1–158.4 bar, balanço de massa exato com o acumulador BDF; reprodução
numpy/JAX a 1e−13 no modo "fd".

**Severe slug com Beggs & Brill** (`severe_slug_case(model=2)`): não é
simulável com esta correlação, e o motivo é físico. Partindo da condição
bloqueada (bolsão com 75 % de gás sobre líquido a 0.3 m/s, riser cheio) a
restrição de escorregamento exige vazões de gás de vários m/s para que a
correlação "explique" 75 % de gás com esse líquido, cria uma recirculação de
±6 kg/s nas faces da entrada e o integrador para em t ≈ 2.3 s, com derivadas
exatas ou por diferenças finitas. Um bolsão parado sobre líquido em movimento
(escorregamento infinito) não é um estado da correlação, que foi ajustada em
escoamento co-corrente permanente. Partindo do hold-up da própria correlação
(`ic_holdup="correlation"`) a mesma geometria vai para um estado estacionário em
157 passos (P_in 4.66 bar, produção igual à alimentação, 20–30 % de gás no
riser): a correlação amarra a vazão de gás à do líquido e não há mecanismo de
bloqueio do gás pela coluna de líquido, que é o que gera o ciclo. Para ter
severe slug com um modelo de deriva seria preciso uma lei de escorregamento em
velocidade (vG = C0·vM + vd) que admita gás parado com líquido em movimento, não
uma correlação de hold-up.

## Sensibilidade ao atrito interfacial do estratificado (item 2)

`PipeParams.fi_mult` multiplica o fator de atrito interfacial do estratificado e
`fi_vgt` é o v_GT de Andritsos & Hanratty (`inf` recupera Taitel & Dukler,
f_I = f_G); ambos entram nos fechamentos numpy e JAX (`ClosureInputs.fi_mult`,
`fi_vgt`). Severe slug de referência (padrão fixo, `relax_target="noslip"`,
JAX, 3000 s):

| atrito interfacial | passos | blowdowns (s) | intervalo | pico de líquido | P_in (bar) |
|---|---|---|---|---|---|
| tese (Andritsos & Hanratty, v_GT = 5.5 m/s) | 25.2k | 38, 704, 1484, 2284 | 749 s | 7.1 kg/s | 4.14–5.92 |
| × 0.5 | 30.6k | 54, 766, 1546, 2330 | 759 s | 7.5 kg/s | 4.17–5.91 |
| × 2 | 20.2k | 34, 726, 1506, 2272 | 746 s | 6.6 kg/s | 4.15–5.92 |
| Taitel & Dukler (f_I = f_G) | 27.3k | 38, 698, 1470, 2244 | 735 s | 6.5 kg/s | 4.13–5.92 |

O ciclo é pouco sensível: intervalo entre blowdowns dentro de ±2 %, pressão de
entrada igual, picos de líquido dentro de ±8 %, produção média igual. A forma de
Taitel & Dukler quase coincide com a da tese porque a velocidade superficial do
gás na linha raramente passa de v_GT = 5.5 m/s, onde o fator de Andritsos &
Hanratty se afasta de f_G. O período é fixado pela compressão do bolsão e pela
carga hidrostática do riser, não pelo atrito da interface. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_fi{05,20,taitel}.npz`.

## Massa virtual como regularização física (item 3)

Termo de massa virtual de Drew & Lahey nas equações de momento
(`PipeParams.virtual_mass` = C_vm, pesado pelo indicador de bolhas, e
`vm_lambda` = parâmetro da forma objetiva; numpy e JAX, `tests/test_jax_rhs.py`):

    IG·aG = F_G − K (aG − aL) − K (vG − vL)[(λ−2) ∂vG/∂x + (1−λ) ∂vL/∂x]
    IL·aL = F_L + K (aG − aL) + (reação),   K = C_vm·w_bolhas·αG·ρL·A

com a = Dv/Dt as acelerações materiais (no esquema escalonado, (dq/dt +
d(qv)/dx)/I, resolvido face a face como um sistema 2×2 e devolvido na forma
dq/dt) e as derivadas espaciais das velocidades por diferenças centrais nas
faces. A análise de características (`TwoFluidModel.eigenvalues(...,
virtual_mass, vm_lambda)`) refaz as linhas de momento da matriz quase-linear com
o mesmo acoplamento.

| mapa (αG 1e−3–0.6 × escorregamento 0–5 m/s) | fração mal posta |
|---|---|
| bolhas, riser 90° | 50.5 % |
| bolhas + massa virtual 0.5, forma simples (λ = 2) | 97.6 % |
| bolhas + massa virtual 0.5, forma objetiva (λ = 1 ou 0) | 0 % |
| estratificado 8.6° + bolhas abaixo de 1 % | 37.6 % |
| idem + massa virtual 0.5 (λ = 1) | 0 % |
| idem + pressão interfacial de Bestion δ = 1.2 | 0 % |

A forma simples (só a diferença de acelerações) piora a má-posição; a forma
objetiva de Drew & Lahey com C_vm = 0.5 torna o modelo hiperbólico em todo o
mapa, tal como a pressão interfacial de Bestion, mas com um termo físico (a
inércia da fase dispersa) em vez de um termo de regularização.

| severe slug (padrão fixo, `noslip`, JAX, 3000 s) | passos | CPU | blowdowns (s) | intervalo | pico | P_in (bar) | produção |
|---|---|---|---|---|---|---|---|
| referência | 25.2k | 47 s | 38, 704, 1484, 2284 | 749 s | 7.1 kg/s | 4.14–5.92 | 0.515 |
| massa virtual 0.5 (λ = 1) | 23.6k | 50 s | 44, 822, 1678, 2522 | 826 s | 7.6 kg/s | 4.02–6.02 | 0.504 |
| massa virtual 0.5 (λ = 1) + bolhas < 1 % | 21.9k | 46 s | 46, 796, 1624, 2450 | 801 s | 6.3 kg/s | 4.03–5.85 | 0.507 |
| massa virtual 0.5 (λ = 2) | 20.0k | 47 s | 44, 832, 1684, 2514 | 823 s | 7.8 kg/s | 4.02–5.93 | 0.505 |
| bolhas < 1 % + Bestion δ = 1.2 | 41.4k | 86 s | 36, 718, 1532, 2280 | 748 s | 5.9 kg/s | 4.16–5.85 | 0.515 |

Ao contrário da pressão interfacial (que não muda o ciclo mas custa 1.6× mais
passos), a massa virtual **muda o ciclo**: intervalo entre blowdowns 7–10 %
maior e pressão mínima de entrada 0.1 bar mais baixa, ao mesmo custo. O efeito
é o mesmo nas duas formas (λ = 1 e 2), portanto vem do acoplamento das inércias
no riser em bolhas (o gás acelera com o líquido, o blowdown é mais lento), não
da regularização. É um efeito físico de primeira ordem no período e entra na
lista de sensibilidades do artigo. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{vm05,vm05_lv,vm05_lam2,bestion}.npz`.

**Consequência prática: a malha fina graduada no pescoço voltou a ser viável.**
A configuração 30/10 com razões 0.25/3 (células de 0.8 m na base do riser), que
com o alvo de fração real e ν = 0.01 custava 302.6k passos e 83 min (*chattering*
nas células cheias de líquido), com `noslip` e a configuração rápida custa 77.9k
passos / 240 s, e com massa virtual 0.5 + bolhas abaixo de 1 % 68.9k passos /
189 s. Ciclos: 5 blowdowns em 3000 s, intervalo 710 s, picos 10.3 kg/s (igual ao
antigo 10.2) sem massa virtual; 4 blowdowns, intervalo 778 s, picos 9.7 kg/s com
massa virtual, ou seja, o alongamento do intervalo pela massa virtual (+7–10 %)
mantém-se com o refinamento. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_fine_graded_{noslip,vm}.npz`.

## Escoamento contracorrente no riser (item 1)

Diagnóstico na corrida de referência (padrão fixo: bolhas no riser): o riser está
com mais de 90 % de gás em 19 % do tempo (depois de cada blowdown), e nessas
células o filme de líquido residual **nunca drena**: desce em 5 % do tempo e sobe
a mais de 0.5 m/s em 28 %, com velocidade média +0.9 m/s contra +0.4 m/s do gás.
O fechamento de bolhas a 95–100 % de gás exerce sobre o líquido um arrasto de 3 a 4
ordens de grandeza acima da gravidade (GammaL de −300 a −20000 N/m contra 0.87
N/m de peso do filme, tabela no `nb_severe_slug`), e o líquido é expulso pelo topo em
vez de escorrer para a base. O fechamento anular, ao contrário, dá um balanço de
forças sensato no filme (atrito de Kowalski na parede, Andritsos & Hanratty na
interface): o filme drena até vG ≈ 6 m/s e passa a ser arrastado perto de 10 m/s,
coerente com a inundação de Wallis para D = 5 cm (vSG ≈ 8 m/s). Não há critério
de inundação explícito: ele emerge do balanço de momento do filme, que é o que um
modelo a dois fluidos deve fazer; os fechamentos co-correntes da tese só falham
onde o padrão prescrito é o errado.

Regra acrescentada (numpy e JAX, simétrica à de fração baixa):
`PipeParams.high_void_regime` (1 = anular) acima de `high_void_alpha` (0.9), rampa
C1 de ±0.05. Severe slug de referência (`noslip`, JAX, 3000 s):

| riser quase cheio de gás | passos | CPU | intervalo | pico | P_in (bar) | filme descendo | líquido subindo > 0.5 m/s |
|---|---|---|---|---|---|---|---|
| bolhas (referência) | 25.2k | 47 s | 749 s | 7.1 kg/s | 4.14–5.92 | 5 % | 28 % |
| anular acima de 90 % | 23.5k | 52 s | 821 s | 8.3 kg/s | 4.01–5.98 | 56 % | 6 % |
| anular acima de 90 % + massa virtual 0.5 + bolhas < 1 % | 16.8k | 39 s | 800 s | 6.1 kg/s | 4.00–5.85 | 66 % | 6 % |
| IPE local (anular acima de 90 % por regra) | 35.3k | 207 s | 851 s | 4.1 kg/s | 4.01–5.86 | 10 % | 1 % |

O filme a drenar alonga o intervalo entre blowdowns em ~10 % (o líquido que
volta à base do riser antecipa o bloqueio, mas o riser esvazia mais e precisa de
mais líquido para reencher). A configuração com as três regras físicas (anular
acima de 90 %, massa virtual, bolhas abaixo de 1 %) é a mais barata de todas e
passa a ser a recomendada; os fechamentos prescritos "estratificado na linha,
bolhas no riser" subestimam o intervalo em ~7 %. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{hv,hv_vm_lv}.npz`;
diagnóstico `mfsim riser` (`mfsim.analysis.riser_countercurrent`).

## Fechamento do riser no blowdown: intermitente contra bolhas (item 5)

Forças sobre o líquido no riser vertical nos estados típicos do blowdown
(αG 0.3–0.85, vG 1.5–6 m/s): o fechamento de **bolhas** dá GammaL de −19 a
−2400 N/m (arrasto do gás sobre o líquido, crescendo com αG e com o quadrado do
escorregamento), o **intermitente** (unidade de slug da tese) de −0.03 a −128 N/m
e o anular quase nada; só o intermitente tem correções de pressão grandes
(dPL até 1.5 kPa a αG = 0.3). As bolhas carregam o líquido pelo gás com muito
mais eficiência, o que explica os picos de 7–8 kg/s com bolhas contra 4–6 kg/s
nas corridas com identificação automática (intermitente → anular no riser).

Corridas com `regime_vertical=3` (intermitente acima de 45°, mistura de 20°),
`noslip`, JAX, 3000 s, com e sem as regras físicas (anular > 90 %, massa virtual,
bolhas < 1 %): **não há ciclo de severe slug**. Depois do primeiro blowdown
(42 s) e do reenchimento, o sistema vai para escoamento em golfadas contínuo pelo
riser: αG no riser 0.01–0.3, pressão de entrada ≈ 5.8 bar quase constante,
produção a oscilar entre 0.3 e 1.2 kg/s, riser com mais de 90 % de gás em 5 % do
tempo (19 % com bolhas); custo 101–112k passos (o escoamento em golfadas contínuo
é caro de resolver). O critério de Bøe na forma das fontes (Taitel 1986, Jansen
et al. 1996; sem fator de inclinação, ver a correção abaixo) dá vSL_mín = 0.06 m/s
para o caso, contra 0.30 m/s de alimentação: o caso base está **dentro** da
região de Bøe, e a ausência de ciclo com o intermitente é uma sensibilidade do
modelo ao fechamento do riser a baixa fração de gás, não uma previsão do
critério. O ciclo de severe slug obtido com bolhas prescritas no riser é
sustentado pelo fechamento: a 1e-2–0.3 de fração de gás as bolhas retêm o gás na
base do riser e permitem o bloqueio; o intermitente deixa-o subir como bolhas de
Taylor e o bloqueio não se forma. Consequência para os resultados anteriores: o
ciclo com bolhas é uma solução do modelo com aquele fechamento, não uma previsão
independente do padrão; a verificação natural é a comparação com dados de
laboratório (item 8).

**Varredura em vSL** (1.2 e 1.6 m/s, mesma geometria,
bolhas e intermitente no riser): os dois fechamentos dão **escoamento
permanente** (pressão de entrada constante, 5.9–6.1 bar; 356–3325 passos em
3000 s). Coerente com Taitel (1986): o critério de Bøe é condição necessária
para o bloqueio só quando a linha está estratificada, e a 1.2 m/s num tubo de
5 cm a linha já escoa em golfadas, com o gás arrastado pelo líquido. Com a forma
correta de Bøe todos estes pontos estão dentro da região (vSL_mín = 0.06 m/s em
300 m a 4 bar, 0.02 m/s em 1000 m a 2 bar); a passagem a escoamento permanente a
1.2 m/s vem da linha deixar de ser estratificada, não de Bøe. O caso de linha
longa abaixo (L_down = 1000 m, P_out = 2 bar) foi montado quando ainda se usava a
forma com sin θ (que dava vSL_mín = 1.03 no caso base e 0.24 na linha longa);
continua a ser o caso em que o ciclo é robusto ao fechamento do riser.

**Caso de linha longa** (antes "Bøe-consistente"; L_down = 1000 m, P_out = 2 bar, 50 células na linha,
6000 s, `noslip`, JAX; "regras" = anular > 90 %, massa virtual 0.5, bolhas < 1 %):

| riser | passos | CPU | blowdowns (s) | intervalo | pico | P_in (bar) |
|---|---|---|---|---|---|---|
| bolhas, sobrepressão inicial 0.3 bar | 105k | 590 s | 136, 4052 | 3916 s | 7.8 kg/s | 2.12–3.81 |
| bolhas, sobrepressão inicial 0.1 bar | 64k | 409 s | 134, 4062 | 3928 s | 8.2 kg/s | 2.12–3.81 |
| bolhas + regras | 57k | 341 s | 122, 3970 | 3848 s | 8.8 kg/s | 2.17–3.81 |
| intermitente + regras | 59k | 458 s | 110, 4386 | 4276 s | 8.9 kg/s | 1.99–3.81 |
| intermitente sem regras | — | — | não passa da partida (blowdown inicial violento com 0.3 bar; rasteja no riser cheio de líquido com 0.1 bar) | | | |

**Conclusão do item 5.** Onde o critério de Bøe prevê intermitência severa com
linha estratificada, o modelo a produz com qualquer fechamento do riser: ciclo
clássico (pressurização lenta da linha de 2.9 a 3.8 bar com o riser cheio de
líquido, blowdown, reenchimento), período 3850–4280 s (±10 % entre fechamentos),
picos 7.8–8.9 kg/s. No caso base (vSL = 0.30, L = 300 m, 4 bar; dentro da região
de Bøe, vSL_mín = 0.06) só o fechamento de bolhas sustenta um ciclo; com
intermitente o modelo dá golfadas contínuas, o que, com Bøe satisfeito, aponta
uma limitação do fechamento intermitente a baixa fração de gás no riser (a
comparação com dados de laboratório do item 8 é o teste directo). Para o artigo:
os resultados do caso base devem ser apresentados como dependentes do fechamento
a baixa fração de gás no riser; o caso de linha longa é o que demonstra o
mecanismo de forma robusta. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_boe_{bub,bub_dp01,bub_rules,int_rules}.npz`.

## Volume de gás a montante (item 6)

`severe_slug_case(upstream_volume=V)` troca o vértice de alimentação por um
vértice dinâmico de volume V (m³) que recebe as vazões de alimentação; o duto
recebe αG e P do vértice e as vazões da face de entrada passam a ser estados. É
o "gasoduto comprimível" que entra no critério de Bøe. Caso base com a
configuração recomendada (`noslip`, anular > 90 %, massa virtual, bolhas < 1 %;
o volume de gás da própria linha é ≈ 0.44 m³):

| V_up (m³) | passos | blowdowns (s) | intervalo | pico de líquido | ΔP_in |
|---|---|---|---|---|---|
| 0 (referência) | 16.8k | 786, 1618, 2452 | 833 s | 6.1 kg/s | 1.85 bar |
| 0.1 | 17.9k | 516, 1370, 2206 | 845 s | 6.4 kg/s | 1.86 bar |
| 0.5 | 17.6k | 632, 1100, 1976, 2878 | 749 s | 7.3 kg/s | 1.87 bar |
| 2.0 | 20.9k | 908, 1714, 2446 | 769 s | 9.3 kg/s | 1.91 bar |

O ciclo mantém-se e o pico de líquido cresce (6.1 → 9.3 kg/s). Mas o vértice
**não representa um gasoduto a montante**: é um tanque de mistura cuja fração de
gás vai para ≈ 0.05 (o líquido alimentado acumula até a composição de saída
igualar a de entrada), de modo que o volume de gás efetivo com V = 2 m³ é
~0.1 m³ e a taxa de pressurização da linha não muda (0.24 bar por 1000 s com e
sem o vértice, no caso de linha longa). O que falta é o vértice tipo separador
(já na lista de pendências): gás acumulado no topo, líquido a passar. Para
representar a tubulação a montante o caminho físico é prolongar a linha
horizontal de entrada (`L_h1`), onde o estratificado guarda o gás no topo e passa
o líquido por baixo; os resultados com L_h1 = 200 e 500 m (gás da linha de
entrada ≈ 0.3 e 0.74 m³, contra 0.44 m³ no trecho descendente) estão na tabela
da fronteira. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_{vup01,vup05,vup20,lh1_200,lh1_500}.npz`.

## Fronteira da intermitência severa: varredura contra Bøe e Taitel

Varredura com a configuração recomendada (padrão fixo, `noslip`, anular > 90 %,
massa virtual 0.5, bolhas < 1 %), linha de 1000 m a −3°, riser de 20 m,
sobrepressão inicial 0.1 bar, 9000 s. "Bøe mín" é o vSL mínimo do critério de
Bøe na forma de Taitel (1986) e Jansen et al. (1996), vSL ≥ P vSG/(ρL g (1 − αG) L),
com vSG à pressão do separador, αG = 0.5 e L = linha de entrada + descendente
(coluna recalculada em 2026-10-13: a versão anterior de `boe_criterion` dividia L
por sin|θ|, o que não consta das fontes e dava valores 19× maiores, como o 1.03 m/s
do caso base); "linha" é o padrão que a
identificação da tese atribui ao estado médio da linha (`mfsim scan`, `mfsim.analysis.scan_summary`).

| P_out (bar) | vSL (m/s) | vSG (m/s) | Bøe mín | linha | ΔP_in | blowdowns | intervalo | pico |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.15 | 0.15 | 0.019 | estratificada | 1.82 bar | 2 em 9000 s | 4714 s | 10.2 kg/s |
| 2 | 0.30 | 0.15 | 0.019 | estratificada | 1.62 bar | 3 | 3602 s | 8.6 kg/s |
| 2 | 0.50 | 0.15 | 0.019 | estratificada | 1.44 bar | 4 | 2858 s | 8.2 kg/s |
| 2 | 0.80 | 0.15 | 0.019 | estratificada | 1.33 bar | 4 | 2099 s | 7.2 kg/s |
| 4 | 0.30 | 0.12 | 0.019 | estratificada | 1.65 bar | 4 | 2265 s | 7.9 kg/s |
| 4 | 0.50 | 0.12 | 0.019 | estratificada | 1.49 bar | 4 | 1863 s | 7.3 kg/s |
| 4 | 0.80 | 0.12 | 0.019 | estratificada | 1.42 bar | 6 | 1349 s | 6.2 kg/s |
| 2 | 0.30 | 0.07 | 0.009 | estratificada | 1.63 bar | 2 | 6524 s | 8.6 kg/s |
| 2 | 0.30 | 0.29 | 0.036 | estratificada | 1.67 bar (bolhas) / 1.83 (intermitente) | 4 / 5 | 2017 s / 2140 s | 8.9 / 9.1 kg/s |
| 4 (L = 300 m) | 0.30 | 0.12 | 0.061 | estratificada | 1.8 bar (bolhas) / 0.2 (intermitente) | ciclo só com bolhas | 749 s | 7.1 kg/s |
| 4 (L = 300 m) | 1.2 e 1.6 | 0.12 | 0.061 | golfadas | < 0.05 bar | 0 | — | — |
| 4 (L = 300 m + 200 m de linha de entrada) | 0.30 | 0.12 | 0.039 | estratificada | 1.88 bar | 4 | 841 s | 6.9 kg/s |
| 4 (L = 300 m + 500 m de linha de entrada) | 0.30 | 0.12 | 0.024 | estratificada | 1.95 bar | 3 | 1077 s | 8.2 kg/s |

Leitura: (i) onde Bøe é satisfeito e a linha é estratificada o modelo dá o ciclo
clássico em todos os pontos, com intervalo a cair com vSL (o riser enche mais
depressa), com a pressão (gás mais denso) e com vSG (bolsão pressuriza mais
depressa), de 1350 s a 6500 s; (ii) onde a linha deixa de ser estratificada o
escoamento é permanente (vSL ≥ 1.2 m/s em 300 m, 4 bar); (iii) a fronteira de
Bøe do lado do gás **não** é testada por esta varredura: todos os pontos,
inclusive vSG = 0.29 m/s (vSL_mín = 0.036), estão dentro da região de Bøe, e o
modelo cicla com os dois fechamentos do riser (2017 e 2140 s); para chegar à
fronteira a 2 bar seria preciso vSG ≳ 2.5 m/s, onde a linha já não é
estratificada e Taitel (1986) exclui o bloqueio; (iv) o caso base de 300 m a 4 bar é
o único ponto em que a resposta depende do fechamento do riser; (v) gasoduto a
montante (linha de entrada de 500 m, ≈ 0.74 m³ de gás a mais): intervalo +29 %
e pico +35 %, o comportamento documentado nos experimentos para o volume a
montante. Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_scan_P{2,4}_v{015,03,05,08}.npz`,
`..._scan_P2_g{005,02,02_int}.npz`, `..._tol_lh1_{200,500}.npz`.

## Arraste de gotas no anular (item 7)

A tese não tem arraste e deposição de gotas; antes de modelar, estimou-se se o
arraste importa nestes casos. Fração arrastada de Ishii & Mishima (1989),
E = tanh(7.25e−7·We^1.25·Re_L^0.25) com We = ρG·jG²·D/σ·(Δρ/ρG)^(1/3) e
Re_L = ρL·jL·D/μL, avaliada em pós-processamento nas células do riser das
trajetórias gravadas (célula no `nb_severe_slug`):

| caso | riser > 90 % de gás | E > 0.1 nesse tempo | E máximo (vSG máx) | líquido produzido com E_topo > 0.1 |
|---|---|---|---|---|
| base, regras físicas | 21 % | 0 % | 0.12 (9.3 m/s) | 0 % |
| base + 500 m de linha de entrada | 47 % | 0.1 % | 0.16 (11.1 m/s) | 5 % |
| linha de 1000 m, 2 bar, bolhas + regras | 11 % | 1.0 % | 0.56 (28.9 m/s) | 1.1 % |
| linha de 1000 m, 2 bar, vSG = 0.29 | 16 % | 0.5 % | 0.61 (27.4 m/s) | 1.5 % |

O arraste só é apreciável (E > 0.5) nos poucos segundos do pico de gás do
blowdown dos casos de linha longa (vSG de 27–29 m/s no topo), e o líquido
produzido nesses instantes é 1–5 % do total. Conclusão: as gotas não mudam o
ciclo nem os picos destes casos; podem alterar a forma do pico de produção por
alguns segundos. Não se justifica um campo de gotas no modelo; se um dia for
preciso, o caminho é um termo de arrasto de gotas no fechamento anular
(fração E do líquido forçada à velocidade do gás com tempo de resposta
≈ 0.03–0.05 s), o que fica anotado.

## Lei do gás (item 4)

O gás é politrópico, ρG = ρG0·(P/P0)^(1/γG), com γG = 1.05 (quase isotérmico)
nos casos do severe slug. Sensibilidade a γG = 1.0 (isotérmico) e 1.3
(adiabático), configuração recomendada:

| caso | γG | passos | blowdowns (s) | intervalo | pico | ΔP_in |
|---|---|---|---|---|---|---|
| base | 1.05 | 16.8k | 786, 1618, 2452 | 833 s | 6.1 kg/s | 1.85 bar |
| base | 1.0 | 17.4k | 826, 1704, 2562 | 868 s (+4 %) | 6.3 kg/s | 1.85 bar |
| base | 1.3 | 18.0k | 634, 1310, 1994, 2674 | 680 s (−18 %) | 5.4 kg/s | 1.85 bar |
| linha de 1000 m, 2 bar | 1.05 | 59.1k | 302, 3640, 7506 | 3602 s | 8.6 kg/s | 1.62 bar |
| linha de 1000 m, 2 bar | 1.3 | 80.7k | 302, 3452, 6892 | 3295 s (−9 %) | 8.3 kg/s | 1.77 bar |

Gás mais rígido pressuriza o bolsão mais depressa (dP/dt ∝ γ para a mesma
massa acumulada) e é menos denso à mesma pressão, logo o ciclo encurta; a
amplitude de pressão, fixada pela carga do riser, não muda. É uma sensibilidade
de primeira ordem no período (da ordem da massa virtual e do filme a drenar),
e a escolha do expoente deve ser justificada caso a caso (o blowdown é rápido,
o enchimento lento: um γ efetivo entre os dois é defensável). Dados em
`notebooks/data/severe_slug_blocked_conservative_arc_tol_gammaG{10,13}.npz` e
`..._scan_P2_v03_gammaG13.npz`.
