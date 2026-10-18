# 060 · Caso offshore com gas lift: linha de gás, anular e válvula de retenção sobre a geometria do artigo 01

| | |
|---|---|
| tipo | feature |
| área | geradores, casos (rede), estado inicial |
| estado | feito (2026-10-18); opção `gas_lift=true` de `generators/offshore.py` (omissão `false`: o caso do artigo 01 regenera-se bit a bit, JSON e `.ic.npz`) |
| onde | `generators/offshore.py` (`gas_lift_geometry`, `_gas_lift_block`, `_gas_column`, `initial_state_guess` com o ramo, chaves `P_gas_bar`, `D_gas_line`, `D_annulus`, `L_inj`, `CV_check`, `V_tank`, `n_*`); casos `cases/paper01/gaslift/offshore_gl_P150.json` (com injeção) e `offshore_gl_P20.json` (sem injeção); corridas `runs/paper01/gaslift/` |
| dados | caso offshore de 2 km do artigo 01 (P_res 200 bar, PI 5, RGO 50, choke aberto) |

## Antes

O caso offshore do artigo 01 tem o poço, a linha e o riser; o gas lift só existia no caso 10.3.2 da tese (linha
de gás, anular e válvula com dutos monofásicos de gás, vazão de gás especificada à entrada). O plano do artigo 02
previa um gerador separado, dutos monofásicos de gás e um orifício em escoamento crítico.

## Depois

Pedido do dono (2026-10-18): um caso base mais complexo que o do artigo 01, com a mesma geometria, mais uma linha
de gás que sai da plataforma, chega à cabeça do poço, entra no anular e injeta na base do poço por uma válvula de
retenção; a especificação do gás é a **pressão na plataforma**; desligar o gas lift é pôr essa pressão baixa, e o
caso fica igual à geometria sem gas lift; **modelo a dois fluidos em todos os dutos**, incluindo os só de gás; o RGO
do poço é só o gás do reservatório. Diâmetros escolhidos com a prática de projeto de linhas de gas lift.

Topologia (a do caso 10.3.2 da tese, com os nomes do gerador): `gl_src` (fronteira, PT = P_gas, qL = 0) →
`gas_line` (o traçado da linha de produção ao contrário, com o mesmo joelho em arco e uma malha mais aberta, metade das células da linha de produção por segmento (`gas_mesh_factor` 0,5: 19 células, pedido do dono): patamar de 10 m, riser, arco, subida a +2° e horizontal até à cabeça do poço — pedido do dono; portas de entrada PT dirichlet e vazões livres) → `gl_wh`
→ conector → `an_top` → `annulus` → tanque `an_bot` (1 m³) → `check_an` (retenção) → `inj`; o poço parte-se em
`well_lower` (fundo → `wl_out` → conector → `inj`) e `well` (`inj` → cabeça). O vértice `inj` recebe a produção do
reservatório e o gás da retenção. `slow_exit = true` pelos eventos da retenção (033). O estado inicial
(`--ic blocked`) marcha a coluna de gás estática desde a plataforma pela linha de gás e pelo anular, calcula a
injeção com a lei da retenção e a pressão do tubing na injeção, e repete a IPR com o gás acrescentado.

Dimensões, da prática offshore brasileira (poços satélite com linhas flexíveis):

| elemento | valor | base |
|---|---|---|
| linha de gás | 4" (0,1016 m), rugosidade da linha, traçado da linha de produção (2325 m) com metade das células | par produção 6" + gas lift 4" dos poços satélite (especificação técnica dos poços satélite do Atapu: riser de serviço/gas lift de 4" flexível; sistemas de coleta do Roncador P-52/P-54, linhas de produção emparelhadas com linhas de gas lift) |
| anular | 9 5/8" × 5 1/2": D de área equivalente 0,171 m (área 0,0229 m²) | revestimento 9 5/8" 47 lb/ft (ID 0,2205 m) e coluna 5 1/2" (OD 0,1397 m, ID 0,124 m ≈ D_well 0,127 m do artigo 01); a área é o que conta (massa de gás e inércia); o diâmetro hidráulico (0,081 m) só mudaria o atrito, desprezável no gás a < 1 m/s |
| válvula | retenção CV = 1,5, a 30 m do fundo, tanque 1 m³ | válvula operadora logo acima do packer; q = 0,0075·CV·√(ρ ΔP[bar]) ≈ 1 kg/s (≈ 100 000 Sm³/d, a ordem das injeções por poço do Campos) a 60 bar de queda |
| pressão do gás | 150 bar (com injeção), 20 bar (sem) | manifold de gas lift a 150 bar: a coluna estática leva a base do anular a 170 bar; a 20 bar a retenção fica fechada |

A retenção não é um orifício em escoamento crítico: a vazão segue √ΔP, mas com ΔP ≈ 110 bar as oscilações de
±10 bar do tubing mudam-na só ≈ ±5 %.

**Fração de gás inicial do ramo (`gas_alpha`, omissão 0,998).** A primeira versão arrancava a linha de gás e o
anular com 2 % de líquido (o corte 0,02–0,98 do estado inicial do artigo 01). Esse líquido escorre durante horas:
acumula-se na base do riser da linha de gás (célula do joelho a 15 % de líquido aos 8000 s) e enche o tanque do
anular (HG de 0,98 para 0,44), e o escoamento lento do filme custa passos — sem injeção, 13 438 passos e 515 s de
CPU nos primeiros 2000 s, contra 2790 passos e 95 s com o ramo quase seco (α_G = 0,998, o modelo admite até
0,999), com a mesma dinâmica (P_in 14,5–30,4 contra 14,5–30,5 bar, picos de líquido 56,5 contra 57,0 kg/s).
Recalcular a Jacobiana só na saída do Newton (`slow_exit = false`) não ajuda (19 099 passos, 585 s). Os casos
gravados usam 0,998; a corrida sem injeção com 0,98 fica em `runs/paper01/gaslift/gl_P20_wet098.npz`.

Como regenerar os casos (parâmetros do artigo 01 mais a opção; P = 150 com injeção, 20 sem):

    uv run python generators/offshore.py --out cases/paper01/gaslift/offshore_gl_P150.json --param gas_lift=true \
        --param P_gas_bar=150 --param P_res_bar=200 --param GOR=50 --param PI=5 --ic blocked
    uv run mfsim run cases/paper01/gaslift/offshore_gl_P150.json --tf 8000 --out runs/paper01/gaslift/gl_P150.npz

## Benefício medido

- O caso do artigo 01 regenerado com o gerador alterado é idêntico (JSON fora de `meta.generator_params` e
  `.ic.npz` bit a bit).
- Estado inicial bloqueado, 323 estados (200 no artigo 01). Sem injeção (20 bar): pressão de fundo 97,0 bar
  (95,2 no artigo 01, a diferença é a célula de 30 m do tubing inferior), q_L 4,14 kg/s, q_G 0,123 kg/s, retenção
  fechada (vazão nula na linha de gás e no anular). Com injeção (150 bar): base do anular 170,8 bar, tubing na
  injeção 58,3 bar, injeção 1,15 kg/s, pressão de fundo 58,3 bar, q_L 5,71 kg/s (+38 %), q_G total na cabeça
  1,32 kg/s.
- **Sem injeção (20 bar), 8000 s, JAX, caso final (traçado da linha de produção, ramo quase seco):** reproduz o
  caso do artigo 01 — ponto a ponto nos dois primeiros ciclos e depois por assinatura de ciclo (a alternância de
  sopros com o choke aberto não é periódica, como entre numpy e JAX): nos últimos 4000 s P_in 17,6–30,8 bar
  (artigo 01: 17,5–30,8), fundo 94–108 bar (95–109), q_L do reservatório 3,94 kg/s (3,90), picos de líquido
  33,8 kg/s (34,3), intervalos entre sopros 1625/1315 s (1632/1284). A retenção fica fechada (vazão nula) e o
  anular a 22,6 bar. Custo igual ao do artigo 01: 9919 passos (35 % rejeitados), 366 s de CPU mais 38 s de
  compilação, contra ~10 400 passos por 8000 s na corrida de referência. Com o ramo a 2 % de líquido (versão
  anterior, `gl_P20_wet098.npz`) eram 40 828 passos e 1261 s.
- **Com injeção (150 bar), 8000 s, JAX:** o gas lift suprime a intermitência severa. Depois do primeiro sopro
  (80 kg/s aos 50 s, a coluna bloqueada do estado inicial a ser varrida pelo gás) o sistema assenta num
  estacionário: P_in 17,9 bar constante (a linha deixa de pressurizar), fundo do poço 69 bar (94–108 sem
  injeção), injeção 1,08 kg/s (≈ 110 000 Sm³/d) com o anular a 169 bar, produção de líquido 5,29 kg/s contra
  3,94 (+34 %), riser com 17–19 % de líquido em padrão intermitente. **Custo 5× o do caso sem injeção**: 51 625
  passos, 1879 s de CPU, 2,1 Jacobianas por passo. A causa não está no ramo de gás: só as duas células horizontais
  dos 10 m do topo do riser (antes do choke) trocam de padrão 75 vezes por 1000 s entre estratificado,
  intermitente e anular (α_G de 0,71 a 0,95 com o riser a 0,82), e a vazão de líquido no choke oscila entre 1,3 e
  8,4 kg/s com período de 37 s; o riser em si está parado. É chaveamento de fechamentos em células curtas
  horizontais com gás rápido, não física, e é o que há a resolver antes de usar o caso em lotes. Sondagens de
  400 s a partir do estado final (`probe_P150_*.npz`): recalcular a Jacobiana só à saída do Newton piora (4336
  passos contra 1893), o volume do tanque não muda nada (1885), e suavizar o chaveamento (`k_switch` 10, 5 e 2:
  49, 240 e 198 passos) tira o custo mas **muda a solução** — o topo fica estratificado, a vazão de líquido no choke
  sobe para 7,5–9,9 kg/s (acima da produção: o riser a drenar) e P_in afasta-se do estacionário —, logo não é um
  remédio, é outra resposta do modelo. A decisão fica para o dono (por exemplo excluir o estratificado no troço
  horizontal curto a jusante do riser, 10 m = 66 D, com issue própria).

## Referências

- Caso 10.3.2 da tese (gas lift intermitente; `notebooks_src/issues/nb_gas_lift_intermitencia.py`), issues 033
  (`slow_exit` nos eventos da válvula) e 039/040 (choke, estado inicial exato); `docs/data_formats.md` (vértices
  tanque, `check_valve`).
- Petrobras, especificação técnica I-ET-3A50.00-1350-940-P56-001 (poços satélite do Atapu): dois risers por poço
  satélite, produção e serviço/gas lift de 4" flexível.
- OTC 19274 (2008), Development of Subsea Facilities in the Roncador Field (P-52): sistema de coleta com linhas de
  produção emparelhadas com linhas de gas lift.
- Takács, G., 2005. Gas Lift Manual. PennWell (dimensionamento de válvulas e anular); API RP 11V6, Design of
  Continuous Flow Gas Lift Installations Using Injection Pressure Operated Valves.
- `papers/02-gas-lift-surrogate-optimization/PLAN.md` (decisões do dono de 2026-10-18).
