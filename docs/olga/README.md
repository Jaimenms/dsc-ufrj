# Casos base do artigo 01 no OLGA: tabelas de propriedades

Documento de apoio para reproduzir no OLGA os casos do sistema offshore do artigo 01 (poço, linha de 2 km, riser de
317 m, choke no topo) a partir do fluido do mfsim. O mfsim não tem equação de energia nem transferência de massa: o
fluido é um par líquido–gás politrópico ajustado a um flash de Peng–Robinson a 40 °C, com a divisão de massa entre as
fases congelada. O OLGA, pelo contrário, só aceita fluidos descritos por tabelas de propriedades em pressão e
temperatura (ou por modelos black-oil e composicionais). A abordagem é escrever essa tabela diretamente a partir das
leis do mfsim, de modo que o OLGA veja o mesmo fluido, e desligar no OLGA o que o mfsim não modela.

Ficheiros desta pasta, gerados por `tools/olga_tables.py` (não editar à mão):

| ficheiro | caso do mfsim | RGO (Sm³/Sm³) | fração mássica de gás w_G |
|---|---|---|---|
| `offshore_gor50.csv` | `cases/paper01/offshore.json` (caso base) | 50 | 0,0288 |
| `offshore_gor30.csv` | `cases/paper01/offshore_gor30.json` | 30 | 0,0074 |
| `offshore_gor80.csv` | `cases/paper01/offshore_gor80.json` | 80 | 0,0592 |
| `offshore_gor122.13.csv` | `cases/paper01/offshore_gor122.13.json` | 122,13 | 0,0988 |

As quatro tabelas só diferem na coluna RS (e no cabeçalho de RGO): os quatro casos usam o mesmo óleo e o mesmo gás e a
RGO só muda a fração mássica de gás livre que entra do reservatório. É o lote `batches/paper01_offshore_gor.json`, a
comparação da secção da razão gás–óleo do artigo.

## 1. O fluido do mfsim

Parâmetros do bloco `fluid` dos casos (Tabela 6 da dissertação de Gustavo Alves de Carvalho, UERJ 2022, flash de
Peng–Robinson do cmfsim a 40 °C; densidades e viscosidades ajustadas a 20 bar):

| grandeza | símbolo | valor |
|---|---|---|
| densidade do líquido a P0 | ρ_L0 | 646,7 kg/m³ |
| densidade do gás a P0 | ρ_G0 | 15,0 kg/m³ |
| pressão de referência | P0 | 20 bar |
| expoente politrópico do líquido | γ_L | 1000 (quase incompressível) |
| expoente politrópico do gás | γ_G | 1,0 (gás ideal isotérmico) |
| viscosidade do líquido | μ_L | 3,14 × 10⁻⁴ Pa·s |
| viscosidade do gás | μ_G | 1,21 × 10⁻⁵ Pa·s |
| tensão interfacial | σ | 0,0163 N/m |

Leis: ρ_N(P) = ρ_N0 (P/P0)^(1/γ_N) e dρ_N/dP = ρ_N/(γ_N P), para N = L, G. Não há dependência da temperatura.

Divisão de massa (`meta.pvt` e `meta.mass` dos casos): nas condições padrão da ANP (1,01325 bar, 20 °C) o óleo de
tanque tem 673,25 kg/m³, o gás dissolvido 1,1547 kg/m³ e o gás livre 0,7763 kg/m³; a 20 bar e 40 °C o óleo satura com
R_s = 23,24 Sm³/Sm³ e o resto da RGO é gás livre. A fração mássica de gás livre é

    w_G = ρ_gf,std (RGO − R_s) / [ρ_o,std + ρ_gd,std R_s + ρ_gf,std (RGO − R_s)]

e fica congelada ao longo de toda a rede: não há transferência de massa entre as fases com a pressão.

## 2. O que o OLGA pede

O OLGA lê as propriedades de um ficheiro `.tab` (`FILES PVTFILE = ...` no caso, `FLUID` com o rótulo da tabela). Há
dois formatos: o fixo (antigo, blocos `TABTEX` por propriedade) e o por palavras-chave (recomendado), com um bloco
`PVTTABLE LABEL = ..., PHASE = TWO, ...` de metadados, as listas `PRESSURE = (...) BARA` e `TEMPERATURE = (...) C`, a
linha `COLUMNS = (...)` e uma linha `PVTTABLE POINT = (P, T, ...)` por nó da malha. As colunas de uma tabela bifásica,
com as unidades do manual:

| coluna | unidade | significado |
|---|---|---|
| PT | Pa | pressão |
| TM | °C | temperatura |
| ROG, ROHL | kg/m³ | densidades do gás e do óleo |
| DROGDP, DROHLDP | s²/m² | derivadas das densidades em pressão |
| DROGDT, DROHLDT | kg/m³/°C | derivadas das densidades em temperatura |
| RS | – | fração mássica de gás na mistura gás + óleo |
| VISG, VISHL | N·s/m² | viscosidades |
| CPG, CPHL | J/kg/°C | capacidades térmicas |
| HG, HHL | J/kg | entalpias |
| TCG, TCHL | W/m/°C | condutividades térmicas |
| SIGGHL | N/m | tensão interfacial gás–óleo |
| SEG, SEHL | J/kg/°C | entropias |

Regras relevantes para nós:

- a malha é sempre bidimensional, com 2 a 50 pontos em cada eixo; não existe tabela só em pressão;
- todas as propriedades têm de existir em todos os nós, para as duas fases, mesmo onde uma delas não existiria;
- as condições padrão (`STDPRESSURE`, `STDTEMPERATURE`) têm de cair dentro da malha, e o cabeçalho leva `GOR`,
  `GLR`, `STDGASDENSITY` e `STDOILDENSITY`, usados quando uma fonte é dada em volumes padrão;
- as curvas de bolha e de orvalho são opcionais para fluidos de mais de um componente;
- o RS da tabela é o que faz a transferência de massa no OLGA: a fração de gás local segue sempre o RS(P, T).

## 3. Mapeamento coluna a coluna

| coluna | valor na tabela | de onde vem |
|---|---|---|
| PT | malha de 27 pressões, 1,01325 a 300 bar | inclui a pressão padrão; densa abaixo de 30 bar; cobre a pressão do reservatório (200 bar) e folga |
| TM | 20 e 60 °C, propriedades iguais | duas temperaturas são o mínimo do OLGA; enquadram a padrão (20 °C) |
| ROG | 15 (P / 20 bar) | lei politrópica com γ_G = 1; a 1,01325 bar dá 0,760 kg/m³, o gás livre padrão do PVT (0,776) a menos de 2 % |
| ROHL | 646,7 (P / 20 bar)^(1/1000) | 644,8 a 1 atm, 648,5 a 300 bar |
| DROGDP, DROHLDP | ρ/(γ P) | derivadas exatas; o inverso é a velocidade do som ao quadrado |
| DROGDT, DROHLDT | 0 | sem dependência da temperatura |
| RS | w_G do caso, igual em todos os nós | sem transferência de massa, como no mfsim |
| VISG, VISHL | 1,21 × 10⁻⁵ e 3,14 × 10⁻⁴ | constantes |
| SIGGHL | 0,0163 | constante |
| CPG, CPHL | 2200 e 2000 J/kg/°C | valores típicos de gás natural leve e óleo leve; o mfsim não os tem |
| HG, HHL | c_p T (referência 0 °C) | coerentes com as capacidades |
| TCG, TCHL | 0,03 e 0,13 W/m/°C | valores típicos |
| SEG, SEHL | c_p ln(T/273,15 K), com −(R/M) ln(P/P_std) no gás | gás ideal de massa molar 19,5 g/mol (a de ρ_G0 a 20 bar e 40 °C) |

Cabeçalho da tabela: `PHASE = TWO`, `GOR = GLR = RGO` do caso, `STDOILDENSITY = 673,25 kg/m³`, `STDGASDENSITY =
0,760 kg/m³`, `STDPRESSURE = 1 atm`, `STDTEMPERATURE = 20 °C`.

## 4. Como configurar o OLGA

- **Temperatura desligada:** `OPTIONS TEMPERATURE = OFF`, para que o OLGA use as temperaturas iniciais e não resolva a
  energia; é o único modo coerente com um fluido sem dependência térmica, e torna irrelevantes as oito colunas
  térmicas preenchidas com valores típicos.
- **Uma tabela por RGO:** como a fração de gás local segue o RS da tabela, variar a RGO exige trocar a tabela
  (`FLUID = <rótulo>`), não a fração de gás da fonte.
- **Fontes em massa:** o poço do mfsim é uma IPR linear em massa (PI em kg/s/bar dividido pelas fases com w_G); no
  OLGA, descrever o reservatório com `WELL` e a IPR em massa ou com uma `SOURCE` de caudal mássico e fração de gás
  igual ao RS da tabela, para não haver ajuste de composição à entrada.
- **Condições padrão e volumes:** se o OLGA pedir caudais em volumes padrão, usar as densidades do cabeçalho; a RGO do
  cabeçalho é a da tabela.

## 5. Diferenças que vão aparecer na comparação

- O gás é ideal e isotérmico (γ_G = 1): a 200 bar a tabela dá 150 kg/m³, sem fator de compressibilidade; o OLGA
  interpola linearmente entre os nós, e a lei é linear em P, pelo que a malha não introduz erro.
- Não há transferência de massa: um OLGA com um PVT real libertaria gás ao subir no riser; com esta tabela não. Essa é
  a diferença a explicar quando se compararem os ciclos.
- Viscosidades e tensão interfacial constantes; o OLGA usa-as nas suas próprias correlações de atrito e de padrão de
  escoamento, que não são as do mfsim.
- O líquido quase incompressível (γ_L = 1000) tem nesta lei uma velocidade do som que cresce com a raiz da pressão
  (400 m/s a 1 atm, 1760 m/s a 20 bar, 6800 m/s a 300 bar), não a de um óleo real; não deve afetar a golfada severa,
  que é lenta.

## 6. Como regenerar e próximos passos

    uv run python tools/olga_tables.py cases/paper01/offshore.json --out docs/olga/offshore_gor50.csv
    uv run python tools/olga_tables.py cases/paper01/offshore_gor30.json --out docs/olga/offshore_gor30.csv

`--gor` recalcula a fração de gás para outra RGO com o PVT do caso, `--P-max` encurta a malha e `--T` muda o par de
temperaturas (tem de enquadrar 20 °C). O CSV tem duas linhas de cabeçalho comentadas (metadados da tabela em JSON e
unidades) e depois as colunas na ordem do OLGA.

Próximos passos: escrever o `.tab` no formato por palavras-chave a partir do mesmo gerador (é a mesma tabela com o
cabeçalho `PVTTABLE`), montar a geometria do caso base no OLGA e comparar o período e a amplitude do ciclo com o lote
`runs/paper01/offshore_gor/`.

Fontes consultadas: manual do OLGA (secções "Keyword based format for fluid properties" e "Fixed/standard format for
fluid properties"), guia "Getting Done With OLGA Input" (formatos Key e Fixed, malha P–T), gerador de tabelas OLGA do
neqsim (`OLGApropertyTableGeneratorKeywordFormat`) e leitor pyfas de ficheiros `.tab`.
