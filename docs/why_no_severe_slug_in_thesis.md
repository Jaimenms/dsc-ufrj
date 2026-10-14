# Por que a tese não conseguiu simular a intermitência severa

*Documentação do mfsim (índice no [README](../README.md)). Síntese do que as entradas de
[issues/INDEX.md](issues/INDEX.md) mostram; cada ponto remete à issue com o antes, o depois e
os números.*

O modelo a dois fluidos a quatro equações da tese estava certo. O que impedia o ciclo de
intermitência severa de aparecer era a discretização, o tratamento das fases que desaparecem,
o fechamento de bolhas e o integrador. Qualquer um destes pontos sozinho já bastava para não
haver ciclo.

## 1. Esquema espacial

Os esquemas colocados da tese, Galerkin e diferenças centrais, não têm dissipação nas ondas de
fração de gás, e o "upwind" original era diferença regressiva em todas as variáveis, pressão
incluída, o que é instável para o modo acústico ([001](issues/001-reproducao-capitulo-10.md);
a corrida `10_1_1_fd_upwind` do lote de issues para em 680 passos). Um ciclo de severe slug
exige propagar frentes nítidas de líquido e gás e deixar fases aparecerem e desaparecerem.
Isso só ficou possível com o esquema de volumes finitos escalonados com upwind pela velocidade
de cada fase ([002](issues/002-fv-upwind.md)).

## 2. Fase que desaparece e balanço de massa

Numa célula cheia de líquido a vazão de gás fica indeterminada. A tese não tinha tratamento
para isso, e qualquer limitador largo de fração destrói massa e produz ciclos falsos
([003](issues/003-fase-que-desaparece.md)). Sem balanço exato, com a formulação conservativa e
o acumulador consistente com o BDF, não se distingue um ciclo físico de um artefacto
([004](issues/004-formulacao-conservativa.md), [005](issues/005-acumulador-bdf.md)): a forma
primitiva perde 2 a 6 % do líquido alimentado, um degrau por blowdown.

## 3. Fechamentos

- O fechamento de bolhas da tese tem a velocidade terminal sem gravidade e sem tensão
  superficial e uma força interfacial dimensionalmente inconsistente; à escala de laboratório
  nenhum caso cicla com ele, e com o arrasto físico Fabre 2 dá 135.8 s contra 137 medidos
  ([015](issues/015-fechamento-de-bolhas.md), [030](issues/030-validacao-laboratorio.md)).
- A identificação automática de padrão tinha quatro defeitos que a tornavam inutilizável em
  dinâmica ([020](issues/020-ipe-da-tese-defeitos.md), [021](issues/021-ipe-suave-e-local.md)).
- Com os fechamentos prescritos faltavam as regras de fração baixa e alta e a massa virtual
  para o modelo ser hiperbólico e para o filme do riser drenar depois do blowdown
  ([016](issues/016-fracao-alta-anular.md), [017](issues/017-fracao-baixa-bolhas-bestion.md),
  [018](issues/018-massa-virtual.md)); e o joelho anguloso com chave dura de fechamento em 45°
  amortecia o ciclo ([008](issues/008-joelho-em-arco.md)).

## 4. Integrador e custo

O ciclo pede dezenas de milhares de passos de milissegundos no blowdown, com Newton a
convergir devagar por não-suavidade dos fechamentos. O `ode15s` fechado não permitia projeção
em limites físicos com recuo de ordem, tolerâncias por faixa de variável, orçamento de Newton
nem lado direito compilado com Jacobiana exata; uma corrida de 3000 s custava de dezenas de
minutos a horas, contra 40 s hoje ([032](issues/032-bdf-implicito-proprio.md),
[034](issues/034-tolerancia-por-faixa.md), [035](issues/035-config-rapida.md),
[036](issues/036-lado-direito-jax.md)).

## O que foi preciso, em uma frase

Esquema escalonado conservativo com upwind por fase e fase ausente relaxada pela célula
doadora, arrasto físico de bolhas com as regras de fração e a massa virtual, joelho em arco, e
um BDF próprio com tolerância por faixa e lado direito compilado: com isso o caso de campo cicla
em 40 s de CPU e os casos de laboratório reproduzem o período medido
([severe_slug.md](severe_slug.md), [lab_validation.md](lab_validation.md)).
