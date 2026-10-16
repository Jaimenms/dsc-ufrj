# 039 · Lei de válvula homogénea (o choke resiste ao gás)

| | |
|---|---|
| tipo | bugfix |
| área | rede |
| estado | feito (2026-10-16); padrão do mfsim; os casos da tese usam `law = "thesis"` |
| onde | `mfsim.network.elements.valve_upstream_pressure`; `Choke.law`, `Choke.q_smooth`; chave `law` das arestas `choke` no JSON |
| dados | `runs/maps/prototype/` (fora do git): mapa abertura × pressão do reservatório com as duas leis, 126 corridas cada |

## Antes

A válvula da tese (`modelo_valvula.m`, portada bit a bit) calcula a queda com uma densidade
da mistura **ponderada pela vazão mássica**, ρM = (qL·ρL + qG·ρG)/(qL + qG), e
Δp = qM·|qM|/ρM/(0,0075·CV·ab)². Como o gás quase não tem massa, ρM fica perto da densidade
do líquido mesmo com o gás a dominar o volume: o choke quase não resiste ao gás. Dois efeitos:

- **Física.** O choke trava os picos de líquido mas deixa passar barata a mistura com gás.
  No mapa da intermitência severa (caso de referência com IPR e choke no topo) o choke
  tira o pico da vazão de saída mas não para o ciclo de pressão até 3 % de abertura, o que
  contraria os ensaios (Schmidt 1980; Jansen, Shoham e Taitel 1996).
- **Numérica.** A densidade tem qL + qG no denominador. No fim do blowdown a vazões baixas o
  gás reflui no choke e a soma passa por zero: ρM chegou a valores entre −46 000 e
  +7 000 kg/m³, a queda passa a ter o sinal errado e deixa de ser monótona, o Newton não
  converge e o passo colapsa. 7 das 126 corridas do mapa falharam assim.

## Depois

Lei **homogénea** (padrão): a densidade sem escorregamento, ponderada por volume,
ρ = qM / (|qL|/ρL + |qG|/ρG), o que dá Δp = qM·Q/(0,0075·CV·ab)² com
Q = |qL|/ρL + |qG|/ρG. Não há divisão pela vazão total, Δp tem sempre o sinal de qM e é
monótona em escoamento co-corrente. Os módulos são suaves, |x| ≈ x·tanh(x/q_s) com
`q_smooth` = 10⁻⁶ m³/s, exatos a menos de exp(−|x|/q_s). As densidades continuam a ser as da
pressão de jusante, como na tese; não há escoamento crítico.

- Em líquido puro as duas leis coincidem (teste a 10⁻¹²).
- A lei da tese continua disponível com `"law": "thesis"` na aresta `choke`. Os casos do
  capítulo 10 (`cases/chapter10/10_3_*.json`) e os grafos lidos dos `.mf2` usam-na, logo os
  goldens de nível A não mudam.
- JSON novo sem `law` significa lei homogénea. As corridas gravadas antes desta alteração não
  têm a chave e foram simuladas com a lei da tese: `load_run` fixa-lhes `law = "thesis"`.

## Benefício

A 4 bar, em função da fração mássica de gás x no choke:

| x | densidade, tese | densidade homogénea | queda homogénea ÷ queda da tese |
|---|---|---|---|
| 0,002 (alimentação do caso de referência) | 898 kg/m³ | 644 kg/m³ | 1,4 |
| 0,01 | 891 kg/m³ | 301 kg/m³ | 3,0 |
| 0,1 | 810 kg/m³ | 43 kg/m³ | 18,8 |

A sensibilidade relativa da queda à fração de gás é ~1 na lei da tese e 140 na homogénea
(x = 0,002): duas ordens de grandeza, e a diferença é maior quando o gás chega ao topo.

No mapa abertura × pressão do reservatório (126 corridas por lei, 8000 s, partida
bloqueada):

| | lei da tese | lei homogénea |
|---|---|---|
| falhas do integrador | 7 | 0 |
| células estáveis | 11 (9 + 2 de deriva lenta) | 70 (57 + 13 de deriva lenta; duas confirmadas estáveis a 40 000 s) |
| fronteira do lado aberto | não existe abaixo de 30 bar | queda no choke 0,5–0,8 bar em todas as linhas; o critério de Jansen et al. (1996) dá 0,6–0,7 bar |

Testes: `tests/test_valve_law.py` (líquido puro, refluxo, sensibilidade ao gás, JSON,
registo legado, corridas antigas, paridade numpy–JAX).

## Referências

Notebook: [nb_estabilidade_mapa](../../notebooks/issues/nb_estabilidade_mapa.ipynb) (fonte em `notebooks_src/issues/`).

Jansen, Shoham e Taitel (1996), *The elimination of severe slugging — experiments and
modeling*, IJMF 22 — critério de estabilidade com choke (eqs. 5 e 10). Schmidt, Brill e
Beggs (1980), *Choking can eliminate severe pipeline-induced slugging*. A tese,
`modelo_valvula.m` (histórico até ao commit 22ecbc9).
