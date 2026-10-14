# Composição por fase e transferência de massa (pré-sal, gás-condensado, severe slug com flash)

*Tema futuro; não iniciado. Contexto em [README.md](README.md).*

## O problema

O modelo tem fases fixas: o gás é gás, o líquido é líquido, e só a pressão os comprime. Na
produção real o gás sai de solução quando a pressão cai (base do riser, blowdown), o condensado
condensa na linha (retrógrado), e no pré-sal o CO₂ rico no gás muda densidades, solubilidades e
o Joule–Thomson. A intermitência severa com flash muda período e picos. Os simuladores comerciais
tratam isto com **equilíbrio local por tabela PVT** (o flash fecha a célula), hipótese que não se
sustenta com escorregamento: o gás e o líquido de uma célula vieram de montantes diferentes.
Decisão tomada em 2026-10-14: **composição por fase e Γ explícito**, não flash de célula fechada.

## Como entra no modelo

- **Estados**: massas por componente **em cada fase**, ρG αG yᵢ e ρL αL xᵢ (2·Nc por célula; com
  dois pseudocomponentes, quatro em vez de duas), cada uma convectada pela velocidade da sua fase
  com a célula doadora e o limitador que já existem; αG e P continuam a sair do mapa inverso
  explícito (densidades de fase ρG(P, y), ρL(P, x)), sem raiz de flash e sem a descontinuidade do
  ponto de bolha.
- **Balanços**: massas com ±Γᵢ; momentos com Γ·v_interface; propriedades de fase dependentes da
  composição (tabelas ou EOS simples por fase).
- **Γᵢ = kᵢ·a·(c_interface − c_seio)** por fase, com a composição da interface do flash a (P, T)
  (a termodinâmica entra só na interface) e k·a por padrão de escoamento: estratificado (área
  da interface com ondas), bolhas (6αG/d_b com o diâmetro de Hinze que já temos), anular (filme e
  gotas), intermitente (composição das três). É a parte empírica e a que mais custará calibrar.
- **Fronteiras e elementos**: composição nas especificações de entrada, nos tanques e no
  separador; IPR por componente.
- **Energia**: segundo degrau (calor latente como fonte na energia quando ela existir).

## Degraus

1. Infraestrutura: contagem de slots por modelo, `qG`/`qL` por componente nos resultados,
   schema (`fluid.type = "two_component"`), acumulador por componente. 1 sessão.
2. Dois pseudocomponentes, isotérmico, Γ com k·a constante por padrão; caso de duto único com
   queda de pressão e gás a sair de solução; verificar balanço por componente e o limite k·a → ∞
   (equilíbrio) contra uma tabela PVT. 2 sessões.
3. k·a por padrão e calibração; severe slug de referência com flash: efeito no período e nos
   picos; lotes e notebook de issues. 1–2 sessões.
4. Energia (partilhada com o CO₂ e o criogénico); CO₂ no gás do pré-sal como terceiro
   pseudocomponente. 2 sessões.

## Dados de validação

Escassos em público: despressurização de tubos com fluidos vivos; a tese da UFRN com modelo a
dois fluidos composicional transiente para poços (Peng–Robinson, totalmente implícito, volumes
finitos escalonados) como referência de formulação; o *benchmark* de simuladores no pré-sal
(JBSMSE 2026, 350 medições de sete unidades, paradas e rearranques) como contexto; dados de
campo e de laboratório da Petrobras, se acessíveis. Sem dados de riser com flash, a validação é
indireta: limite de equilíbrio contra tabela PVT e sensibilidade a k·a.

## Riscos

k·a com incerteza de ordem de grandeza; rigidez quando k·a é grande (tende ao equilíbrio);
propriedades de fase por composição; e o custo por avaliação a dobrar com os estados.
