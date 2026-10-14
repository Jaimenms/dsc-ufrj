# Despressurização de dutos de CO₂ (CCS)

*Tema futuro; não iniciado. Contexto em [README.md](README.md).*

## O problema

Num duto de CO₂ em fase densa (impurezas: N₂, CH₄, H₂, O₂, He), uma ruptura ou despressurização
controlada gera uma onda de descompressão cuja velocidade decide se uma fratura dúctil se propaga
(critério de Battelle das duas curvas). A onda tem um patamar quando entra na região bifásica; a
posição do patamar depende de **quando** o líquido flasheia. Os ensaios (SINTEF/NTNU, DNV) mostram
atraso de nucleação com *undershoot* de pressão e recuperação, e escorregamento entre fases; os
modelos homogéneos de equilíbrio (HEM) erram o patamar e a temperatura. A literatura de 2023–2025
(Log, Hammer, Munkejord; Uilhoorn et al.) está a passar para modelos a dois fluidos com
transferência de massa fora do equilíbrio e diz que o fechamento de Γ é a peça em aberto.
Conferências: TCCS (sessões "CO₂ transport – pipelines"), GHGT, Pipeline Technology Conference.

## Por que é o primeiro dos temas futuros

Fluido puro ou quase (sem composição por fase na primeira versão), isotérmico aceitável como
primeiro degrau, dados públicos de tubo de choque e de tubos longos, e o Γ explícito construído
aqui serve ao composicional e ao criogénico.

## O que entra no modelo

- **Γ explícito** nos balanços de massa (±Γ) e de momento (Γ·v_interface): taxa de nucleação e
  crescimento de bolhas em função do sobreaquecimento P_sat(T) − P (Log et al. 2023), com
  nucleação homogénea e heterogénea; alternativa mais simples para começar: relaxação com tempo
  característico (HRM), Γ = (α_eq − α)/τ.
- **Fluido real**: `fluid.type = "table"` com Span–Wagner (ou GERG para misturas) em (P, T) ou
  (P, ρ), incluindo a região metaestável do líquido (extrapolação da EOS abaixo de P_sat).
- **Energia**: o primeiro degrau pode ser isotérmico-adiabático por fase com tabelas; a
  temperatura durante a despressurização é um dos resultados medidos, por isso a energia é o
  segundo degrau.
- **Condições de contorno**: saída crítica (choked flow) na ruptura ou no orifício: fluxo
  sónico bifásico como condição de fronteira, que não temos.
- **Impurezas**: como pseudocomponente único com K-value, ou pela tabela da mistura.

## Degraus

1. HRM isotérmico com Span–Wagner tabelado; tubo de choque de CO₂ puro (SINTEF) e comparação do
   patamar com HEM. 2–3 sessões.
2. Γ por nucleação (homogénea + heterogénea) e vapor sobreaquecido; os ensaios a diferentes
   temperaturas iniciais de Log et al. 2–3 sessões.
3. Energia e temperatura; ensaios de temperatura em despressurização rápida. 2 sessões.
4. Impurezas (CO₂–N₂, CO₂–He) e a curva de descompressão para o critério de fratura. 1–2 sessões.

## Dados de validação

Log et al., IJMF 2023 (efeito do estado inicial, dados de alta resolução de P e T); o modelo de
flashing com nucleação e crescimento de bolhas (IJMF); a tese de Log (NTNU 2023); SINTEF
"CO₂ dynamics" (despressurização de misturas, CO₂–N₂, CO₂–He); Uilhoorn et al., IJGGC 2025
(choque hidráulico, cinco ensaios de três instalações, HEM contra dois fluidos); programa da
TCCS-13 (2025) com as comunicações de Equinor/TotalEnergies (LedaFlow), SINTEF/NTNU, Varsóvia,
LSU e IFE ("terrain slugging" em duto bifásico de CO₂, que é o nosso severe slug com mudança
de fase).

## Riscos

Rigidez de Γ quando a nucleação dispara; condição de fronteira sónica; propriedades na região
metaestável; a onda de descompressão exige malha fina e passos pequenos só nos primeiros
milissegundos (o BDF com ordem variável e a viscosidade artificial ajudam, mas é um regime que
nunca corremos).
