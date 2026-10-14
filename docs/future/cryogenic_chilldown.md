# Chill-down de linhas criogénicas (hidrogénio líquido)

*Tema futuro; não iniciado. Contexto em [README.md](README.md).*

## O problema

Arrefecer uma linha de transferência de ~300 K até 20 K (LH₂; LN₂ 77 K como fluido de ensaio)
ao introduzir o líquido criogénico. Três fases: ebulição em película (parede acima da temperatura
de molhagem, vapor isola o líquido, anular invertido com gotas à frente, coeficiente de calor
baixo, quase todo o líquido sai como vapor), frente de molhagem que avança a uma velocidade fixada
pela massa térmica da parede e pelo caudal (transição para ebulição nucleada, transferência de
calor sobe uma a duas ordens de grandeza), e líquido monofásico. Instabilidades: golpes de pressão
por colapso de bolsões de vapor, pulsações e *chugging* a baixo caudal. Perguntas de engenharia:
tempo e *boil-off* do chill-down, cargas térmicas e de pressão na linha, estratégia (caudal
contínuo baixo, pulsado, pré-arrefecimento com LN₂, contrapressão).

Por que é promissor: o abastecimento de LH₂ a aviões, camiões e navios repete o transiente em cada
operação; a literatura usa quase só modelos homogéneos 0D/1D ajustados; há dados públicos de
qualidade (NASA/NIST/KSC com LN₂ e LH₂); o fluido é puro, sem composição.

## O que o mfsim tem e o que falta

| tem | falta |
|---|---|
| esquema escalonado conservativo com fase que aparece e desaparece (a frente de líquido a entrar num tubo de vapor é esse problema) | **equação de energia** por fase (gás e líquido) com entalpia e trabalho de pressão |
| BDF implícito próprio para transientes de milissegundos; tolerâncias por faixa; JAX | **parede como campo**: condução radial e axial, capacidade térmica dependente de T (conjugado) |
| análise acústica e viscosidade artificial (golpes de pressão) | **transferência de massa Γ** comandada pelo calor da parede e da interface, não pela pressão |
| fechamentos por padrão com misturas suaves | **ebulição por regime**: película, transição, nucleada; temperatura de molhagem; anular invertido |
| fluido politrópico | **propriedades do parahidrogénio** perto do ponto crítico (tabelas NIST/REFPROP), vapor sobreaquecido |
| lotes, notebooks, issues, goldens | dados de chill-down e métricas (tempo até molhagem, perfil de temperatura da parede, pressão) |

## Como entra no modelo

- **Estados por célula**: αG, P e duas energias (ou temperaturas) TG, TL; por face qG, qL; mais a
  temperatura da parede Tw por célula (e, se preciso, por camada radial). Contagem de slots passa a
  depender do modelo (a mesma mudança de infraestrutura do trifásico e do composicional).
- **Balanços**: massas com ±Γ; momentos com Γ·v_interface; energias com calor parede→fase,
  calor interfacial, calor latente em Γ; parede com condução e o calor trocado com cada fase.
- **Γ**: da interface, Γ = (q_wall,L + q_interface)/h_lv quando a parede molha, e da evaporação
  na película quando não molha; vapor fora do equilíbrio térmico (TG > Tsat) como estado próprio,
  o que é o que os ensaios da NASA medem.
- **Fechamentos**: coeficientes de ebulição por regime (Darr & Hartwig para LH₂; Bromley para
  película; Chen para nucleada), critério de molhagem (Leidenfrost / temperatura mínima de
  ebulição em película), atrito e interface do anular invertido.
- **Propriedades**: `fluid.type = "table"` com tabelas em (P, T) ou (P, h) para parahidrogénio
  e azoto, com derivadas, geradas uma vez a partir do NIST.

## Degraus

1. Energia isotérmica-adiabática: equação de energia por fase sem parede, validada em
   despressurização adiabática simples (reutiliza o degrau 1 do CO₂). 1–2 sessões.
2. Parede conjugada 1D (radial agregado) com propriedades de T; teste com tubo cheio de vapor a
   arrefecer. 1–2 sessões.
3. Γ por ebulição com os três regimes e a frente de molhagem; caso LN₂ horizontal do NIST
   (Brentari / Darr) como primeira validação, porque é o mais documentado. 3–4 sessões.
4. LH₂: propriedades do parahidrogénio, correlações específicas de Darr & Hartwig, ensaios de
   chill-down de linha de LH₂ da NASA. 2–3 sessões.
5. Estratégias: contínuo vs pulsado vs pré-arrefecimento; métricas de *boil-off* e tempo. 1 sessão.

## Dados de validação

Darr & Hartwig (NASA, correlações e ensaios de chill-down de linha de LH₂; vapor fora do
equilíbrio em ebulição em película); NASA NTRS: visualização de chill-down de LH₂, modelos
GFSSP de transferência tanque a tanque (contínuo e pulsado); NIST: linha horizontal de LN₂;
capítulo CAMPE 2025 (Springer 2026) sobre chill-down de sistemas de hidrogénio.

## Riscos

Dois novos campos de estado (energias) e um terceiro (parede) no mesmo passo; não-suavidade do
critério de molhagem (chave dura película → nucleada), que pede rampa C1 como as regras de fração;
velocidade do som bifásica muito baixa (passos pequenos nos golpes); propriedades perto do ponto
crítico do H₂ (33 K, 13 bar) se a pressão de operação for alta. Ortho-para: desprezável na escala
de tempo do chill-down.
