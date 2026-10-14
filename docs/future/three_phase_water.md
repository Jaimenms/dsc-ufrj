# Água como terceira fase (óleo, gás e água)

*Tema futuro; não iniciado. Contexto em [README.md](README.md).*

## Dois caminhos

**A. Três campos com três momentos.** Seis equações por célula, três pares de interface
(gás–óleo, gás–água, óleo–água), estratificado em três camadas (Taitel, Barnea e Brill),
dispersões com ponto de inversão e viscosidade de emulsão (Brauner), três combinações de fase
que desaparece, hiperbolicidade e regularizações refeitas por par. 6–8 sessões sem dados públicos
de severe slug trifásico para validar. Só se um caso concreto exigir a camada de água a mover-se
independentemente do óleo.

**B. Dois momentos e a água como campo dentro do líquido** (o que o OLGA faz na prática). Uma
equação de massa a mais, a água convectada com a velocidade do líquido mais um escorregamento
óleo–água algébrico (zero nas dispersões, correlação de deriva no estratificado); propriedades
do líquido como mistura (densidade ponderada, viscosidade de emulsão com inversão); fração de
água como onda de contacto, hiperbolicidade inalterada; nada de novo na fase que desaparece
(corte de água zero é óleo puro). Captura corte de água no holdup, densidade e viscosidade,
acumulação de água em pontos baixos pelo escorregamento, intermitência severa com água.
2–3 sessões, caso bifásico bit-idêntico com `qW = 0`.

## O que muda na infraestrutura (comum aos dois caminhos e ao composicional)

Hoje o layout de estados é global com quatro slots por nó, `[αG, PT, qL, qG]`, e atravessa
portas, especificações, `Results`, esparsidade, registo da tese e schema. Tornar a contagem de
slots dependente do modelo é a mudança invasiva, mecânica mas larga, que este tema partilha com o
composicional e com o criogénico; convém fazê-la uma vez, antes de qualquer um deles.

## Degraus (caminho B)

1. Slots por modelo; `qW`, `alphaW` em portas, especificações, tanques, reservatórios (IPR com
   corte de água), resultados e schema. 1 sessão.
2. Modelo `three_phase` com líquido mistura e escorregamento óleo–água; verificar `qW = 0`
   bit-idêntico e balanço de água exato. 1 sessão.
3. Casos: duto inclinado com acumulação de água em ponto baixo; severe slug de referência com
   corte de água 0, 30 e 70 %. 1 sessão.

## Dados de validação

Ensaios trifásicos de laboratório são raros e em geral proprietários (TUFFP, SINTEF); os nossos
casos de laboratório são ar–água, portanto "líquido" já é água. A validação possível é
qualitativa (inversão de fases, acumulação em pontos baixos) e por comparação com correlações
de holdup trifásico em regime permanente.
