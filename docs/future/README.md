# Temas futuros

*Planos para levar o mfsim além da tese, escritos em 2026-10-14 a partir das discussões e da
pesquisa de literatura dessa data. Nenhum foi iniciado. Cada plano diz o que o modelo tem, o que
falta, por que degraus avançar, com que dados validar e onde está o risco.*

| plano | o que é | degrau maior | esforço estimado |
|---|---|---|---|
| [cryogenic_chilldown.md](cryogenic_chilldown.md) | *chill-down* de linhas de hidrogénio líquido e criogenia: ebulição em película, frente de molhagem, parede conjugada | equação de energia e parede como campo | 8–12 sessões |
| [co2_depressurization.md](co2_depressurization.md) | despressurização de dutos de CO₂ para CCS: flash fora do equilíbrio, onda de descompressão, fratura dúctil | transferência de massa explícita (Γ) num fluido puro | 5–8 sessões |
| [compositional_mass_transfer.md](compositional_mass_transfer.md) | composição por fase e transferência de massa (pré-sal, gás-condensado, intermitência severa com flash) | massas por componente por fase e Γ por padrão | 4–6 sessões + energia |
| [three_phase_water.md](three_phase_water.md) | água como terceira fase (óleo, gás, água) | contagem de slots por modelo; escorregamento óleo–água | 2–3 sessões (caminho B) |
| [slug_signatures_routing.md](slug_signatures_routing.md) | catálogo de assinaturas de intermitência severa por abertura do choke; alocação de poços a separadores e faseamento das golfadas; contexto da linha de controle anti-slug e gêmeo digital | vértice separador com controle de nível e choke como entrada de controle (itens 6 e 9 do README) | 7–9 sessões |

Ordem sugerida: o CO₂ primeiro (fluido puro, dados públicos, o Γ explícito que os outros dois
reutilizam), depois o composicional, depois o criogénico, que precisa de tudo o que os anteriores
constroem mais a energia. O trifásico é independente e pode entrar a qualquer momento.
O catálogo de assinaturas e a alocação a separadores não dependem de nenhum dos anteriores: só
do vértice separador e do choke controlado, e podem começar já.

Regras que se mantêm: tudo como opção com bandeira, caso bifásico bit-idêntico quando a opção está
desligada, uma issue por alteração em `docs/issues/`, corridas por lotes, notebooks só a ler.
