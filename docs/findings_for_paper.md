# Achados para o artigo

*Documentação do mfsim (índice no [README](../README.md)).*

## Achados para o artigo (resumo)

*Detalhe em [issues/004](issues/004-formulacao-conservativa.md), [008](issues/008-joelho-em-arco.md) e [003](issues/003-fase-que-desaparece.md).*

1. **Conservação de massa vs variáveis de estado.** A integração temporal das
   equações de massa na forma primitiva (dαG/dt, dP/dt) perde massa em degraus,
   um por *blowdown* (2–6 % do líquido alimentado; 8–12 % do gás), mesmo com
   balanços exatos por célula no espaço. A formulação com resíduo conservativo
   (BDF aplicado a ρG·αG e ρL·αL, Newton nas primitivas, limitador de fluxo da
   célula doadora) zera o erro a menos de arredondamento, com a mesma dinâmica.
   O mapa (αG, P) → massas é singular em αG = −γG/γL; a pressão passa a ser
   escrava da massa de líquido (ruído de pressão ∝ γL), o que custa passos. A
   verificação exige integrar os fluxos de contorno com a mesma recorrência BDF
   dos estados (a regra dos trapézios nas saídas não fecha).
2. **Sensibilidade ao tratamento do joelho.** Com joelho anguloso e chave dura de
   fechamento em 45°, o ciclo de intermitência severa é amortecido (pressão da
   linha 5.7–6.4 bar, blowdowns a cada ~240 s). Com joelho em arco e mistura suave
   dos fechamentos por ângulo aparece o *blowdown* completo "de livro" (4.2–6.1 bar,
   blowdowns a cada ~660–700 s, picos menores e mais largos), em ambas as formulações.
   Misturar fechamentos sem o arco (só na face do joelho) piora a convergência.
   Malha graduada no pescoço reproduz a malha fina pela metade do custo; refinar
   além de ~1.5 m na base do riser dispara *chattering* ligado à má-posição dos
   fechamentos em fração de gás muito baixa (ver "Hiperbolicidade").
3. **Ingredientes numéricos indispensáveis** para a fase rápida: coeficientes de
   massa com a fração real, relaxação da fase ausente pela célula doadora com
   chaveamento abrupto, chaveamento *upwind* adaptativo, recuo BDF2 → BDF1 com
   projeção, regularização do atrito em v = 0, compressibilidade artificial do
   líquido. Limitadores "generosos" de fração de gás produzem ciclos falsos.
