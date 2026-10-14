# 013 · Filtro anti-refluxo do código original removido

| | |
|---|---|
| tipo | bugfix |
| área | rede / esquema numérico |
| estado | feito (2026-10-13) |
| onde | campo `check` dos `.mf2` continua a ser lido, sem efeito; `NetworkSystem` sem ramos de corte |
| dados | caso 10.2.2 reexecutado contra o MATLAB |

## Antes

O código original tinha um filtro nas saídas dos dutos marcados com `check`: forçava as
vazões de saída acima de 1e−6 kg/s e anulava derivadas negativas. Era um remendo para
contornar a falta de informação de fração nas extremidades em refluxo (a dificuldade
relatada na tese), e um ramo em valores que não compila nem diferencia.

## Depois

Removido do modelo (numpy e JAX). O refluxo é resolvido pelas equações e pelo tratamento
das extremidades com fração imposta (014).

## Benefício

O caso 10.2.2, que usava o filtro em quatro arestas, reproduz o MATLAB exatamente como
antes: o filtro **nunca atuava** nessa trajetória. Sem ramos em valores o lado direito ficou
puro, o que é o que permite o `jit` e a Jacobiana por diferenciação automática (036).

## Referências

[docs/severe_slug.md](../severe_slug.md) § "Filtro anti-refluxo removido".
