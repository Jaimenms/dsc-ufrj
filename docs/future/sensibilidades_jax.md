# Sensibilidades através da integração (plano)

*Pergunta do dono (2026-10-17): com o BDF em JAX ganha-se o gradiente em relação aos parâmetros da EDO através de toda a
integração. Vale a pena explorar?* Sim; em dois degraus, depois de o artigo 01 fechar.

## Para que serve

- Calibração dos fechamentos contra os dados de laboratório (Fabre, Jansen): ∂(período, amplitude, holdup)/∂(constantes
  de Barnea e Brauner, v_GT, C_vm, k das sigmoides) em vez de varrimentos.
- Projeto e operação: ∂(produção média, pico de líquido)/∂(abertura, RGO, PI, pressão do separador) — o mapa de
  estabilidade passa a problema de otimização com restrição (máxima produção sem golfada).
- Sensibilidade do artigo: que parâmetros mandam no período e no surto, ao custo de uma corrida.

## Degrau 1 — sensibilidades discretas pelo teorema da função implícita (sem reescrever o BDF)

No passo aceite o resíduo é R(x_{n+1}, p) = T(x_{n+1}) − c_hist − γ f(x_{n+1}, p) = 0, logo

    dx_{n+1}/dp = −(J_T − γ J)⁻¹ [ −γ ∂f/∂p − ∂c_hist/dp ],

com ∂c_hist/dp a recorrência BDF das sensibilidades anteriores (as mesmas combinações w1, w2 das massas). A
fatorização LU de (J_T − γJ) que o Newton já tem serve: uma substituição por parâmetro por passo. ∂f/∂p vem do JAX
(`jacfwd` em relação aos parâmetros, não ao estado), o que exige só que os parâmetros a derivar entrem no lado direito
compilado como argumentos (hoje são constantes fechadas no `jit`). Modo direto, bom para dezenas de parâmetros;
reproduz exatamente a sensibilidade da trajetória que o integrador calcula, sem diferenciar as iterações do Newton nem
o controlo de passo. Onde pôr: `ImplicitBDF.integrate(..., sens=params)` devolvendo S(t) = dx/dp nos instantes de
saída; `mfsim.compiled.get_jax` com um segundo traçado `dfdp`.

## Degrau 2 — BDF em JAX com adjunto

`lax.while_loop` sobre os passos com `custom_vjp` no Newton (diferenciação implícita no ponto convergido): modo inverso,
custo de uma corrida independentemente do número de parâmetros. Dificuldades: trajetórias de comprimento variável
(passos rejeitados, paragem adaptativa) dentro do laço, checkpointing para a memória, e os ramos que não são `where`
(projeção nos limites, troca de ordem do BDF, seleção de entradas nas tabelas, hoje com `stop_gradient`) dão gradientes
por troços.

## Cuidado físico

A golfada severa é um ciclo-limite com sensibilidade da trajetória ponto a ponto que cresce com o tempo. O que faz
sentido derivar são médias sobre ciclos (produção, período, amplitude, por médias de janela) ou, melhor, o estacionário
e os seus autovalores: ∂λ/∂p pela derivada do problema generalizado J v = λ M v (048), que localiza a bifurcação de
Hopf em função dos parâmetros sem integrar no tempo — este último é o mais barato e já está a meio caminho.
