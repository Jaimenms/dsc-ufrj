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

## Protótipo no problema-modelo (issue 056, 2026-10-17)

`papers/01-severe-slugging-two-fluid/toy/model_problem_jaxbdf.py`: o BDF do simulador inteiro num `lax.while_loop`
(Newton com Armijo, `jacfwd`, LU denso) reproduz a sequência de passos do `ImplicitBDF` nos quatro casos do
problema-modelo (626 a 43 587 passos) com trajetórias a menos de 1,5e−4. Sensibilidade dG/da por `jvp` através do
laço adaptativo: concorda com diferenças finitas a 1 % (Hopf) e 0,004 % (relaxação), 4× o custo de uma integração.
Modo inverso por reposição em grelha congelada (`scan`): a 1 % na Hopf, mas 16 % fora na relaxação (gradiente do
operador de reposição, não do integrador adaptativo) e com todos os intermédios em memória. O tempo (3–5× mais rápido)
é do problema pequeno: no offshore o custo está na Jacobiana e no Newton, não no laço. Conclusão: o degrau 1 é o
caminho para o simulador; o degrau 2 só com muitos parâmetros e *checkpointing*.

## Onde vai ser usado

O degrau 1 com a sensibilidade em relação ao **estado inicial** dá os produtos M·v da matriz de monodromia, que é o que o
*shooting* de órbitas periódicas do artigo 03 precisa (`papers/03-severe-slugging-bifurcations/PLAN.md`); os gradientes
das grandezas do ciclo servem também ao artigo 02.

## Cuidado físico

A golfada severa é um ciclo-limite com sensibilidade da trajetória ponto a ponto que cresce com o tempo. O que faz
sentido derivar são médias sobre ciclos (produção, período, amplitude, por médias de janela) ou, melhor, o estacionário
e os seus autovalores: ∂λ/∂p pela derivada do problema generalizado J v = λ M v (048), que localiza a bifurcação de
Hopf em função dos parâmetros sem integrar no tempo — este último é o mais barato e já está a meio caminho.
