# Registo dos ficheiros golden

- 2026-10-13 d2955c0 `chapter10.npz` criado: geração inicial
- 2026-10-13 d2955c0 `runs_meta.npz` criado: geração inicial
- 2026-10-13 d2955c0 `variants.npz` criado: geração inicial
- 2026-10-13 d2955c0 `severe.npz` criado: geração inicial
- 2026-10-13 d2955c0 `results.npz` criado: geração inicial
- 2026-10-13 d2955c0 `nightly.npz` criado: geração inicial
- 2026-10-13 9c43f24 `severe.npz` regenerado (desvio máx. relativo 3.01e-01): fase 2e: lado direito JAX = mesmo código numpy com backend injetado (mfsim.compiled.CompiledNetwork); rhs_jax/jac_jax diferem do JAX anterior a 2e-15/5e-16 relativo (reordenação no grafo compilado) -> nível B rtol 1e-9; trajetórias JAX (referência 150 s, Jansen 10 s) seguem outra sequência de passos (sensibilidade caótica): referência 1787 passos (antes 2556; numpy 1871), 1.º blowdown 48 s (antes 42 s; numpy 48 s), P_min 4.006 bar (4.029; numpy 4.028), P_max 6.058 (igual), pico qL 6.42 kg/s (6.45; numpy 6.44), balanço acumulado 18.88/-0.788 kg (18.75/-0.778); Jansen 10 s 7037 passos (antes 7102), desvio máx. 1e-2; arrays numpy bit-idênticos
- 2026-10-13 3c82ea2 `nightly.npz` regenerado (desvio máx. relativo 9.64e-01): fase 2e: caminho JAX unificado (mfsim.compiled); trajetórias JAX seguem outra sequência de passos (ver severe.npz)
- 2026-10-13 3c82ea2 `severe.npz` regenerado (desvio máx. relativo 5.90e-01): correção do tangente NaN (0 * inf) do diâmetro máximo de bolha com a mistura parada (bubble, regime.dmax): a Jacobiana JAX nas células paradas passa a ser finita; valores numpy inalterados (bit-idênticos); trajetórias JAX seguem outra sequência de passos
- 2026-10-13 3c82ea2 `nightly.npz` regenerado (desvio máx. relativo 0.00e+00): correção do tangente NaN (0 * inf) do diâmetro máximo de bolha com a mistura parada (bubble, regime.dmax): a Jacobiana JAX nas células paradas passa a ser finita; valores numpy inalterados (bit-idênticos); trajetórias JAX seguem outra sequência de passos
