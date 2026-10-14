#!/bin/zsh
# Corridas de validação (item 8), partida de linha fechada (shut_in, rampa 20 s), fechamento de bolhas da tese e físico.
cd /Users/jaimenms/code/personal/doutorado
AW='"rhoL":998.0,"muL":1e-3,"rhoG0":1.2,"muG":1.8e-5,"P0":1.013e5,"sigma":0.072,"gammaG":1.0,"P_out":1.013,"P_gas_ref":1.013'
NUM='"conservative":1,"relax_target":"noslip","high_void":1,"virtual_mass":0.5,"low_void":2,"art_visc":0.05,"tol_range":0.01,"rtol":0.003,"jax":1,"max_newton":20,"start":"shut_in","t_ramp":20.0,"gammaL":100.0'
FIX='"regime":4,"rv":2,"vblend":20.0'
PHYS='"bubble_drag":"fisico"'
FAB='"D":0.0508,"L_h1":1.0,"n_h1":2,"L_down":25.0,"n_down":50,"slope":-0.57,"H":13.5,"n_riser":27,"L_h2":1.0,"n_h2":2,"dt_out":0.5'
JAN='"D":0.0254,"L_h1":15.0,"n_h1":15,"L_down":9.1,"n_down":36,"slope":-1.0,"H":3.0,"n_riser":12,"L_h2":0.5,"n_h2":2,"dt_out":0.25'
TEN='"D":0.0762,"rhoL":856.0,"muL":0.026,"rhoG0":1.2,"muG":1.8e-5,"P0":1.013e5,"sigma":0.03,"gammaG":1.0,"P_out":1.013,"P_gas_ref":1.013,"L_h1":65.0,"n_h1":20,"L_down":19.81,"n_down":40,"slope":-1.0,"H":14.63,"n_riser":30,"L_h2":1.0,"n_h2":2,"dt_out":0.5'
run() { tag=$1; shift; nohup uv run python -u scratch/severe_D.py "{$*}" $tag > scratch/severe_D_$tag.log 2>&1 & echo "lançado $tag (pid $!)"; }
run v3_jansen_phys "$AW,$NUM,$FIX,$PHYS,$JAN,\"vSL\":0.0959,\"vSG\":0.0753,\"tf\":600.0"
run v3_jansen_tese "$AW,$NUM,$FIX,$JAN,\"vSL\":0.0959,\"vSG\":0.0753,\"tf\":600.0"
run v3_fabre2_phys "$AW,$NUM,$FIX,$PHYS,$FAB,\"vSL\":0.13,\"vSG\":0.20,\"tf\":1200.0"
run v3_fabre2_tese "$AW,$NUM,$FIX,$FAB,\"vSL\":0.13,\"vSG\":0.20,\"tf\":1200.0"
run v3_fabre1_phys "$AW,$NUM,$FIX,$PHYS,$FAB,\"vSL\":0.13,\"vSG\":0.45,\"tf\":1000.0"
run v3_teng1_phys  "$NUM,$FIX,$PHYS,$TEN,\"vSL\":0.32,\"vSG\":0.35,\"tf\":800.0"
