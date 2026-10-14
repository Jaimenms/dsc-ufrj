#!/bin/zsh
# Corridas de validação com dados de laboratório (item 8). uso: zsh scratch/val_launch.sh
cd /Users/jaimenms/code/personal/doutorado
AW='"rhoL":998.0,"muL":1e-3,"rhoG0":1.2,"muG":1.8e-5,"P0":1.013e5,"sigma":0.072,"gammaG":1.0,"P_out":1.013,"P_gas_ref":1.013'
NUM='"conservative":1,"relax_target":"noslip","high_void":1,"virtual_mass":0.5,"low_void":2,"art_visc":0.05,"tol_range":0.01,"rtol":0.003,"jax":1,"max_newton":20,"start":"blocked","dP_margin":0.05'
FIX='"regime":4,"rv":2,"vblend":20.0'
AUTO='"regime":6,"ipe_smooth":1,"ipe_local":1,"k_switch":20,"vblend":0.0'
# Fabre et al. (1990) via Tengesdal (2002): 2" (50.8 mm), linha 25 m a -0.57 graus, riser 13.5 m, ar-água, saída atmosférica
FAB='"D":0.0508,"L_h1":1.0,"n_h1":2,"L_down":25.0,"n_down":50,"slope":-0.57,"H":13.5,"n_riser":27,"L_h2":1.0,"n_h2":2,"dt_out":0.5'
# Jansen et al. (1996) / Taitel et al. (1990): 1" (25.4 mm), linha 9.1 m a -1 grau, riser 3 m, +10 m equivalentes de gás a montante (L_h1 = 15 m com hold-up ~1/3)
JAN='"D":0.0254,"L_h1":15.0,"n_h1":15,"L_down":9.1,"n_down":36,"slope":-1.0,"H":3.0,"n_riser":12,"L_h2":0.5,"n_h2":2,"dt_out":0.25'
# Tengesdal (2002) instalação grande: 3" (76.2 mm), linha 65 ft (19.81 m) + tanques até 280 ft efetivos (L_h1 = 65 m), riser 49 ft (14.63 m), óleo Crystex AF-M (SG 0.856, ~26 mPa.s), ar
TEN='"D":0.0762,"rhoL":856.0,"muL":0.026,"rhoG0":1.2,"muG":1.8e-5,"P0":1.013e5,"sigma":0.03,"gammaG":1.0,"P_out":1.013,"P_gas_ref":1.013,"L_h1":65.0,"n_h1":20,"L_down":19.81,"n_down":40,"slope":-1.0,"H":14.63,"n_riser":30,"L_h2":1.0,"n_h2":2,"dt_out":0.5'
run() { tag=$1; shift; nohup uv run python -u scratch/severe_D.py "{$*}" $tag > scratch/severe_D_$tag.log 2>&1 & echo "lançado $tag (pid $!)"; }
run val_fabre2_fix  "$AW,$NUM,$FIX,$FAB,\"vSL\":0.13,\"vSG\":0.20,\"tf\":1000.0"
run val_fabre2_auto "$AW,$NUM,$AUTO,$FAB,\"vSL\":0.13,\"vSG\":0.20,\"tf\":1000.0"
run val_fabre1_fix  "$AW,$NUM,$FIX,$FAB,\"vSL\":0.13,\"vSG\":0.45,\"tf\":800.0"
run val_fabre1_auto "$AW,$NUM,$AUTO,$FAB,\"vSL\":0.13,\"vSG\":0.45,\"tf\":800.0"
run val_jansen_fix  "$AW,$NUM,$FIX,$JAN,\"vSL\":0.0959,\"vSG\":0.0753,\"tf\":400.0"
run val_jansen_auto "$AW,$NUM,$AUTO,$JAN,\"vSL\":0.0959,\"vSG\":0.0753,\"tf\":400.0"
run val_teng1_fix   "$NUM,$FIX,$TEN,\"vSL\":0.32,\"vSG\":0.35,\"tf\":600.0"
