function analisaGeometriaAnu
% Função que plota informações relevantes sobre o regime intermitente.

aux=cd;
aux=aux(1:length(aux)-9);
addpath(aux)

D=0.1;
u_GS=10.^linspace(-1,2,100);
u_LS=ones(size(u_GS));
densL=800.*ones(size(u_GS));
densG=160.*ones(size(u_GS));
rugo=0;
angulo_teta=0.*ones(size(u_GS));
viscL=0.01;
viscG=3e-5;
CL=1e5;
CG=1e5;
tensao=0.1;
alfaG = 0.4;

[Sin,ls,lf,lu] = modelo_intermitente2 (alfaG,u_LS, u_GS,...
        D, rugo, angulo_teta,...
        densL, densG, viscL, viscG, CL, CG,tensao);

% [Ss,Sf,ls,lf,lu] = modelo_intermitente4 (alfaG,u_LS, u_GS,...
%     D, rugo, angulo_teta,...
%     densL, densG, viscL, viscG, CL, CG,tensao);


figure
loglog(u_GS,lu./ls,'b','linewidth',2)
title('Comprimento da unidade de Slug','fontsize',14)
xlabel('Velocidade Superficial Gas','fontsize',12)
ylabel('lu/ls','fontsize',12)
hold on

u_LS=ones(size(u_GS)).*2;

[Sin,ls,lf,lu] = modelo_intermitente2 (alfaG,u_LS, u_GS,...
        D, rugo, angulo_teta,...
        densL, densG, viscL, viscG, CL, CG,tensao);

loglog(u_GS,lu./ls,'r','linewidth',2)

u_LS=ones(size(u_GS)).*5;

[Sin,ls,lf,lu] = modelo_intermitente2 (alfaG,u_LS, u_GS,...
        D, rugo, angulo_teta,...
        densL, densG, viscL, viscG, CL, CG,tensao);

loglog(u_GS,lu./ls,'k','linewidth',2)
legend('Vel. Sup. Líq de 1 m/s','Vel. Sup. Líq de 2 m/s','Vel. Sup. Líq de 5 m/s')
set(gca,'Fontsize',12)

