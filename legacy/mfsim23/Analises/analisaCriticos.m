function analisaCriticos
% Função que plota o gráfico como número de Froude Crítico

aux=cd;
aux=aux(1:length(aux)-9);
addpath(aux)

hL = 0:0.01:1;

D = 1;

beta = 2.*acos(1-2.*hL);

alfaL = (beta-sin(beta))./2./pi;

alfaG = 1-alfaL;

Area = pi.*D.^2./4;

AreaG = Area.*alfaG;

PerI = D.*sin(beta./2);

FGC = ((1-hL).^2.*alfaG.^3.*(pi./4.*D)./PerI).^0.5;


figure
semilogy(hL,FGC,'linewidth',2)
title('N° de Froude Crítico','fontsize',12)
xlabel('Altura de camada de líquido (-)','fontsize',10)
ylabel('FGc*','fontsize',10)
ylim([1e-3 3])