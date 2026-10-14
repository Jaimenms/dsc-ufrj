function analisaDifPres
% Função que plota as diferenças de pressão para o regime estratificado

addpath('C:\Arquivos de programas\MATLAB\R2007a\work\mfsim11')

densL = 1000;
densG = 1;
D=1;
angulo_teta=0;

CL=1;
CG=1;

alfa = 0:0.01:1;

[Est,Area,AreaG,AreaL]=geometriaEst(D,alfa);

[dPL,dPG,ddPLdP,ddPGdP,ddPLdalfaG,ddPGdalfaG]=difpresEstrat...
    (densL,densG,Est.betah,1-alfa,alfa,D,angulo_teta,CL,CG);

hDdifpresL=dPL./densL./9.81./D;

hDdifpresG=dPG./densG./9.81./D;

subplot(2,1,1)
plot(alfa,hDdifpresL,alfa,hDdifpresG,'linewidth',2)
xlabel('Fração de Área de Gás (m²/m²)','fontsize',12)
title('Diferença de pressão (m) / Diâmetro (m)','fontsize',14)
a=legend('Fase Líquida', 'Fase Gás','location','north');
set(a,'fontsize',12)

subplot(2,1,2)
plot(alfa,Est.betah,'linewidth',2)
xlabel('Fração de Área de Gás (m²/m²)','fontsize',12)
title('Beta','fontsize',14)
