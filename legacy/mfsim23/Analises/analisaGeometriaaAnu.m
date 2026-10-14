function analisaGeometriaAnu
% Função que plota informações relevantes sobre geometria anular.

alfaG = 0.01:0.01:0.99;

% Calculando estimativa inicial para hD
hD = 0.5.*(1-sqrt(alfaG));

% Calculando Perímetros
PerL = pi;
PerG = 0;
PerI = pi.*sqrt(alfaG);

% Calculando Área
AreaL = pi./4.*(1-alfaG);
AreaG = pi./4.*alfaG;

%% Calculando diâmetros hidráulicos
DG = 4.*AreaG./(PerG+PerI);
DL = 4.*AreaL./PerL;

figure
plot(hD,1-alfaG,'-',hD,alfaG,'-.','linewidth',2)
title('Frações de Área','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Fração de Área','fontsize',12)
legend('Líquido','Gas')
set(gca,'Fontsize',12)

figure
plot(hD,PerL,'-',hD,PerG,'-.',hD,PerI,'--','linewidth',2)
title('Perímetros adimensionais','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Perímetro / Diâmetro','fontsize',12)
legend('Líquido','Gás','Interface')
set(gca,'Fontsize',12)

figure
plot(alfaG,PerL,'-',alfaG,PerG,'-.',alfaG,PerI,'--','linewidth',2)
title('Perímetros adimensionais','fontsize',14)
xlabel('Fração de Área','fontsize',12)
ylabel('Perímetro / Diâmetro','fontsize',12)
legend('Líquido','Gás','Interface')
set(gca,'Fontsize',12)

figure
plot(alfaG,DL,'-',alfaG,DG,'-.','linewidth',2)
title('Diâmetros Hidráulicos Adimensionais','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Diam. Hid. / Diâmetro','fontsize',12)
legend('Líquido','Gás')
set(gca,'Fontsize',12)

figure
plot(alfaG,DL,'-',alfaG,DG,'-.','linewidth',2)
title('Diâmetros Hidráulicos Adimensionais','fontsize',14)
ylabel('Diam. Hid. / Diâmetro','fontsize',12)
legend('Líquido','Gás')
xlabel('Fração de Área','fontsize',12)
set(gca,'Fontsize',12)
