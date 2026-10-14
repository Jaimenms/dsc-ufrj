function analisaGeometriaEst
% Função que plota informações relevantes sobre geometria estratificada.

alfaG = 0:0.01:1;

% Calculando estimativa inicial para hD
[hD,beta]=alfa2h2(alfaG);

% Calculando rigorosamente hD
[hD,beta]=alfa2h(alfaG,beta);

% Calculando Perímetros
PerL = beta/2;
PerG = pi-PerL;
PerI = sin(beta/2);

% Calculando Área
AreaL = pi./4.*(1-alfaG);
AreaG = pi./4.*alfaG;

%% Calculando diâmetros hidráulicos
ix = alfaG<1;
DG = 4.*AreaG./(PerG+PerI);
DL = DG.*0;
DL(ix) = 4.*AreaL(ix)./PerL(ix);

figure
plot(hD,1-alfaG,'-',hD,alfaG,'-.','linewidth',2)
title('Frações de Área','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Fração de Área','fontsize',12)
legend('Líquido','Gas')
set(gca,'Fontsize',12)

figure
plot(hD,beta,'linewidth',2)
title('Angulo de Contato','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Beta','fontsize',12)
set(gca,'Fontsize',12)

figure
plot(alfaG,beta,'linewidth',2)
title('Angulo de Contato','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('Beta','fontsize',12)
set(gca,'Fontsize',12)

figure
plot(hD,PerL,'-',hD,PerG,'-.',hD,PerI,'--','linewidth',2)
title('Perímetros adimensionais','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('Perímetro / Diâmetro','fontsize',12)
legend('Líquido','Gás','Interface')
set(gca,'Fontsize',12)

figure
plot(hD,DL,'-',hD,DG,'-.','linewidth',2)
title('Diâmetros Hidráulicos Adimensionais','fontsize',12)
xlabel('Altura de líquido / Diâmetro','fontsize',10)
ylabel('Diam. Hid. / Diâmetro','fontsize',10)
legend('Líquido','Gás')

zCML = 0.5-1./3./pi./(1-alfaG).*(sin(beta./2)).^3;
zCMG = 0.5+1./3./pi./(alfaG).*(sin(beta./2)).^3;

figure
plot(hD,zCML,'-',hD,zCMG,'-.','linewidth',2)
title('Posição dos Centros de Massas das fases','fontsize',14)
xlabel('Altura de líquido / Diâmetro','fontsize',12)
ylabel('z / Diâmetro','fontsize',12)
legend('Líquido','Gás')
set(gca,'Fontsize',12)

figure
plot(alfaG,zCML,'-',alfaG,zCMG,'-.','linewidth',2)
title('Posição dos Centros de Massas das fases','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('z / Diâmetro','fontsize',12)
legend('Líquido','Gás')
set(gca,'Fontsize',12)


densL = 1000;
densG = 1;
dPLG1 = ((densL-densG).*hD-densL.*zCML+densG.*zCMG)./densL;
dPL1 = alfaG.*dPLG1;
dPG1 = -(1-alfaG).*dPLG1;
dPI1  = ((1-alfaG).*densL.*(zCML-hD)+(alfaG).*densG.*(zCMG-hD))./densL;

densL = 1000;
densG = 100;
dPLG2 = ((densL-densG).*hD-densL.*zCML+densG.*zCMG)./densL;
dPL2 = alfaG.*dPLG2;
dPG2 = -(1-alfaG).*dPLG2;
dPI2  = ((1-alfaG).*densL.*(zCML-hD)+(alfaG).*densG.*(zCMG-hD))./densL;

densL = 1000;
densG = 250;
dPLG3 = ((densL-densG).*hD-densL.*zCML+densG.*zCMG)./densL;
dPL3 = alfaG.*dPLG3;
dPG3 = -(1-alfaG).*dPLG3;
dPI3  = ((1-alfaG).*densL.*(zCML-hD)+(alfaG).*densG.*(zCMG-hD))./densL;


figure
plot(hD,dPLG1,'-',hD,dPLG2,'-.',hD,dPLG3,'--','linewidth',2)
title('Diferença de pressão entre as fases','fontsize',14)
xlabel('Altura de Líquido / Diâmetro','fontsize',12)
ylabel('dPLG* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(alfaG,dPLG1,'-',alfaG,dPLG2,'-.',alfaG,dPLG3,'--','linewidth',2)
title('Diferença de pressão entre as fases','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('dPLG* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(hD,dPL1,'-',hD,dPL2,'-.',hD,dPL3,'--','linewidth',2)
title('Diferença de pressão da fase líquida','fontsize',14)
xlabel('Altura de Líquido / Diâmetro','fontsize',12)
ylabel('dPL* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(alfaG,dPL1,'-',alfaG,dPL2,'-.',alfaG,dPL3,'--','linewidth',2)
title('Diferença de pressão da fase líquida','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('dPL* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(hD,dPG1,'-',hD,dPG2,'-.',hD,dPG3,'--','linewidth',2)
title('Diferença de pressão da fase gás','fontsize',14)
xlabel('Altura de Líquido / Diâmetro','fontsize',12)
ylabel('dPG* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(alfaG,dPG1,'-',alfaG,dPG2,'-.',alfaG,dPG3,'--','linewidth',2)
title('Diferença de pressão da fase gás','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('dPG* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(hD,dPI1,'-',hD,dPI2,'-.',hD,dPI3,'--','linewidth',2)
title('Diferença de pressão da interface','fontsize',14)
xlabel('Altura de Líquido / Diâmetro','fontsize',12)
ylabel('dPI* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)

figure
plot(alfaG,dPI1,'-',alfaG,dPI2,'-.',alfaG,dPI3,'--','linewidth',2)
title('Diferença de pressão da interface','fontsize',14)
xlabel('Fração de Área de Gás','fontsize',12)
ylabel('dPI* (Pa/Pa)','fontsize',12)
legend('1 kg/m³','100 kg/m³','250 kg/m³')
set(gca,'Fontsize',12)





function [hD,beta]=alfa2h(alfaG,beta)
% Determinando altura de líquido /diâmetro para escoamento estratificado de
% forma rigorosa

Tol=1e-9;
maxIter=100;
[nx,ny] = size(alfaG);

for i=1:nx
    for j=1:ny
        alfaLi = (1-alfaG(i,j));
        if alfaLi>(1-1e-6) % Caso apenas Líquido
            betai=2*pi;
        elseif alfaLi<(1e-6) % Caso apenas Gás
            betai = 0;
        else % Caso intermediário
            % Preparando método
            cont=0;
            erro=1;
            betai=beta(i,j);
            % Executando Newton-Raphson
            while erro>Tol
                if cont>=maxIter;break,end
                F0=sin(betai)-betai+2*pi*alfaLi;
                dF=cos(betai)-1;
                passo = F0/dF;
                betai = betai - passo;
                erro  = abs(passo);
                cont=cont+1;
            end
        end
        beta(i,j) = betai;
    end
end

% Calculando hD a partir do angulo de molhamento
hD=1/2*(1-cos(beta/2));


function [hD,beta]=alfa2h2(alfaG)
% Determinando altura de líquido /diâmetro para escoamento estratificado de
% forma aproximada

% Polinomios
P_h=[0.00000000000197;...
       -3.66048527474641;...
       9.15121318685793;...
       -8.89104962298284;...
       4.18536124761860;...
       -1.75567780293538;...
       0.98531913309307]';

P_beta=1.0e+002.*...
    [0.00000000000056;...
     -0.67451330764088;...
     1.68628326909991;...
     -1.58733054088785;...
     0.69471254223251;...
     -0.17280444315654;...
     0.05824216671205]';

% Calculando hD
hD=polyval(P_h,alfaG);

% Calculando beta
beta=polyval(P_beta,alfaG);

% Tratando insonsistências
beta = max(0,beta);
beta = min(2.*pi,beta);


