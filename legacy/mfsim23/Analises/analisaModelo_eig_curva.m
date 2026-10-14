% Rotina que analisa os coeficiente do modelo de escoamento bifasico.

clear all
aux=cd;
aux=aux(1:length(aux)-9);
addpath(aux)

% Variando as Vazões
ii=0;
for alfa0=0.1:0.05:0.9
    ii=ii+1;
    alfai = alfa0;

    vL=1;
    vG=2;
    densL = 1000;
    densG =1;
    D = 0.1;

    Area = pi.*D.^2/4

    vL=ones(length(vL),1)*vL;
    vG=ones(length(vL),1)*vG;
    vG=vG';

    nx=size(vL,1);

    alfa = alfai.*ones(size(vL));
    pt = 1.*ones(size(vL));
    qL = Area.*vL.*densL;
    qG = Area.*vG.*densG;

    y([1:4:4*nx],:)=alfa;
    y([2:4:4*nx],:)=pt;
    y([3:4:4*nx],:)=qL;
    y([4:4:4*nx],:)=qG;

    parametrosDuto.angulo_teta = 0.*ones(size(qL,1),1);
    parametrosDuto.rugo = 0;
    parametrosDuto.Nk=1;
    parametrosDuto.D = 0.1;
    parametrosDuto.PR = 1e5;
    parametrosDuto.regime = 4;
    parametrosFluido.densL0 = 1000;
    parametrosFluido.densG0 = 1;
    parametrosFluido.P0 = 1e5;
    parametrosFluido.gamaL = 3000;
    parametrosFluido.gamaG = 1.44;
    parametrosFluido.tensao = 0.3;
    parametrosFluido.viscL = 1e-3;
    parametrosFluido.viscG = 1.78e-5;

    parametrosDuto.angulo_teta = 0;
    parametrosDuto.x = [0 1];
    parametrosDuto.angulo_teta = [0 0];
    x = 1;

    [F,saidas]=modelo_duto(y,x,parametrosDuto,parametrosFluido);

    F1=F(nx*(1-1)+[1:nx],:);
    F2=F(nx*(2-1)+[1:nx],:);
    F3=F(nx*(3-1)+[1:nx],:);
    F4=F(nx*(4-1)+[1:nx],:);
    F5=F(nx*(5-1)+[1:nx],:);
    F6=F(nx*(6-1)+[1:nx],:);
    F7=F(nx*(7-1)+[1:nx],:);
    F8=F(nx*(8-1)+[1:nx],:);
    F9=F(nx*(9-1)+[1:nx],:);
    F10=F(nx*(10-1)+[1:nx],:);
    F11=F(nx*(11-1)+[1:nx],:);
    F12=F(nx*(12-1)+[1:nx],:);


    for j=1:nx,
        for k=1:nx,
            A = [0 0 F1(j,k) F2(j,k);...
                 0 0 F3(j,k) F4(j,k);...
                 F5(j,k) F6(j,k) F7(j,k) 0;...
                 F8(j,k) F9(j,k) 0 F10(j,k)];

            E = eig(A,'nobalance');
            [aux,iEaux] = sort(real(E));

            Eaux=E(iEaux);            

            EIG_1s(j,k)=Eaux(1);
            EIG_2s(j,k)=Eaux(2);
            EIG_3s(j,k)=Eaux(3);
            EIG_4s(j,k)=Eaux(4);

            EIG_1(j,k)=E(1);
            EIG_2(j,k)=E(2);
            EIG_3(j,k)=E(3);
            EIG_4(j,k)=E(4);

        end
    end
    
    aux1(ii)=EIG_1s;
    aux2(ii)=EIG_2s;
    aux3(ii)=EIG_3s;
    aux4(ii)=EIG_4s;    
    auxa(ii)=alfa0;
end

figure(1)
subplot(2,2,1)
plot(auxa,aux1,'bo-','linewidth',2);
axis tight
grid on
xlabel('Fração Área Gás')
ylabel('Autovalores (m/s)')
subplot(2,2,2)
plot(auxa,aux2,'bo-','linewidth',2);
axis tight
grid on
xlabel('Fração Área Gás')
ylabel('Autovalores (m/s)')
subplot(2,2,3)
plot(auxa,aux3,'bo-','linewidth',2);
axis tight
grid on
xlabel('Fração Área Gás')
ylabel('Autovalores (m/s)')
subplot(2,2,4)
plot(auxa,aux4,'bo-','linewidth',2);
axis tight
grid on
xlabel('Fração Área Gás')
ylabel('Autovalores (m/s)')

