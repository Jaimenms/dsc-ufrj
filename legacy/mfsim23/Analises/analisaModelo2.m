% Rotina que analisa os coeficiente do modelo de escoamento bifasico.

clear all
aux=cd;
aux=aux(1:length(aux)-9);
addpath(aux)

% Variando as Vazões
vL = 1;
vG = 1;

N=200;

alfa1=0.1;
alfa2=0.9;
PT1=1;
PT2=20;
D = 0.1;
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

Area = pi.*D.^2/4;

alfa=alfa1+(alfa2-alfa1)./N.*[0:N];
PT=PT1+(PT2-PT1)./N.*[0:N];

alfa=ones(length(alfa),1)*alfa;
PT=ones(length(PT),1)*PT;
PT=PT';

nx=size(alfa,1);


P = PT.*parametrosFluido.P0;

[propL, propG, tensao] = modelo_fluido(P,0,parametrosFluido);
densL = propL.dens;
densG = propG.dens;
qL = Area.*vL.*densL;
qG = Area.*vG.*densG;

y([1:4:4*nx],:)=alfa;
y([2:4:4*nx],:)=PT;
y([3:4:4*nx],:)=qL;
y([4:4:4*nx],:)=qG;

parametrosDuto.angulo_teta = 0;
parametrosDuto.x = [0 1];
parametrosDuto.angulo_teta = [0 0];
parametrosDuto.xCD_slug=[0 1];
parametrosDuto.CD_slug=[1 1];
x = 1;


parametros{1} = parametrosDuto;
parametros{1}.Fluido = parametrosFluido;


[F]=modelo_duto(zeros(size(alfa)),ones(size(alfa)),y,parametros,1);

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


% for j=1:nx,
%     for k=1:nx,
%         A = [0 0 F1(j,k) F2(j,k);...
%              0 0 F3(j,k) F4(j,k);...
%              F5(j,k) F6(j,k) F7(j,k) 0;...
%              F8(j,k) F9(j,k) 0 F10(j,k)];
% 
%         E = eig(A,'nobalance');
%         [aux,iEaux] = sort(real(E));
% 
%         Eaux=E(iEaux);            
% 
%         EIG_1s(j,k)=Eaux(1);
%         EIG_2s(j,k)=Eaux(2);
%         EIG_3s(j,k)=Eaux(3);
%         EIG_4s(j,k)=Eaux(4);
% 
%         EIG_1(j,k)=E(1);
%         EIG_2(j,k)=E(2);
%         EIG_3(j,k)=E(3);
%         EIG_4(j,k)=E(4);
% 
%     end
% end
    
tit{1} = 'Coeficiente M_A _1_,_3';
tit{2} = 'Coeficiente M_A _1_,_4';
tit{3} = 'Coeficiente M_A _2_,_3';
tit{4} = 'Coeficiente M_A _2_,_4';
tit{5} = 'Coeficiente M_A _3_,_1';
tit{6} = 'Coeficiente M_A _3_,_2';
tit{7} = 'Coeficiente M_A _3_,_3';
tit{8} = 'Coeficiente M_A _4_,_1';
tit{9} = 'Coeficiente M_A _4_,_2';
tit{10} = 'Coeficiente M_A _4_,_4';
tit{11} = 'Coeficiente M_B _3';
tit{12} = 'Coeficiente M_B _4';
for i=1:12
            
    figure(i)
    
    colormap ([0 0 0])
    [C,h] = contour(alfa,P,F(nx*(i-1)+[1:nx],:));
    set(h,'linewidth',2)
    text_handle = clabel(C,h,'labelspacing',300);
    set(text_handle,'BackgroundColor',[1 1 .6],...
        'Edgecolor',[.7 .7 .7])

    xlabel('Fração Área de Gás (-)','Fontsize',12)
    ylabel('Pressão (Pa)','Fontsize',12)
    title(tit,'Fontsize',14)
    set(gca,'fontsize',12)
    shading interp
    axis tight
    pause
    
end

    
%     figure(11),set(gcf,'Position',[360   200   348   722])
%     subplot(3,1,i)
%     surf(qL,qG,EIG_1)
%     title(['Primeiro Autovalor para alfaG de ' num2str(alfai)])
%     xlabel('qL (m3/s)')
%     ylabel('qG (m3/s)')
%     zlabel('F10')
%     shading interp

%     figure((i-1)*6+1)
%     subplot(2,1,1)
%     surf(qL,qG,abs(EIG_2s))
%     title(['Autovalor 1 para alfaG de ' num2str(alfai)])
%     zlabel('Parte Real')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
%     subplot(2,1,2)
%     surf(qL,qG,imag(EIG_2s))
%     zlabel('Parte Imaginária')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
% 
%     figure((i-1)*6+2)
%     subplot(2,1,1)
%     surf(qL,qG,abs(EIG_1s))
%     title(['Autovalor 1 para alfaG de ' num2str(alfai)])
%     zlabel('Parte Real')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
%     subplot(2,1,2)
%     surf(qL,qG,imag(EIG_1s))
%     zlabel('Parte Imaginária')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp    
% 
%     figure((i-1)*6+3)
%     subplot(2,1,1)
%     surf(qL,qG,abs(EIG_2s))
%     title(['Autovalor 2 para alfaG de ' num2str(alfai)])
%     zlabel('Parte Real')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
%     subplot(2,1,2)
%     surf(qL,qG,imag(EIG_2s))
%     zlabel('Parte Imaginária')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp    
% 
%     figure((i-1)*6+4)
%     subplot(2,1,1)
%     surf(qL,qG,abs(EIG_3s))
%     title(['Autovalor 3 para alfaG de ' num2str(alfai)])
%     zlabel('Parte Real')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
%     subplot(2,1,2)
%     surf(qL,qG,imag(EIG_3s))
%     zlabel('Parte Imaginária')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp    
% 
%     figure((i-1)*6+5)
%     subplot(2,1,1)
%     surf(qL,qG,abs(EIG_4s))
%     title(['Autovalor 4 para alfaG de ' num2str(alfai)])
%     zlabel('Parte Real')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp
%     subplot(2,1,2)
%     surf(qL,qG,imag(EIG_4s))
%     zlabel('Parte Imaginária')
%     xlabel('qL (kg/s)')
%     ylabel('qG (kg/s)')
%     shading interp    

    
%     figure((i-1)*6+6+10)
%     subplot(2,1,1)
%     ix=20;
%     plot(qG(:,ix),real(EIG_1s(:,ix)),'r-','linewidth',2)
%     hold on
%     plot(qG(:,ix),real(EIG_2s(:,ix)),'b-','linewidth',2)
%     plot(qG(:,ix),real(EIG_3s(:,ix)),'k-','linewidth',2)
%     plot(qG(:,ix),real(EIG_4s(:,ix)),'m-','linewidth',2)
%     xlabel('qG (kg/s)')
%     ylabel('Parte Real')
%     title(['Vazão de líquido ' num2str(qL(1,ix)) ' e alfaG de ' num2str(alfai)])
%     subplot(2,1,2)
%     ix=20;
%     plot(qG(:,ix),imag(EIG_1s(:,ix)),'r-','linewidth',2)
%     hold on
%     plot(qG(:,ix),imag(EIG_2s(:,ix)),'b-','linewidth',2)
%     plot(qG(:,ix),imag(EIG_3s(:,ix)),'k-','linewidth',2)
%     plot(qG(:,ix),imag(EIG_4s(:,ix)),'m-','linewidth',2)
%     xlabel('qG (kg/s)')
%     ylabel('Parte Imaginária')
    
%     figure((i-1)*6+6+11)
%     eixoX = saidas.velocidadeG(:,ix)-saidas.velocidadeL(:,ix);
%     subplot(2,1,1)
%     ix=20;
%     plot(eixoX,real(EIG_1s(:,ix)),'r-','linewidth',2)
%     hold on
%     plot(eixoX,real(EIG_2s(:,ix)),'b-','linewidth',2)
%     plot(eixoX,real(EIG_3s(:,ix)),'k-','linewidth',2)
%     plot(eixoX,real(EIG_4s(:,ix)),'m-','linewidth',2)
%     xlabel('vG - vL (m/s)')
%     ylabel('Parte Real')
%     title(['Vazão de líquido ' num2str(qL(1,ix)) ' e alfaG de ' num2str(alfai)])
%     subplot(2,1,2)
%     ix=20;
%     plot(eixoX,imag(EIG_1s(:,ix)),'r-','linewidth',2)
%     hold on
%     plot(eixoX,imag(EIG_2s(:,ix)),'b-','linewidth',2)
%     plot(eixoX,imag(EIG_3s(:,ix)),'k-','linewidth',2)
%     plot(eixoX,imag(EIG_4s(:,ix)),'m-','linewidth',2)
%     xlabel('vG - vL (m/s)')
%     ylabel('Parte Imaginária')


%     IKH = (saidas.densidadeL-saidas.densidadeG).*(saidas.densidadeL.*saidas.alfaG+saidas.densidadeG.*saidas.alfaL)./saidas.densidadeL./saidas.densidadeG.*9.8.*1.*pi.*parametrosDuto.D.^2/4./(parametrosDuto.D.*sin(saidas.betah./2));
%     vR=saidas.velocidadeG-saidas.velocidadeL;
%     FG=saidas.fFGcos;
%     VKH = ((1-saidas.hD).^2.*saidas.alfaG.^3.*pi./4./sin(saidas.betah./2)).^0.5;
%     figure((i-1)*6+6+11)
%     subplot(2,1,1)
%     plot(qG(:,ix),IKH(:,ix),'r-','linewidth',2)
%     hold on
%     plot(qG(:,ix),[vR(:,ix)].^2,'b-','linewidth',2)    
%     hold off
%     legend('IKH','vR^2')
% 
%     subplot(2,1,2)
%     plot(qG(:,ix),FG(:,ix),'b-','linewidth',2)
%     hold on
%     plot(qG(:,ix),VKH(:,ix),'r-','linewidth',2)    
%     hold off
%     legend('FG','VKH')
% 
%     
%     pause



% for i=1:10
%     a=figure(i);
%     b=get(a,'children');
%     maxc=[];
%     minc=[];
%     for j=1:length(b)
%         c=get(b(j),'children');
%         maxc(j)=max(max(get(c,'Zdata')));
%         minc(j)=min(min(get(c,'Zdata')));
%     end
%     for j=1:length(b)
%         set(b(j),'Zlim',[min(minc)-0.1.*abs(min(minc)) max(maxc)+0.1.*abs(max(maxc))])
%     end
% end



