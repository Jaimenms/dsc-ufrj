function [saidas1,saidas2,saidas3]=analisaRegimes(dados,estados,saidas)
% Rotina que analisa regimes de escoamento.

clc, clear functions, warning('off')
aux=cd;
aux=aux(1:length(aux)-9);
addpath(aux)

disp('(1) Executando cálculos iniciais')
for i=1:length(dados.tubo)
    dados.tubo{i}.regime=0;
end
[parametros]=calculos_iniciais(dados);
disp('Cálculos iniciais executados com sucesso')
disp('  ')

ne=4;
m = length(saidas.tubo);
for j=1:m
    
    densLmed = mean(mean(saidas.tubo{j}.densidadeL(:)));
    densGmed = mean(mean(saidas.tubo{j}.densidadeG(:)));
    Area = (pi*parametros.Duto{j}.D^2/4);
    
    qLmin = 0.001*Area*densLmed;
    qGmin = 0.001*Area*densGmed;
    qLmax = 10*Area*densLmed;
    qGmax = 10*Area*densGmed;
    
    alfa(:,1)= saidas.tubo{j}.alfaG(:);
    p(:,1)   = saidas.tubo{j}.pressao(:);
    q_L(:,1) = saidas.tubo{j}.vazaoMassaL(:);
    q_G(:,1) = saidas.tubo{j}.vazaoMassaG(:);

    alfam = min(min(alfa));
    alfaM = max(max(alfa));
    pm = min(min(p));
    pM = max(max(p));
    pmed = mean(mean(p));
    alfamed = mean(mean(alfa));
    q_Lmed = mean(mean(q_L));
    q_Gmed = mean(mean(q_G));

    np = 100;

    v_q_L=linspace(qLmin,qLmax,np);
    v_q_G=linspace(qGmin,qGmax,np);
    
    Y_1=[];
    for i=1:np
        for ii=1:np
            Y_1((i-1)*4+1,ii) =alfamed;
            Y_1((i-1)*4+2,ii) =pmed.*1e5./dados.simulador.pressaoR;
            Y_1((i-1)*4+3,ii) =v_q_L(i);
            Y_1((i-1)*4+4,ii) =v_q_G(ii);
        end
    end
    
    Y_2=[];
    for i=1:np
        for ii=1:np
            Y_2((i-1)*4+1,ii) =0.1;
            Y_2((i-1)*4+2,ii) =pmed.*1e5./dados.simulador.pressaoR;
            Y_2((i-1)*4+3,ii) =v_q_L(i);
            Y_2((i-1)*4+4,ii) =v_q_G(ii);
        end
    end
    
    Y_3=[];
    for i=1:np
        for ii=1:np
            Y_3((i-1)*4+1,ii) =0.9;
            Y_3((i-1)*4+2,ii) =pmed.*1e5./dados.simulador.pressaoR;
            Y_3((i-1)*4+3,ii) =v_q_L(i);
            Y_3((i-1)*4+4,ii) =v_q_G(ii);
        end
    end
        
    parametros.Duto{j}.Nk = size(Y_1,1)/ne;
    parametros.Duto{j}.D=parametros.Duto{j}.D(1);
    parametros.Duto{j}.angulo_teta=[parametros.Duto{j}.angulo_teta(1) parametros.Duto{j}.angulo_teta(1)];    
    parametros.Duto{j}.x=[parametros.Duto{j}.x(1) parametros.Duto{j}.x(end)];
    x = parametros.Duto{j}.x(1).*ones(parametros.Duto{j}.Nk,size(Y_1,2));

    
    disp('(2) Organizando saidas do simulador dinamico')
    [F,saidas1]=modelo_duto(Y_1,x,parametros.Duto{j},parametros.Fluido);
    [F,saidas2]=modelo_duto(Y_2,x,parametros.Duto{j},parametros.Fluido);
    [F,saidas3]=modelo_duto(Y_3,x,parametros.Duto{j},parametros.Fluido);
    disp('Saidas do simulador dinamico organizadas com sucesso.')
    disp('  ')
    
    
    figure
    CMAP = [0 0.7 0.25;0 0.3 0.6;0 0.3 0.9;0.5 0.5 0.5];
    x= saidas1.vazaoVolG./(pi*parametros.Duto{j}.D^2/4);
    y= saidas1.vazaoVolL./(pi*parametros.Duto{j}.D^2/4);
    z= saidas1.reg1;
    surf(x,y,z)
    axis tight
    set(gca,'View', [0 90],'CLim',[1 4])
    colormap(CMAP)
    a=colorbar;
    set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
    shading interp
    xlabel('Vel. Sup. Gás (m/s)')
    ylabel('Vel. Sup. Líquido (m/s)')
    title(['Regimes - Tubo ' num2str(j) ' - P e alfa médios'])
    
    figure
    x= saidas2.vazaoVolG./(pi*parametros.Duto{j}.D^2/4);
    y= saidas2.vazaoVolL./(pi*parametros.Duto{j}.D^2/4);
    z= saidas2.reg1;
    surf(x,y,z)
    axis tight
    set(gca,'View', [0 90],'CLim',[1 4])
    colormap(CMAP)
    a=colorbar;
    set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
    shading interp
    xlabel('Vel. Sup. Gás (m/s)')
    ylabel('Vel. Sup. Líquido (m/s)')
    title(['Regimes - Tubo ' num2str(j) ' - P média e alfa 0.1'])

    figure
    x= saidas3.vazaoVolG./(pi*parametros.Duto{j}.D^2/4);
    y= saidas3.vazaoVolL./(pi*parametros.Duto{j}.D^2/4);
    z= saidas3.reg1;
    surf(x,y,z)
    axis tight
    set(gca,'View', [0 90],'CLim',[1 4])
    colormap(CMAP)
    a=colorbar;
    set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
    shading interp
    xlabel('Vel. Sup. Gás (m/s)')
    ylabel('Vel. Sup. Líquido (m/s)')
    title(['Regimes - Tubo ' num2str(j) ' - P média e alfa 0.9'])
    
end

