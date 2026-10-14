function [dados,estados,saidas,eixos]=simulador(dados,estados,flagSS,flagSS2,flagD,flagRES,flagCD)
% Função que executa a simulação bifásica de escoamento em redes de dutos.

%% Tratamentos Iniciais
clc, clear functions, warning('off')
global parametros
tic
tempo_cpu = cputime;

disp('(1) Executando cálculos iniciais')
[parametros]=calculos_iniciais(dados);
disp('Cálculos iniciais executados com sucesso')
disp('  ')

disp('(2) Desempacotando variaveis')
T = parametros.T;
Tol    = dados.simulador.tolerancia;
Delta  = dados.simulador.delta;
Delta2 = dados.simulador.delta2;
maxIter= dados.simulador.maxIter;

ode = dados.simulador.tipoOde;
disp('Variaveis desempacotadas com sucesso')
disp('  ')

if ~exist('flagCD'),flagCD=0;end

%% Executando simulação estacionária
if flagSS

    disp('(3) Gerando estimativa inicial')
    [X0]=simulador_SS(parametros,dados);    
    disp('Estimativa inicial gerada com sucesso')
    disp('  ')

    clear functions

    disp('(4) Gerando saidas secudarias do estado estacionario')
    [dXdt,Y0,saidas,eixos]=modelo_rede(0,X0,parametros);

    disp('Saidas do estado estacionario organizadas com sucesso')
    disp('  ')

    disp('(5) Atualizando condições iniciais')
    dados = XY2dados(X0,Y0,dados,parametros);
    disp('Condições iniciais atualizadas com sucesso')
    disp('   ')

    T = 0;
    X = X0;
    
    estados.T = T;
    estados.X = X;
    
    disp('Tempo Real          : ESTACIONÁRIA')
    disp(['Tempo Transcorrido : ' num2str(toc) ' s'])
    disp(['Tempo de Máquina   : ' num2str(cputime-tempo_cpu) ' s'])
    
end

%% Executando simulação estacionária tipo 2
if flagSS2

    disp('(6) Capturando de condicões iniciais')
    [X0] = dados2XY(dados,parametros);    
    disp('Condicões iniciais capturadas com sucesso')
    disp('  ')

    disp('(7) Determinando estado estacionario')
    options.Tol    = Tol;
    options.Delta  = Delta;
    options.Delta2 = Delta2;
    options.maxIter= maxIter;
    [X0]=solver_sea(options,@modelo_rede,0,X0,parametros);
    disp('Estado estacionario determinado com sucesso')
    disp('  ')
    
    disp('(8) Gerando saidas secudarias do estado estacionario')
    [dXdt,Y0,saidas,eixos]=modelo_rede(0,X0,parametros);
    disp('Saidas do estado estacionario organizadas com sucesso')
    disp('  ')

    disp('(9) Atualizando condições iniciais')
    dados = XY2dados(X0,Y0,dados,parametros);
    disp('Condições iniciais atualizadas com sucesso')
    disp('   ')
    
    T = 0;
    X = X0;
    
    estados.T = T;
    estados.X = X;

    disp('Tempo Real          : ESTACIONÁRIA')
    disp(['Tempo Transcorrido : ' num2str(toc) ' s'])
    disp(['Tempo de Máquina   : ' num2str(cputime-tempo_cpu) ' s'])
    
end

%% Executando simulação Dinâmica
if flagD
    disp('(10) Capturando de condicões iniciais')
    [X0] = dados2XY(dados,parametros);    
    disp('Condicões iniciais capturadas com sucesso')
    disp('  ')

    disp('(11) Iniciando integraçao do modelo')
    

    
    nx = length(parametros.IndX.tipo);
    nnMat = zeros(nx,1);
    nnMat(parametros.IndX.tipo==1)=1;
    nnMat(parametros.IndX.tipo==2)=1;
    nnMat(parametros.IndX.tipo==5)=1;

    opt = odeset('MaxStep',dados.simulador.maxStep,...
        'AbsTol',dados.simulador.absTol.*ones(nx,1),...
        'Vectorized','on',...
        'MaxOrder',5, ...
        'BDF','off',...
        'normcontrol','off','OutputFcn',@odeprogressbar,...
        'NonNegative',find(nnMat),...
        'RelTol',dados.simulador.relTol);
%       'Jacobian','modelo_rede_jacobiana');%

    
    T = dados.simulador.tempoInicial:dados.simulador.incrementoTempo:dados.simulador.tempoFinal;

    switch ode
        case 1
            [T,X] = ode45(@modelo_rede,T,X0,opt,parametros);
        case 2
            [T,X] = ode23(@modelo_rede,T,X0,opt,parametros);
        case 3
            [T,X] = ode113(@modelo_rede,T,X0,opt,parametros);
        case 4
            [T,X] = ode15s(@modelo_rede,T,X0,opt,parametros);
        case 5
            [T,X] = ode23s(@modelo_rede,T,X0,opt,parametros);
        case 6
            [T,X] = ode23t(@modelo_rede,T,X0,opt,parametros);
        case 7
            [T,X] = ode23tb(@modelo_rede,T,X0,opt,parametros);
    end
    T=T';    X=X';
    disp('Integraçao do modelo do modelo executada com sucesso')
    disp('  ')

    disp('(12) Atualizando condições iniciais')
    [dXdt,Y,saidas,eixos]=modelo_rede(T(end),X(:,end),parametros);
    
    dados = XY2dados(X(:,end),Y,dados,parametros);
    disp('Condições iniciais atualizadas com sucesso')
    disp('   ')
    
    estados.T = T;
    estados.X = X;
    disp(['Tempo Real        : ' num2str(T(end)-T(1))])
    disp(['Tempo Transcorrido: ' num2str(toc) ' s'])
    disp(['Tempo de Máquina  : ' num2str(cputime-tempo_cpu) ' s'])

end

if flagRES
    %% Tratamentos finais
    disp('(13) Organizando saidas do simulador dinamico')
    [dXdt,Y,saidas,eixos]=modelo_rede(estados.T,estados.X,parametros);
    disp('Saidas do simulador dinamico organizadas com sucesso.')
    disp('  ')
end


%% Subfunções
function dados = XY2dados(X,Y,dados,parametros)

Nk = parametros.Nk;
m = parametros.m;
NVertA = parametros.NVertA;
iVertA=parametros.iVertA;
Sxa = parametros.Sxa;

Ya=Sxa*X;
for i=1:NVertA
    dados.vertice{iVertA(i)}.condicaoInicial1 = Ya((i-1)*2+1);
    dados.vertice{iVertA(i)}.condicaoInicial2 = Ya((i-1)*2+2);
end
for i=1:m
    y=Y{i};
    ix = 0:4:4*Nk(i);
    dados.tubo{i}.condicaoInicial1=[];
    dados.tubo{i}.condicaoInicial2=[];
    dados.tubo{i}.condicaoInicial3=[];
    dados.tubo{i}.condicaoInicial4=[];
    dados.tubo{i}.condicaoInicial1(1,:)=y(ix+1);
    dados.tubo{i}.condicaoInicial2(1,:)=y(ix+2);
    dados.tubo{i}.condicaoInicial3(1,:)=y(ix+3);
    dados.tubo{i}.condicaoInicial4(1,:)=y(ix+4);
end


function    [X,Y] = dados2XY(dados,parametros)
    
m = parametros.m;
Nk = parametros.Nk;
NVertA = parametros.NVertA;
iVert = parametros.iVert;
Sw = parametros.Sw;
ne = parametros.ne;

iVertA=find(iVert==1);

W=[];
Wv=[];
for i=1:m
    Swk = Sw{i};
    y=[];
    auxy=0:ne:ne*Nk(i);
    y(auxy+1,:) = dados.tubo{i}.condicaoInicial1(:);
    y(auxy+2,:) = dados.tubo{i}.condicaoInicial2(:);
    y(auxy+3,:) = dados.tubo{i}.condicaoInicial3(:);
    y(auxy+4,:) = dados.tubo{i}.condicaoInicial4(:);
    W = [W;Swk'*y];
    Y{i} = y;
    
end
for i=1:NVertA
    Wv(2*(i-1)+1,1) = dados.vertice{iVertA(i)}.condicaoInicial1;
    Wv(2*(i-1)+2,1) = dados.vertice{iVertA(i)}.condicaoInicial2;
end
X = [W;Wv];