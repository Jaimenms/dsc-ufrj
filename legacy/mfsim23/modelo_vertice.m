function [dWvdt,saidas,eixos] = modelo_vertice(t,Y,Ya,E,parametros)
% Função que determina o modelo dos vertices dinamicos

% Desempacotando parametros
ne = parametros.ne;
V = parametros.V;
Nk = parametros.Nk;
iVertA = parametros.iVertA;
NVert = parametros.NVert;
NVertA = parametros.NVertA;
PR = parametros.PR;
m = parametros.m;

nt = size(Y{1},2);

% Calculando as vazoes nas extremidades dos dutos
QLs=zeros(NVert,nt);
QGs=zeros(NVert,nt);
QLe=zeros(NVert,nt);
QGe=zeros(NVert,nt);
for i=1:m
    y=Y{i};
    vIn=parametros.Duto{i}.vIn;
    vOut=parametros.Duto{i}.vOut;    
    QLs(vIn,:)=QLs(vIn,:)-y(3,:);
    QGs(vIn,:)=QGs(vIn,:)-y(4,:);
    QLe(vOut,:)=QLe(vOut,:)+y(Nk(i).*ne+3,:);
    QGe(vOut,:)=QGe(vOut,:)+y(Nk(i).*ne+4,:);
    saidas{vIn}.holdupL       = 1-y(1,:);
    saidas{vIn}.holdupG       = y(1,:);
    saidas{vIn}.pressao       = y(2,:).*PR./1e5;
    saidas{vIn}.vazaoMassaL   = QLs(vIn,:);
    saidas{vIn}.vazaoMassaG   = QGs(vIn,:);
    saidas{vOut}.holdupL      = 1-y(Nk(i).*ne+1,:);
    saidas{vOut}.holdupG      = y(Nk(i).*ne+1,:);
    saidas{vOut}.pressao      = y(Nk(i).*ne+2,:);
    saidas{vOut}.vazaoMassaL  = QLe(vOut,:);
    saidas{vOut}.vazaoMassaG  = QGe(vOut,:);
end

% Capturando as especificacoes dos vertices dinamicos
ALFA= E(1:4:end,:);
PT  = E(2:4:end,:);
QL  = E(3:4:end,:);
QG  = E(4:4:end,:);

% Gerando os balanços
dQL = QLe+QLs+QL;
dQG = QGe+QGs+QG;

% Calculando dinamica dos vertices tipo A
dWvdt=[];
for i=1:NVertA

    % Calculado pressao dos vertices
    PA(i,:) = PT(iVertA(i),:).*PR;

    % Calculando propriedades dos fluidos:
    [propL,propG,tensao] = modelo_fluido(PA(i,:),0,parametros.Fluido);
    densLA(i,:)=propL.dens;
    densGA(i,:)=propG.dens;
    ddensLdPA(i,:)=propL.C;
    ddensGdPA(i,:)=propG.C;
    
    % Capturando informações dos vertices A
    VA     = V (iVertA(i));    
    ALFAA(i,:)  = ALFA(iVertA(i),:);
    ALFALA(i,:) = 1-ALFAA(i,:);

    
    dQLA = dQL(iVertA(i),:);
    dQGA = dQG(iVertA(i),:);

    % Parametros do modelo dos vertices
    CVLA = ALFALA(i,:)./densLA(i,:).*ddensLdPA(i,:);
    CVGA = ALFAA(i,:) ./densGA(i,:).*ddensGdPA(i,:);
    CVTA = CVLA + CVGA;

    BVLA = dQLA./VA./densLA(i,:);
    BVGA = dQGA./VA./densGA(i,:);

    BVTA = BVLA + BVGA;

    dPTAdt = BVTA./CVTA./PR;
    dALFAAdt = BVGA-CVGA.*BVTA./CVTA;
    
    dWvdt(2*(i-1)+1,:) = dALFAAdt;
    dWvdt(2*(i-1)+2,:) = dPTAdt;
end
        
% Calculando saidas secundarias
for i=1:NVert
    eixos.X.valor{i}= t;
    eixos.estilo  = '2D';
    eixos.X.nome  = 'Tempo (s)';
    eixos.X.tipo  = 'T';        
end


%% Subfuncoes
function [F] = sigmoidal(x1,x2)
% Função que gera uma saída sigmoidal para eliminação de descontinuidades
% em chaveamento de modelos. A Função tem o seguintes comportamento:
% Se x1 = x2, então F = 0.5
% Se x1 > X2, então F tende a 1
% Se x1 < x2, então F tende a 0

% variável de entrada:
% x1, x2 -> coordenadas

% variável de saída:
% F -> resultado da função

% Parâmetros:
% fator Exp -> determina a curvatura da função
fatorExp = 2e6;

% Cálculo:
F= 1./(1+exp(-fatorExp.*(x1-x2)));

