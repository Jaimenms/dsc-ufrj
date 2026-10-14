function [dXdt,Y,saidas,eixos]=modelo_rede(t,X,parametros)
% Função referente ao modelo de redes de escoamento bifásico

%% Calculos Iniciais
m     = parametros.m;
nx = size(X,2);
nt = length(t); 

if nx>1&&nt==1, t = t.*ones(1,nx); end


%% Aplicando filtro para impedir refluxos - Parte 1
tolq = 1e-6;
ixCheck = parametros.ixCheck;
for i=1:length(ixCheck), X(ixCheck(i),X(ixCheck(i),:)<tolq) = tolq; end

%% Separando variaveis de estado de vértices do tipo A
Sxa = parametros.Sxa;
Ya = Sxa*X;

%% Determinando especificações simples

% Especificações dos vertices
tE = parametros.tE;
Matriz_E = parametros.Matriz_E;
E=interpola(t,tE,Matriz_E);


%% Determinando especificaçoes especiais

% ESPECIFICAÇÕES RELACIONADAS A PRESSÕES

% Especificações do tipo conexão (cruzando pressões)
% obs.: substituindo especificação de pressão no final do tubo a montante
% por pressão no começo do tubo a jusante
Sc1  = parametros.Sc1;
Sc2  = parametros.Sc2;
E = Sc1*X + Sc2*E;

% Especificações de valvula
% obs.: substituindo especificação de pressão no final do tubo a montante
% por pressão a montante da valvula
[ZP, saidasVal,eixosVal] = modelo_valvula(t,X,E,parametros);
SVal1 = parametros.valvula.SVal1;
SVal2 = parametros.valvula.SVal2;
E = SVal1*ZP + SVal2*E;


% ESPECIFICAÇÕES RELACIONADAS A VAZÕES

% Especificações do reservatório
% obs.: substituindo especificação de vazão no começo do tubo a jusante por
% vazoes geradas pelo reservatorio
[ZL,ZG,saidasRes,eixosRes] = modelo_reservatorio(t,X,parametros);
SRes1 = parametros.reservatorio.SRes1;
SRes2 = parametros.reservatorio.SRes2;
SRes3 = parametros.reservatorio.SRes3;
E = SRes1*ZL+SRes2*ZG+SRes3*E;

% Especificações de valvula Retenção
% obs.: substituindo especificação de vazão no vertice dinamico pelas
% vazoes geradas pelo modelo de valvula de retenção a partir da pressão
% externa do vertice A
[ZL,ZG, saidasValRet,eixosValRet] = modelo_valvulaRet(t,X,E,parametros);
SValRet1 = parametros.valvulaRet.SValRet1;
SValRet2 = parametros.valvulaRet.SValRet2;
SValRet3 = parametros.valvulaRet.SValRet3;
E = SValRet1*(-ZL) + SValRet2*(-ZG) + SValRet3*E;

% Especificações do tipo conexão (cruzando vazões)
% obs.: substituindo especificação da vazão no começodo tubo a jusante pela
% vazão no vertice a montante.
Sc3  = parametros.Sc3;
Sc4  = parametros.Sc4;
E = Sc3*X + Sc4*E;

% Capturando estados dos vertices dinamicos
% obs.:substitui pressão e hold-up obtidos dos estados dos vertices
% dinamicos sobre o vetor de especificações. Valores de pressão no vetor de
% especificação deixam de ser de pressão externa e passam a ser de pressão
% no vertice.
SVert1 = parametros.SVert1;
SVert2 = parametros.SVert2;
E = SVert1*X + SVert2*E;

%% Dutos
% Calculando o vetor de estados reduzidos dos dutos
SY  = parametros.SY;
Yr = [X;E];
for k=1:m
    Y{k}=SY{k}*Yr;
end

% Calculando o vetor de acúmulo dos estados reduzidos dos dutos
modelo=parametros.modelo;
metodo=parametros.metodo;
Nk  = parametros.Nk;
n   = parametros.n;
h   = parametros.h;
Sw  = parametros.Sw;
switch metodo
    case 1, [dWp]=solver_edp_diferencas    (t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
    case 2, [dWp]=solver_edp_galerkin      (t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
    case 3, [dWp]=solver_edp_petrovgalerkin(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
    case 4, [dWp]=solver_edp_diferencas_upwind(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
end        

%%  Vertices Dinamicos
% Calculando derivada dos estados dos vértices
[dWa,saidasVert,eixosVert] = modelo_vertice(t,Y,Ya,E,parametros);


%% Rede
% Gerando derivada do vetor de estados da rede
dXdt=[dWp;dWa];

% Aplicando filtro para impedir refluxos  - Parte 2
for i=1:length(ixCheck)
    ixNEG    = find(X(ixCheck(i),:)<tolq);
    ixNEGNEG = find(X(ixCheck(i),:)<tolq&dXdt(ixCheck(i),:)<0);
    dXdt(ixCheck(i),ixNEGNEG) = 0;
end


%% Calculando Saídas
if ~isempty(saidasRes),    saidas.reservatorio = saidasRes;  eixos.reservatorio = eixosRes;  end
if ~isempty(saidasVal),    saidas.valvula = saidasVal;       eixos.valvula = eixosVal;       end
if ~isempty(saidasValRet), saidas.valvulaRet = saidasValRet; eixos.valvulaRet = eixosValRet; end
if ~isempty(saidasVert),   saidas.vertice = saidasVert;      eixos.vertice = eixosVert;      end

if nargout > 2
    % Calculando saidas dos dutos
    for i=1:m
        [F{i},saidas.tubo{i}]=feval(modelo{i},t,parametros.Duto{i}.x*ones(1,size(Y{i},2)),Y{i},parametros.Duto,i);
        hk=parametros.Duto{i}.h;
        eixos.tubo.X.valor{i}   = cumsum([0;hk(:)]);
        eixos.tubo.Y.valor{i}   = t;
        eixos.tubo.estilo  = '3D';
        eixos.tubo.X.nome  = 'Posição (m)';
        eixos.tubo.X.tipo  = 'L';
        eixos.tubo.Y.nome  = 'Tempo (s)';
        eixos.tubo.Y.tipo  = 'T';
    end
end