function [ZL,ZG,saidas,eixos] = modelo_valvulaRet(t,X,E,parametros)
% Função que determina as vazões de líquido e de gás em função da pressão
% a montante da válvula, da pressão a jusante e do hold-up do vértice a
% montante.

% Iniciando variaveis
saidas=[];
eixos =[];
nt = length(t);

% Coletando parametros

% indices
ixP     = parametros.valvulaRet.ixP; % coleta pressões de vertices dinamicos
ixH     = parametros.valvulaRet.ixH; % coleta hold-up de vertices dinamicos
ixPext  = parametros.valvulaRet.ixPext; % coleta pressões externas de vertices dinamicos

% parametros
CV = parametros.valvulaRet.CV;
PR = parametros.PR;
NValRet = parametros.NValRet;
dPMin = parametros.valvulaRet.dPMin;

% Tratando entradas
P_Mon = X(ixP,:);
P_Jus = E(ixPext,:);
HG = X(ixH,:);
HL = 1-HG;

% Determinando diferença de pressão
dP = (P_Mon-P_Jus).*PR./1e5; % bar

P_Mean = (P_Mon+P_Jus)./2;

% Determinando densidade da mistura
[propL, propG] = modelo_fluido(P_Mean.*PR,0,parametros.Fluido);
densM = (HL.*propL.dens+HG.*propG.dens);

% Calculando vazões
qM = 0.0075.*diag(CV)*(sign(dP).*((densM.*abs(dP)).^0.5));
qL = HL.*qM;
qG = HG.*qM;

ZL = zeros(size(dP));
ZG = ZL;

% Verificando onde check está ativa
if NValRet
    ix = dP>(dPMin*ones(1,nt));
    % Determinando vazoes
    ZL(ix) = qL(ix);
    ZG(ix) = qG(ix);
end

for i=1:NValRet
    saidas{i}.ZP    = P_Mon(i,:).*PR./1e5;
    saidas{i}.dP    = dP(i,:);
    saidas{i}.qL    = ZL(i,:);
    saidas{i}.qG    = ZG(i,:);
    eixos.X.valor{i}= t;
    eixos.estilo  = '2D';
    eixos.X.nome  = 'Tempo (s)';
    eixos.X.tipo  = 'T';
end