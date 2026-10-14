function [ZL,ZG,saidas,eixos] = modelo_reservatorio(t,X,parametros)
% Função que determina as vazões de gas e de liquido em função de
% parametros do reservatorio

NRes = parametros.NRes;
ixP = parametros.reservatorio.ixP;
IPL = parametros.reservatorio.IPL;
IPG = parametros.reservatorio.IPG;
PRes = parametros.reservatorio.PRes;
PR = parametros.PR;

nt = size(X,2);
P = X(ixP,:);
dP = (PRes(:)*ones(1,nt)-P.*(PR./1e5));
ZL = diag(IPL)*dP;
ZG = diag(IPG)*dP;

ZL = max(0,ZL);
ZG = max(0,ZG);

saidas=[];
eixos =[];
for i=1:NRes
    saidas{i}.ZL = ZL(i,:);
    saidas{i}.ZG = ZG(i,:);
    saidas{i}.dP = dP(i,:);
    eixos.X.valor{i}= t;
    eixos.estilo  = '2D';
    eixos.X.nome  = 'Tempo (s)';
    eixos.X.tipo  = 'T';
end