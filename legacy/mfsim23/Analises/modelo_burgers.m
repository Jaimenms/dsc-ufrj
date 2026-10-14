function [F,aux1,aux2,A,B,ixModel]=modelo_burgers(t,x,y,parametros,j)
% Função para o modelo de Burgers.

aux1=[];
aux2=[];
A = [1];
B = [0];
if nargin==0
    F=[];
else
    F=y;
end

ixModel = 1;