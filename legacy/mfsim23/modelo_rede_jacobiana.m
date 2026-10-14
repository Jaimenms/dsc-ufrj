function [J]=modelo_rede_jacobiana(t,X,parametros)
% Função que gera Jacobiana do modelo de redes de escoamento bifásico

%% Gerando Matriz de X
DeltaRel = 1e-3;
DeltaAbs = 1e-8;
filtra=1;


Xantes = X;

nx = length(X);

X0 = X*ones(1,nx);
X2 = X0.*(ones(nx)+eye(nx)*DeltaRel)+ eye(nx)*DeltaAbs;
X1 = X0.*(ones(nx)-eye(nx)*DeltaRel)- eye(nx)*DeltaAbs;

[dXdt1]=modelo_rede(t,X1,parametros);
[dXdt2]=modelo_rede(t,X2,parametros);

J = (dXdt2-dXdt1)./(ones(nx,1)*(diag(X2-X1)'));