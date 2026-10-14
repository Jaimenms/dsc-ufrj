function [F,aux1,aux2,A,B,ixModel]=modelo_agua(t,x,y,parametros,j)
% Função para o modelo de Águas Rasas.


F=[];
aux1=0;
aux2=0;
A = [1 2;3 4];
B = [0;0];

ixModel = [1;1];

if nargin>0

    Nk = size(y,1)./2-1;
    
    u = y(1:2:(2*Nk+1),:);
    h = y(2:2:(2*Nk+2),:);    
    g = 9.81;
    
    u = max(0,u);
    
    
    F((1-1)*(Nk+1) + (1:(Nk+1)),:)=abs(u);
%    F((1-1)*(Nk+1) + (1:(Nk+1)),:)=u;
    F((2-1)*(Nk+1) + (1:(Nk+1)),:)=g.*ones(size(u));
    F((3-1)*(Nk+1) + (1:(Nk+1)),:)=h;
    F((4-1)*(Nk+1) + (1:(Nk+1)),:)=u;

end
