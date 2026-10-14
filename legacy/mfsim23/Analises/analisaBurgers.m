function [T,X] = analisaBurgers
% Função que valida do metodo numerico atraves do problema de Burgers

clear all
%aux=cd;
%aux=aux(1:length(aux)-9);
%addpath(aux)


%% 1-Especificações

% Comprimento
L=2;

% Ordem da quadratura
n = 5;

% numero de estados
ne = 1;

% Coeficientes das matrizes das EDPs
A = [1];
B = [0];

% Numero de Elementos
Nk = 100;

% Serie de instantes de tempo
%T  = [0:0.01:1.5];

T  = [0:0.01:1.5];

% Fator para atenuar descontinuidade da condição inicial
fatorExp = 20*1e3;

% Coordenada da descontinuidade da condição inicial
xi=0.2;

% Velocidade a esquerda
uL=1;

% Velocidade a direita
uR=0.5;


%% 2-Calculos Iniciais

% Matriz seletora de estados reduzidos
Sw{1}=eye(Nk+1);

% Vetor de tamanho de elementos
aga{1} = L(1)./Nk.*ones(Nk,1);

% Vetor de posições
xL = [0;cumsum(aga{1})];

% Vetor de condições iniciais
X0= uL-(uL-uR)./(1+exp(-fatorExp.*(xL-xi)));

% Configuração do odesolver
opt = odeset('MaxStep',0.01,...
    'RelTol',1e-6,...
    'AbsTol',1e-6,...
    'Vectorized','on',...
    'MaxOrder',5, ...
    'BDF','off');

% Integra Equações
[T,X] = ode15s(@simula_burgers,T,X0,opt,ne,A,B,Nk,n,aga,Sw);
%[T,X] = ode45(@simula_burgers,T,X0,opt,ne,A,B,Nk,n,aga,Sw);

X2 = zeros(size(X));
for i=1:length(T)
    t = T(i);
    S = 0.5.*(uL+uR);
    if uL>uR
        ix = find((xL-xi)<(S*t));
        X2(i,ix)=uL;
        ix = find((xL-xi)>=(S*t));
        X2(i,ix)=uR;
    else
        X2(i,:)=(xL-xi)./t;
        ix = find((xL-xi)<(uL*t));
        X2(i,ix)=uL;
        ix = find((xL-xi)>=(uR*t));
        X2(i,ix)=uR;
    end
end


%% 3-Plota resultados

% Plota resultados
figure
surf(T(:)*ones(1,length(xL)),ones(length(T),1)*(xL(:)'),X)
title('Avaliação do Galerkin para Modelo Burgers')
xlabel('Tempo (s)')
ylabel('Posição (m)')
zlabel('Velocidade (m/s)')
shading interp

figure
surf(T(:)*ones(1,length(xL)),ones(length(T),1)*(xL(:)'),X2)
title('Solução para Modelo Burgers')
xlabel('Tempo (s)')
ylabel('Posição (m)')
zlabel('Velocidade (m/s)')
shading interp

figure
plot(xL,X2(end,:),'k-','linewidth',2)
hold on
plot(xL,X(end,:),'b-o','linewidth',2)
hold on
plot(xL,X2(1,:),'k-.')
title(['Solução para Modelo Burgers (tempo = ' num2str(T(end))  ' ) usando ' num2str(Nk) ' células'])
xlabel('Posição (m)')
ylabel('Velocidade (m/s)')
legend('Solução (Analítica)','Solução (Numérica)','Condição Inicial')



%% MODELO BURGERS

function dx = simula_burgers(t,x,ne,A,B,Nk,n,h,Sw)

parametros.Duto{1} = [];
parametros.Duto{1}.x = [];


Y{1}= x;
modelo{1} = @modelo_burgers;
%[dx]=solver_edp_galerkin      (t,Y,ne,Nk,n,h,Sw,@modelo_burgers,parametros.Duto);
%[dx]=solver_edp_petrovgalerkin(t,Y,ne,Nk,n,h,Sw,@modelo_burgers,parametros.Duto);
%[dx]=solver_edp_diferencas    (t,Y,ne,Nk,n,h,Sw,@modelo_burgers,parametros.Duto);
[dx]=solver_edp_diferencas_upwind    (t,Y,Nk,n,h,Sw,modelo,parametros.Duto);

disp(t)