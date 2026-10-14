function [T,X] = analisaAguaRasa
% Função que valida do metodo numerico atraves do problema de águas resas


%% 1-Especificações

% Comprimento
L=1;

% Ordem da quadratura
n = 12;

% numero de estados
ne = 2;

% Coeficientes das matrizes das EDPs
A = [1 2;...
     3 4];
B = [0;0];

% Numero de Elementos
Nk = 300;

% Serie de instantes de tempo
T  = [0:0.001:0.1];

% Coordenada da descontinuidade da condição inicial
xi=0.5;

% Altura a esquerda
hL=0.75;

% Altura a direita
hR=0.25;

tempo_inicial = 0.00;
ix_inicial = find(T==tempo_inicial);

%% 2-Calculos Iniciais


Swa = eye(2*(Nk+1));
Swa(:,[1 2 2*(Nk+1)-1 2*(Nk+1)])=[]; 
Sw{1}= Swa;

EB = zeros(2*(Nk+1),1);
EB(2,1) = hL;
EB(end,1) = hR;

% Vetor de tamanho de elementos
aga{1} = L(1)./Nk.*ones(Nk,1);

% Vetor de posições
xL = [0;cumsum(aga{1})];

% Determinando solução analítica
options.maxIter=100;
options.Tol=1e-6;
options.Delta=1e-6;
options.Delta2=1e-6;
parametros.hR = hR;
parametros.hL = hL;

auxSS = [0.1;0.5;5];
[auxSS]=solver_sea(options,@modelo_analitico,0,auxSS,parametros);
s = auxSS(1);
um = auxSS(2);
hm = auxSS(3);

g = 9.81;
X1SS = zeros(length(T),Nk+1);
for i=1:length(T)
    ti = T(i);
    for j=1:length(xL)
        xj = xL(j)-xi;
        if (xj./ti)<=(-(g.*hL).^0.5)
            X1SS(i,j)=0;
        elseif (xj./ti)>=s
            X1SS(i,j)=0;
        elseif (-(g.*hL).^0.5)<(xj./ti)&(xj./ti)<=(um-(g.*hm).^0.5)
            X1SS(i,j)=2./3.*((xj./ti)+(g.*hL).^0.5);
        else
            X1SS(i,j)=um;
        end
    end
end
X2SS = zeros(length(T),Nk+1);
for i=1:length(T)
    ti = T(i);
    for j=1:length(xL)
        xj = xL(j)-xi;
        if xj<=(-(g.*hL).^0.5).*ti
            X2SS(i,j)=hL;
        elseif xj>(-(g.*hL).^0.5).*ti&xj<=(um-(g.*hm).^0.5).*ti
            X2SS(i,j)=1./9./g.*(2.*(g.*hL).^0.5-(xj./ti)).^2;
        elseif xj>(um-(g.*hm).^0.5).*ti&xj<=s.*ti
            X2SS(i,j)=hm;
        elseif xj>=s.*ti
            X2SS(i,j)=hR;
        end
    end
end

% Vetor de condições iniciais
X01=X1SS(ix_inicial,:);
X02=X2SS(ix_inicial,:);
X0=zeros(2*(Nk+1),1);

X0(1:2:(2*Nk+1),:) = X01;
X0(2:2:(2*Nk+2),:) = X02;

X0 = Swa'*X0;

% Configuração do odesolver
opt = odeset('MaxStep',0.001,...
    'Vectorized','on',...
    'MaxOrder',5,...
    'RelTol',1e-3,...
    'AbsTol',1e-3);

% Integra Equações
TSS = T-tempo_inicial;
[T,X] = ode15s(@simula_agua,T,X0,opt,ne,A,B,Nk,n,aga,Sw,EB);

X = [Swa*X']'+ones(length(T),1)*EB';

nx = size(X,2)./2;
X1 = X(:,1:2:(2*Nk+1));
X2 = X(:,2:2:(2*Nk+2));


%% 3-Plota resultados

% Plota resultados
t_analise = 0.05;
ix = find(T>=t_analise);
ix=max(ix);


if isempty(ix)
    ix = length(T);
    ixSS = ix;
    ixSS = find(TSS>=T(ix));
    ixSS=max(ixSS);
else
    ixSS = find(TSS>=t_analise);
    ixSS=max(ixSS);
end

figure
plot(xL,X1SS(ixSS,:),'k-','linewidth',2)
hold on
plot(xL,X1(ix,:),'b-o','linewidth',2)
plot(xL,X1SS(ix_inicial,:)  ,'k-.','linewidth',2)
title(['Solução para Modelo Águas Rasas (tempo = ' num2str(T(ix))  ' ) usando ' num2str(Nk) ' células'],'Fontsize',14)
xlabel('Posição (m)','Fontsize',12)
ylabel('Velocidade (m/s)','Fontsize',12)
legend('Solução (Analítica)','Solução (Numérica)','Condição Inicial','Fontsize',12)
set(gca,'FontSize',12)

figure
plot(xL,X2SS(ixSS,:),'k-','linewidth',2)
hold on
plot(xL,X2(ix,:),'b-o','linewidth',2)
plot(xL,X2SS(ix_inicial,:)  ,'k-.','linewidth',2)
title(['Solução para Modelo Águas Rasas (tempo = ' num2str(T(ix))  ' ) usando ' num2str(Nk) ' células'],'Fontsize',14)
xlabel('Posição (m)','Fontsize',12)
ylabel('Altura (m)','Fontsize',12)
legend('Solução (Analítica)','Solução (Numérica)','Condição Inicial','Fontsize',12)
set(gca,'FontSize',12)

figure
subplot(2,1,1)
surf(TSS(:)*ones(1,length(xL)),ones(length(TSS),1)*(xL(:)'),X1SS)
title('Solução analitica para modelo aguas rasas','Fontsize',14)
xlabel('Tempo (s)','Fontsize',12)
ylabel('Posição (m)','Fontsize',12)
zlabel('Velocidade (m/s)','Fontsize',12)
shading interp
subplot(2,1,2)
surf(TSS(:)*ones(1,length(xL)),ones(length(TSS),1)*(xL(:)'),X2SS)
xlabel('Tempo (s)','Fontsize',12)
ylabel('Posição (m)','Fontsize',12)
zlabel('Altura (m)','Fontsize',12)
shading interp
set(gca,'FontSize',12)

figure
subplot(2,1,1)
surf(T(:)*ones(1,length(xL)),ones(length(T),1)*(xL(:)'),X1)
title('Solução numerica para modelo aguas rasas','Fontsize',14)
xlabel('Tempo (s)','Fontsize',12)
ylabel('Posição (m)','Fontsize',12)
zlabel('Velocidade (m/s)','Fontsize',12)
shading interp
subplot(2,1,2)
surf(T(:)*ones(1,length(xL)),ones(length(T),1)*(xL(:)'),X2)
xlabel('Tempo (s)','Fontsize',12)
ylabel('Posição (m)','Fontsize',12)
zlabel('Altura (m)','Fontsize',12)
shading interp
set(gca,'FontSize',12)



%% MODELO BURGERS
function dx = simula_agua(t,x,ne,A,B,Nk,n,h,Sw,EB)

parametros.Duto{1} = [];
parametros.Duto{1}.x = [];
parametros.Fluido = [];
parametros.Duto{1}.Sw = Sw{1}; 

Y{1}= Sw{1}*x + EB*ones(1,size(x,2));

modelo{1} = @modelo_agua;
[dx]=solver_edp_diferencas(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
%[dx]=solver_edp_diferencas_upwind(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
%[dx]=solver_edp_galerkin(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
%[dx]=solver_edp_petrovgalerkin(t,Y,Nk,n,h,Sw,modelo,parametros.Duto);
disp(t)


function dx = modelo_analitico(t,x,parametros)

s = x(1,:);
um = x(2,:);
hm = x(3,:);

hR = parametros.hR;
hL = parametros.hL;
g = 9.81;

dx1 = hm - 0.5.*hR.*((1+8.*s.^2./g./hR).^0.5-1);
dx2 = um - s + 0.25.*g.*hR./s.*((1+8.*s.^2./g./hR).^0.5+1);
dx3 = um + 2.*(g.*hm).^0.5 - 2.*(g.*hL).^0.5;

dx = [dx1;dx2;dx3];

function [X1,saidas,flag]=solver_sea(options,FUN,t,X1,parametros)
% Função que resolve um sist. alg. linear (Newton-Raphson).
global F2 X2

% Tol -> Tolerância do método (norma do vetor de residuos)
Tol=options.Tol;
% Numero máximo de iterações
maxIter=options.maxIter;

% Calculando valores iniciais
[F1]=feval(FUN,t,X1,parametros);

% Iniciando variáveis
cont=0; % contador
Res=norm(F1); % Resíduo
flag = 1; % Flag de convergência

% Iniciando painel
disp(['it.:     ' num2str(cont) '    norma :  ' num2str(norm(F1))])

status = odeprogressbar([0 1],[],'init');

% Executando laço de convergência
while Res>Tol&status==0

    if ~isreal(X1)
        disp('Parou devido à número complexo.')
        flag = 0;
        break
    end

    if cont==maxIter
        disp('Parou por atingir número máximo de iterações.')
        flag = 0;
        break
    end
    
    % Determinando jacobiana
    dF=jacobiana(options,F1,FUN,t,X1,parametros);

    % Busca unidimensional por passo ótimo
    passo=dF\F1;
    [fator_passo,Res] = fminbnd(@(fator_passo) veros(fator_passo,passo,FUN,t,X1,parametros),0,1);

    % Atualização dos estados
    X2=X1-fator_passo.*passo;

    % Cálculo do vetor de residuos
    [F2]=feval(FUN,t,X2,parametros);

    % Atualização das variaveis
    X1 = X2;
    F1 = F2;
    Res =norm(F1);
    cont=cont+1;    

    % Atualização do painel
    disp(['it.:     ' num2str(cont) '    norma :  ' num2str(norm(F1))])
    
    % Barra de Status
    lRes = -log(Res);
    lRes = max(0,lRes);
    lTol = -log(Tol);
    t = lRes./lTol;
    status = odeprogressbar(t,[],'on');
    
end

% Check final de convergência
if sum(sum(isnan(X1)))
    disp('Não Convergiu')
    flag = 0;
end

% Check final
if status==1
    disp('Parado pelo usuário')
end
odeprogressbar(t,[],'done');


%Empacotando saidas
saidas.cont=cont;
saidas.F=F1;
saidas.X=X1;
saidas.Res=Res;

function [Res] = veros(fator_passo,passo,FUN,t,X1,parametros)
% Função que faz a busca unidimensional pelo passo potimo
global X2 F2
X2=X1-fator_passo.*passo;   
[F2]=feval(FUN,t,X2,parametros);
Res =norm(F2);

function [dF]=jacobiana(options,F0,FUN,t,X1,parametros)
% Função para cálculo da matriz Jacobiana.

% Delta e Delta2 -> Parametros para calculo da Jacobiana numérica
Delta=options.Delta;
Delta2=options.Delta2;
% F0 -> vetor de resíduos no X0 médio da Jacobiana

dF=[];
for j=1:length(X1)
    X2=X1;
    X2(j)=(1+Delta)*X1(j)+Delta2;
    dF=[dF (feval(FUN,t,X2,parametros)-F0)./(X2(j)-X1(j))];
end


