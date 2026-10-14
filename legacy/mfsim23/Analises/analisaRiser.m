function [vLS,vGS,vLSb,vGSb]=analisaRiser
% Rotina que analisa riser.


%% 1- Entrada de dados

% Ponto Operacional
vGS_op =  0.25; % Velocidade operacional superficial do líquido (m/s)
vLS_op =  0.6; % Velocidade operacional superficial do líquido (m/s)

% Parametros da Linha de Produção
ETA         = 1.01/360.*2.*pi; % Angulo de inclinaçao da linha de produção (rad)
ZETA        = 90/360.*2.*pi; % Angulo de inclinaçao do riser (rad)
DP          = 25.4./1000.*6; % Diâmetro da linha de produção(m)
DR          = 25.4./1000.*4; % Diametro do Riser (m)
LP          = 4352; % Comprimento da linha de produção (m)
LR          = 620; % Altura do Riser (m)

% Parametros do Fluido
MG    = 0.016; % Peso Molecular (kg/gmol)
TG    = 5 + 273.15; % Temperatura (K)
ZG    = 0.7; % Fator de Compressibilidade (-)
PG    = 57.*101325; % Pressao na base do riser (Pa)
rohL  = 900; % Densidade do Líquido (kg/m³)
viscL = 8e-3; % Viscosidade dinamica do Líquido (Pa.s)
viscG = 5e-5; % Viscosidade dinamica do Gás (Pa.s)

% Constantes
np = 100;
R = 8.314; % Pa*m3/gmol.K
g = 9.81;  % m/s^2

% Parametros dos fatores de atrito
CL = 0.046;
CG = 0.046;
n = 0.2;
m = 0.2;

% Variar hd
hD     = linspace(0.001,0.999,np);


%% 2- Calculos Iniciais
% Calculando Densidade do Gas
rohG = PG*MG/(ZG*R*TG);

% Calculando geometria do escoamento
betah = 2.*acos(1-2.*hD);
alfa  = 1-(betah-sin(betah))./2./pi;
h = hD.*DP; % Altura de líquido


%% 3- Aplicando critério de Taitel para calcular a velocidade crítica do gás
[vGS] = taitel(hD,DP,betah,rohL,rohG,g,ETA);


%% 4 - Calculando velocidade da fase líquida segundo Lockhart-Martinelli (1949)
[vLS] = lockhart(vGS,hD,DP,ETA,viscL,viscG,rohL,rohG,CL,CG,n,m,g);


%% 5 - Aplicando critério de estabilidade de Boe

vLSb = vLS;
vGSb = vLSb.*(LP.*rohL.*g.*sin(ZETA)./PG).*(DP./DR).^2.*alfa;



%% Gerando Gráficos

figure
loglog(vGS,vLS,'b','linewidth',2)
hold on
loglog(vGSb,vLSb,'r','linewidth',2)
axis tight
xlabel('Velocidade Superficial Gás (m/s)')
ylabel('Velocidade Superficial Líquido (m/s)')
title('Estabilidade do Escoamento Estratificado (Taitel 1976)')


function [vLS, X, Y] = lockhart(vGS,hD,DP,ETA,viscL,viscG,rohL,rohG,CL,CG,n,m,g)

% Calculando viscosidades cinematicas
nuL = viscL./rohL;
nuG = viscG./rohG;

% Calculando informações geométricas
h = hD.*DP; % Altura de líquido
betah = 2.*acos(1-2.*hD);
alfaL = (betah-sin(betah))./2./pi;
alfa = 1-alfaL;
PerL = betah/2 * DP    ; %Líquido
PerG = pi*DP - PerL    ; %Gás
PerI = DP.*sin(betah/2); % Interface
Area  = (pi/4).*DP.^2; % Área da seçao
AreaG = Area*alfa;     % Área de gás na seção
AreaL = Area-AreaG;    % Área de líquido na seção

% Calculando os diâmetros Hidráulicos
DG = 4.*AreaG./(PerG+PerI);
DL = DG.*0;
ix = alfa<1;
DL(ix) = 4.*AreaL(ix)./PerL(ix);

% Reynolds
Re_GS = DG.*abs(vGS).*rohG./viscG;

% Fator de atrito superficial da fase gás
f_GS  = CG.*Re_GS.^(-m);

% Tensao no gas
tensao_GS = f_GS.*rohG.*(vGS).*abs(vGS)./2;

% Calcular Coeficiente Y (Lockhart-Martinelli)
Y = (rohL-rohG).*g.*sin(ETA)./(4./DP.*tensao_GS);

% Adimensionais
At  = pi./4;
ALt = (pi - acos(2.*hD-1) + (2.*hD-1).*(1-(2.*hD-1).^2).^0.5)./4;
AGt = (     acos(2.*hD-1) - (2.*hD-1).*(1-(2.*hD-1).^2).^0.5)./4;
SLt = (pi - acos(2.*hD-1));
SGt = (     acos(2.*hD-1));
SIt = (                                (1-(2.*hD-1).^2).^0.5);
uLt = At./ALt;
uGt = At./AGt;
DGt = 4.*AGt./(SGt + SIt);
DLt = 4.*ALt./(SLt      );

% Calcular FI e LAMBDA
auxG = ((DG./DP)./(alfa )).^(-m).*(alfa .^-2);
auxL = ((DL./DP)./(alfaL)).^(-n).*(alfaL.^-2);
FI   = (SGt./AGt+SIt./ALt+SIt./AGt).*(uGt.*DGt).^(-m).*uGt.^2;
LAMBDA = (uLt.*DLt).^(-n).*uLt.^2.*SLt./ALt;

% Calcular Coeficiente X (Lockhart-Martinelli)
X = ((4.*Y+FI)./LAMBDA).^0.5;

% Calcular velocidade do líquido
tensao_LS = X.^2.*tensao_GS;
vLS = (tensao_LS./CL).^(1/(2-n)).*(DP.*rohL./viscL).^(n/(2-n)).*(rohL./2).^(1/(n-2));


function [vGS] = taitel(hD,DP,betah,rohL,rohG,g,ETA)

% Calculando características geométricas do escoamento
alfaL = (betah-sin(betah))./2./pi;
alfa = 1-alfaL;
Area = pi.*DP.^2/4;
AreaG = Area.*alfa;     % Área de gás na seção
PerI = DP.*sin(betah/2); % Interface

vG = (1-hD).*(((rohL-rohG)./rohG).*g.*cos(ETA).*(AreaG./PerI)).^0.5;
vGS = alfa .* vG;



function [X2,Y2]=solver1(FUN,X1,parametros)
% Solver de sistema de equações algebricas

global F2 X2
% Tol -> Tolerância do método (norma do vetor de residuos)
Tol=1e-9;
% Numero máximo de iterações
maxIter=100;
% Calculando valores iniciais
[F0]=feval(FUN,X1,parametros);
% Partindo variaveis
cont=0;
flag=1;
Res=norm(F0);
Y2=0;
while Res>Tol
    if cont>=maxIter,flag=0;break,end
    dF=jacobiana(F0,FUN,X1,parametros);
    passo=dF\F0;
%    [fator,Res] = fminbnd(@(fator) veros(fator,passo,FUN,X1,parametros),0,1);
%    while fator>1e-3&(isnan(Res)|~isreal(F2))
%        fator = fator/2;
        fator = 1;
        X2=X1-fator.*passo;
        [F2,Y2]=feval(FUN,X2,parametros);
        Res =norm(F2);
    %end
    X1 = X2;
    F0 = F2;
    cont=cont+1;
end

function [Res] = veros(fator,passo,FUN,X1,parametros)
global X2 F2
X2=X1-fator.*passo;   
[F2]=feval(FUN,X2,parametros);
Res =norm(F2);

function [dF]=jacobiana(F0,FUN,X1,parametros)
% Função para cálculo da matriz Jacobiana.
Delta=1e-6;
Delta2=1e-9;
dF=[];
for j=1:length(X1)
    X2=X1;
    X2(j)=(1+Delta)*X1(j)+Delta2;
    dF=[dF (feval(FUN,X2,parametros)-F0)./(X2(j)-X1(j))];
end

