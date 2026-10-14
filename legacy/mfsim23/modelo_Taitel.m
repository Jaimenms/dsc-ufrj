function [vSL,X,Y] = modelo_Taitel(hD,vSG,densL,densG,viscL,viscG,angulo,D,regL,regG)
% Modelo simplificado de Taitel para escoamento estratificado

% Parametros para calculo do fator de atrito
ixL1=find(regL==1);
ixL2=find(regL==2);
ixG1=find(regG==1);
ixG2=find(regG==2);

CL = zeros(size(regL));
CG = zeros(size(regG));
n = zeros(size(regL));
m = zeros(size(regG));

CL(ixL1) = 16;    % laminar
CL(ixL2) = 0.046; % turbulento
CG(ixG1) = 16;    % laminar
CG(ixG2) = 0.046; % turbulento
n(ixL1) = 1;      % laminar
n(ixL2) = 0.2;    % turbulento
m(ixL1) = 1;      % laminar
m(ixL2) = 0.2;    % turbulento

% Constantes
g = 9.81;

% Calculando velocidade da fase liquida dados hD e velocidade da fase gás
[vSL,X,Y] = modelo_Lockhart_Taitel(vSG,hD,D,-angulo,viscL,viscG,densL,densG,CL,CG,n,m,g);


function [vLS, X, Y] = modelo_Lockhart_Taitel(vGS,hD,DP,ETA,viscL,viscG,rohL,rohG,CL,CG,n,m,g)
% Sub-Função para determinação da velocidade superficial do líquido

% Calculando viscosidades cinematicas
nuL = viscL./rohL;
nuG = viscG./rohG;

% Calculando informações geométricas
betah = 2.*acos(1-2.*hD);
alfaL = (betah-sin(betah))./2./pi;
alfa = 1-alfaL;
PerL = betah./2 .* DP    ; %Líquido
PerG = pi*DP - PerL    ; %Gás
PerI = DP.*sin(betah/2); % Interface
Area  = (pi/4).*DP.^2; % Área da seçao
AreaG = Area.*alfa;     % Área de gás na seção
AreaL = Area-AreaG;    % Área de líquido na seção

% Reynolds
Re_GS = DP.*abs(vGS).*rohG./viscG;

% Fator de atrito superficial da fase gás
f_GS  = CG.*Re_GS.^(-m);

% Tensao no gas
tensao_GS = f_GS.*rohG.*(vGS).*abs(vGS)./2;

% Calcular Coeficiente Y (modelo_lockhart-Martinelli)
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
FI = (SGt./AGt+SIt./ALt+SIt./AGt).*(uGt.*DGt).^(-m).*uGt.^2;
LAMBDA = (uLt.*DLt).^(-n).*uLt.^2.*SLt./ALt;

% Calcular Coeficiente X (modelo_lockhart-Martinelli)
X2 = ((4.*Y+FI)./LAMBDA);

% Calcular velocidade do líquido
X2(find(X2<0))=nan;
X=sqrt(X2);
tensao_LS = X2.*tensao_GS;
vLS = (tensao_LS./CL).^(1/(2-n)).*(DP.*rohL./viscL).^(n/(2-n)).*(rohL./2).^(1/(n-2));