function [F,x,alfa]=modelo_rede_SS(t,x,parametros)
% Função referente ao modelo de redes de escoamento bifásico simplificado

% Desempacotando variaveis
M = parametros.M;
E = parametros.E;
e = parametros.e;
N = parametros.N;
S = parametros.S;
pr   = parametros.PR;
D    = parametros.D;
L    = parametros.L;
z    = parametros.Z;
rugo = parametros.rugo;
g    = 9.80665;

% Calculando solução simplificada
if isempty(x),
    x=laminar(parametros);
end
nx = size(x,2);

% Capturando variaveis
alfa=x(0*N+0*S+[1:S],:);
qL  =x(0*N+1*S+[1:S],:);
qG  =x(0*N+2*S+[1:S],:);
p   =x(0*N+3*S+[1:N],:);
wL  =x(1*N+3*S+[1:N],:);
wG  =x(2*N+3*S+[1:N],:);

% Determinando parametros
angulo = real(asin(M'*z./L));
Area = pi.*D.^2./4;

% Determinando termos fonte
fonteL=[];
fonteG=[];
for i=1:S
    % Montando parametros
    parametrosDuto = parametros.Duto{i};
    parametrosDuto.angulo_teta = [angulo(i) angulo(i)];
    parametrosDuto.x = [0 L(i)];
    parametrosDuto.rugo = rugo(i);
    parametrosDuto.Nk = 1;
    parametrosDuto.D = D(i);
    parametrosDuto.PR = pr;
    parametrosDuto.regime = parametros.regime(i);
    
    
    % Montando estados
    alfa1 = alfa(i,:);
    alfa2 = alfa(i,:);
    qL1 = qL(i,:);
    qL2 = qL(i,:);
    qG1 = qG(i,:);
    qG2 = qG(i,:);
    p1  = p(find(M(:,i)==-1),:);
    p2  = p(find(M(:,i)==+1),:);
    
    % Vetor de estados
    y = [alfa1;p1;qL1;qG1;...
        alfa2;p2;qL2;qG2];
    
    % Calculando modelo
    [F,saidas,FSS]=modelo_duto(0,[0;L(i)]*ones(1,size(y,2)),y,parametros.Duto,i);

    % Termos fonte
    fonteL=[fonteL;mean(FSS.L.*L(i),1)];
    fonteG=[fonteG;mean(FSS.G.*L(i),1)];

end

% Balanço de massa
F1 = M*qL+wL;
F2 = M*qG+wG;
% Balanço de momento
F3 = (-M'*p)-fonteL./pr;
F4 = (-M'*p)-fonteG./pr;
% Especificações
F5 = E*x-e*ones(1,nx);

% Vetor de Resíduos
F =[F1;F2;F3;F4;F5];

function x=laminar(parametros)
% LAMINAR   Calculo das variaveis da rede utilizando escoamento laminar e
% sem atrito entre os fluidos. A fração de área é utilizada apenas para
% determinar o diaâmetro hidráulico para cada fluido.
%   X = laminar     Esta funçao utiliza informaçoes contidas no arquivo de
%   dados da rede a ser estudada atraves de variaveis globais.

% Organização do vetor de resultados
% alfa-->x(        [1:S],:);
% qL  -->x(    1*S+[1:S],:);
% qG  -->x(    2*S+[1:S],:);
% p   -->x(    3*S+[1:N],:);
% wL  -->x(1*N+3*S+[1:N],:);
% wG  -->x(2*N+3*S+[1:N],:);

% Capturando parametros
M = parametros.M;
E = parametros.E;
e = parametros.e;
N = parametros.N;
S = parametros.S;
D    = parametros.D;
L    = parametros.L;
pr   = parametros.PR;
z    = parametros.Z;
g    = 9.80665;

% Capturando propriedades dos fluidos
viscL= parametros.Fluido.viscL;
viscG= parametros.Fluido.viscG;
densL= parametros.Fluido.densL0;
densG= parametros.Fluido.densG0;

% Gerando vetor de especificações
iy=1;
ealfa=zeros(0,1);
Ealfa=zeros(0,3*(N+S));
for i=1:S
    aux= find(E(:,i)>0);
    if isempty(aux)
        alfa0(i,1) = parametros.alfa0;
        ealfa(iy,1) = parametros.alfa0;
        Ealfa(iy,i) = 1;
        iy = iy+1;
    else
        alfa0(i,1) = e(aux,1);
    end
end

% Calculando diametro hidráulico a partir da fraçao de área
DG=sqrt(alfa0).*D;
DL=sqrt(1-alfa0).*D;

% Calculando coeficiente de atrito entre fases e parede
KL=diag(128.*viscL.*L./(DL.^4.*pi))./densL./pr;
KG=diag(128.*viscG.*L./(DG.^4.*pi))./densG./pr;

% Calculando termo fonte para balanço de momento
cL = (M'*z).*densL.*g./pr;
cG = (M'*z).*densG.*g./pr;

% Juntando vetor C do sistema linear M*x-C=0
C=[zeros(N,1); zeros(N,1); cL; cG; ealfa;e];

% Coeficientes da equação de balanço de massa
A1=[zeros(N,S) M zeros(N,S) zeros(N) eye(N) zeros(N)];
A2=[zeros(N,S) zeros(N,S) M zeros(N) zeros(N) eye(N)];

% Coeficientes da equação de balanço de momento
A3=[zeros(S) -KL zeros(S) -M' zeros(S,N) zeros(S,N)];
A4=[zeros(S) zeros(S) -KG -M' zeros(S,N) zeros(S,N)];

A=[A1;A2;A3;A4;Ealfa;E];

x=A\C;