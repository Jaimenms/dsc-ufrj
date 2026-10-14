function [Ses] = modelo_estratificado(entradas)
% Função que determina o fechamento do modelo para padrão de escoamento
% estratificado.

%% Capturando variaveis de entrada
alfaG = entradas.alfaG;
alfaL = entradas.alfaL;
v_L = entradas.v_L;
v_LS = entradas.v_LS;
v_G = entradas.v_G;
v_GS = entradas.v_GS;
AreaL = entradas.AreaL;
AreaG = entradas.AreaG;
D = entradas.D;
rugo = entradas.rugo;
angulo_teta = entradas.angulo_teta;
densL = entradas.densL;
densG = entradas.densG;
viscL = entradas.viscL;
viscG = entradas.viscG;
CL = entradas.CL;
CG = entradas.CG;
tensao = entradas.tensao;
beta = entradas.beta;
hD = entradas.hD;


%% Calculando perímetros

% Calculando Perímetros
PerL = beta/2.*D;
PerG = pi*D-PerL;
PerI = D.*sin(beta/2);

%% Calculando diâmetros hidráulicos
ix = alfaG<1;
DG = 4.*AreaG./(PerG+PerI);
DL = DG.*0;
DL(ix) = 4.*AreaL(ix)./PerL(ix);


%% Calculando diferença de pressão das fases
    
% Calculando o centro de massa
zcml = D.*(0.5-1/3/pi./alfaL.*(sin(beta./2)).^3);
zcmg = D.*(0.5+1/3/pi./alfaG.*(sin(beta./2)).^3);        

% Calculando a diferença entre as fases
hL = hD.*D;
dPLG = ((densL-densG).*hL-densL.*zcml+densG.*zcmg).*9.81.*cos(angulo_teta);

% Calculando a diferença entre as fases e a pressão média
dPL =  alfaG.*dPLG;
dPG = -alfaL.*dPLG;

% Calculando a diferença entre a pressão na interface e a pressão média
dPI = (alfaL.*densL.*(zcml-hL)+alfaG.*densG.*(zcmg-hL)).*9.81.*cos(angulo_teta);

% Calculando as derivadas em relação a pressao
ddPLGdP = ((CL-CG).*hL-CL.*zcml+CG.*zcmg).*9.81.*cos(angulo_teta);
ddPLdP  =  alfaG.*ddPLGdP;
ddPGdP  = -alfaL.*ddPLGdP;

% Calculando derivadas intermediárias
dbetadalfaG = -2*pi./(1-cos(beta));
dhLdalfaG= 0.5.*sin(beta./2).*dbetadalfaG;
dzcmldalfaG= -D./(2.*pi.*alfaL).*(sin(beta./2)).^2.*...
    (2./3./alfaL.*sin(beta./2)+cos(beta./2).*dbetadalfaG);
dzcmgdalfaG= -D./(2.*pi.*alfaG).*(sin(beta./2)).^2.*...
    (2./3./alfaG.*sin(beta./2)-cos(beta./2).*dbetadalfaG);

% Calculando as derivadas em relaçao a fração de área
ddPLGdalfaG = ((densL-densG).*dhLdalfaG-densL.*dzcmldalfaG+densG.*dzcmgdalfaG).*9.81.*cos(angulo_teta);

ddPLdalfaG  = alfaG.*ddPLGdalfaG+dPLG;
ddPGdalfaG  =-alfaL.*ddPLGdalfaG+dPLG;

%% Calculando Fatores de Atrito

% Fluido-Parede
[f_L,Re_L]   =fator  (v_L,densL,viscL,DL,rugo);
[f_G,Re_G]   =fator  (v_G,densG,viscG,DG,rugo);

% Fluido-Interface
[f_I,f_L,f_G]=fator_I_Est(f_L,Re_L,v_L,densL,viscL,DL,...
    f_G,Re_G,v_GS,densG,viscG,DG,rugo,D,hL,alfaG);

%% Calculando Tensoes

% Fluido-Parede
tensao_G  = f_G.*densG.*(v_G).*abs(v_G)./2;
tensao_L  = f_L.*densL.*(v_L).*abs(v_L)./2;

% FLuido-Interface
tensao_IG = f_I.*densG.*(v_G-v_L).*abs(v_G-v_L)./2;
tensao_IL = -tensao_IG;

%% Calculando fluxo dissipativo
%PerI = max(PerI,1e-8);
%PerL = max(PerL,1e-8);
%PerG = max(PerG,1e-8);
GamaL = PerI.*tensao_IL+PerL.*tensao_L;
GamaG = PerI.*tensao_IG+PerG.*tensao_G;

%% Juntando saidas

Ses.GamaL      =GamaL;
Ses.GamaG      =GamaG;
Ses.dPL        =dPL;
Ses.dPG        =dPG;
Ses.ddPLdP     =ddPLdP;
Ses.ddPGdP     =ddPGdP;
Ses.ddPLdalfaG =ddPLdalfaG;
Ses.ddPGdalfaG =ddPGdalfaG;
Ses.dPI        =dPI;
Ses.hL         =hL;
Ses.beta       =beta;
Ses.tensao_G = tensao_G;
Ses.tensao_L = tensao_L;

function [f_I,f_L,f_G]=fator_I_Est(f_L,Re_L,v_L,densL,viscL,DL,...
    f_G,Re_G,v_GS,densG,viscG,DG,rugo,D,h,alfaG)
% Geraçao dos fator de atrito para a interface no escoamento estratificado

correlacao = 2;
switch correlacao
    case 1 % Para Escoamento Estratificado Horizontal
        % Taitel Y., & Duckler A. E., (1976) “A Model For Predicting Flow
        % Regime Transitions in Horizontal and Near-Horizontal Gas-Liquid
        % Flow”. AIChE Journal, 22 (1), 47-55.
        f_I = f_G;
    case 2 % Para Escoamento Estratificado Horizontal
        % Andritsos N., & Hanratty T. J., (1987) “Influence of interfacial
        % waves in stratified gas-liquid flows”. AIChE Journal, 33 (3),
        % 444-454.
        % Spedding P. L., & Hand N. P., (1997) “Prediction in Stratified
        % Gas-Liquid Co-Current Flow in Horizontal Pipelines”. Int. J.
        % Heat Mass Transfer, 40 (8), 1923-1935.
        v_GT = 5.5;
        % Andritsos N., & Hanratty T. J., (1987) “Influence of interfacial
        % waves in stratified gas-liquid flows”. AIChE Journal, 33 (3),
        % 444-454.
        f_I=f_G.*(1+max(0,15.*(abs(v_GS)./v_GT-1).*(h./D).^0.5));
end