function [San] = modelo_anular(entradas)
% Função que determina o fechamento do modelo para padrão de escoamento
% anular.

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

%% Calculando altura de líquido e angulo de molhamento

% Calculando rigorosamente hD
hL=D./2.*(1-alfaG.^0.5);
beta = 2.*pi;

%% Calculando perímetros

% Calculando Perímetros
PerG = 0;
PerL = pi*D;
PerI = pi*D.*sqrt(alfaG);

%% Calculando diâmetros hidráulicos
DG = 4.*AreaG./(PerG+PerI);
DL = 4.*AreaL./PerL;

%% Calculando diferença de pressão das fases
dPL        = 0.;
dPG        = 0.;
ddPLdP     = 0.;
ddPGdP     = 0.;
ddPLdalfaG = 0.;
ddPGdalfaG = 0.;
dPI        = 0.;

%% Calculando Fatores de Atrito

% Fluido-Parede
[f_L,Re_L]   =fator  (v_L,densL,viscL,DL,rugo);
[f_G,Re_G]   =fator  (v_G,densG,viscG,DG,rugo);

% Fluido-Interface
[f_I,f_L,f_G]=fator_I_An(f_L,Re_L,v_L,densL,viscL,DL,...
    f_G,Re_G,v_GS,densG,viscG,DG,rugo,D,hL,alfaG);

%% Calculando Tensoes

% Fluido-Parede
tensao_G  = f_G.*densG.*(v_G).*abs(v_G)./2;
tensao_L  = f_L.*densL.*(v_L).*abs(v_L)./2;

% FLuido-Interface
tensao_IG = f_I.*densG.*(v_G-v_L).*abs(v_G-v_L)./2;
tensao_IL = -tensao_IG;

%% Calculando fluxo dissipativo
PerI = max(PerI,1e-8);
PerL = max(PerL,1e-8);
PerG = max(PerG,1e-8);
GamaL = PerI.*tensao_IL+PerL.*tensao_L;
GamaG = PerI.*tensao_IG+PerG.*tensao_G;

%% Juntando saidas

San.GamaL      =GamaL;
San.GamaG      =GamaG;
San.dPL        =dPL;
San.dPG        =dPG;
San.ddPLdP     =ddPLdP;
San.ddPGdP     =ddPGdP;
San.ddPLdalfaG =ddPLdalfaG;
San.ddPGdalfaG =ddPGdalfaG;
San.dPI        =dPI;
San.hL         =hL;
San.beta       =beta;
San.tensao_G = tensao_G;
San.tensao_L = tensao_L;


function [f_I,f_L,f_G]=fator_I_An(f_L,Re_L,v_L,densL,viscL,DL,...
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
        % Kowalski J. E., (1987) “Wall and Interfacial Shear Stress in
        % Stratified Flow in a Horizontal Pipe”. AIChE Journal, 33 (2),
        % 274-281.
        f_L = max(24./Re_L,0.263./(1-alfaG).*(DL./Re_L./D).^0.5);
        % Andritsos N., & Hanratty T. J., (1987) “Influence of interfacial
        % waves in stratified gas-liquid flows”. AIChE Journal, 33 (3),
        % 444-454.
        f_I=f_G.*(1+max(0,15.*(abs(v_GS)./v_GT-1).*(h./D).^0.5));
        
    case 3 % Para Escoamento Anular Vertical
        % Wongwises, S., Kongkiatwanitch, W., INTERFACIAL FRICTION FACTOR IN
        % VERTICAL UPWARD GAS-LIQUID ANNULAR TWO-PHASE FLOW, Int. Comm. Heat Mass
        % Transfer, v.28, n.3, pp.323-336, 2001.
        Re_SG=4*abs(q_G).*alfaG.^0.5./viscG./pi./D;
        ix = find(Re_SG==0);
        a=17.172;
        b=-0.768;
        c=-0.253;
        f_I=4.*a.*(Re_SG.^b).*(h.^c);
        f_I(ix)=0;
end