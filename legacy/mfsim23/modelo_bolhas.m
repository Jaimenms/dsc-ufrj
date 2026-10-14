function [Ses,saidas] = modelo_bolhas(entradas)
% Função que determina o fechamento do modelo para padrão de escoamento
% bolhas.

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

% Calculando h
beta = 2.*pi;
hL   = D;

%% Calculando diametro das bolhas

v_M   = v_GS + v_LS;

f_L =fator(v_L,densL,viscL,D,rugo);
f_M =fator(v_M,densL,viscL,D,rugo);
f_G =0.0;

dbmax = (0.725+4.15.*(v_GS./v_M).^0.5).*(tensao./densL).^(3/5).*(2.*f_M.*v_M.^3./D).^(-2/5);

db = (16.*tensao./9.81./(densL-densG)).^0.5;

db = min(db,dbmax);

vb = (0.51.*db+2.14./db).^0.5;


%% Calculando perímetros

% Calculando Perímetros
PerL = pi.*D.*alfaL;
PerG = pi.*D.*alfaG;
PerI = pi.*D.^2.*alfaG./db;


%% Calculando diferença de pressão das fases
    
% Drew D. A., & Passman S. L., (1999) “Theory of Multicomponent
% Fluids”. Applied Mathematical Sciences, 135, Springer.
dPL = 1/4.*densL.*(v_G-v_L).^2;
dPG = 0;

ddPLdP    = dPL./densL.*CL;
ddPGdP    = 0.;
ddPLdalfaG= 0.;
ddPGdalfaG= 0.;
dPI       = 0.;


%% Calculando Fatores de Atrito


% Fluido-Interface

%Reb = db.*abs(vb).*densL./viscL;
%Eo = 9.81.*(densL-densG).*db.^2./tensao;
%CD = max(24./Reb.*(1+0.15.*Reb.^0.687),8/3.*Eo./(Eo+4));

CDb = 4/3.*((densL-densG)./densL).*9.81.*db./(vb.^2);

f_I = 1.5.*alfaG.*densL.*CDb./db;
%f_I = max(1e-9,f_I);

%% Calculando Tensoes

% Fluido-Parede
tensao_G  = f_G.*densG.*(v_G).*abs(v_G)./2;
tensao_L  = f_L.*densL.*(v_L).*abs(v_L)./2;

% FLuido-Interface
tensao_IG = f_I.*densG.*(v_G-v_L).*abs(v_G-v_L)./2;
tensao_IL = -tensao_IG;

%% Calculando fluxo dissipativo
PerI = max(PerI,1e-9);
PerL = max(PerL,1e-9);
PerG = max(PerG,1e-9);
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
Ses.tensao_G   =tensao_G;
Ses.tensao_L   =tensao_L;

saidas.db = db; % Diametro das Bolhas



function [hD,beta]=alfa2h(alfaG,beta)
% Determinando altura de líquido /diâmetro para escoamento estratificado de
% forma rigorosa

Tol=1e-9;
maxIter=100;
[nx,ny] = size(alfaG);

for i=1:nx
    for j=1:ny
        alfaLi = (1-alfaG(i,j));
        if alfaLi>(1-1e-6) % Caso apenas Líquido
            betai=2*pi;
        elseif alfaLi<(1e-6) % Caso apenas Gás
            betai = 0;
        else % Caso intermediário
            % Preparando método
            cont=0;
            erro=1;
            betai=beta(i,j);
            % Executando Newton-Raphson
            while erro>Tol
                if cont>=maxIter;break,end
                F0=sin(betai)-betai+2*pi*alfaLi;
                dF=cos(betai)-1;
                passo = F0/dF;
                betai = betai - passo;
                erro  = abs(passo);
                cont=cont+1;
            end
        end
        beta(i,j) = betai;
    end
end

% Calculando hD a partir do angulo de molhamento
hD=1/2*(1-cos(beta/2));


function [hD,beta]=alfa2h2(alfaG)
% Determinando altura de líquido /diâmetro para escoamento estratificado de
% forma aproximada

% Polinomios
P_h=[0.00000000000197;...
       -3.66048527474641;...
       9.15121318685793;...
       -8.89104962298284;...
       4.18536124761860;...
       -1.75567780293538;...
       0.98531913309307]';

P_beta=1.0e+002.*...
    [0.00000000000056;...
     -0.67451330764088;...
     1.68628326909991;...
     -1.58733054088785;...
     0.69471254223251;...
     -0.17280444315654;...
     0.05824216671205]';

% Calculando hD
hD=polyval(P_h,alfaG);

% Calculando beta
beta=polyval(P_beta,alfaG);

% Tratando insonsistências
beta = max(0,beta);
beta = min(2.*pi,beta);


function [f_I,f_L,f_G]=fator_I_Est(f_L,Re_L,v_L,densL,viscL,DL,...
    f_G,Re_G,v_GS,densG,viscG,DG,rugo,D,h,alfaG)
% Geraçao dos fator de atrito para a interface no escoamento estratificado

correlacao = 2;
DG = mac(DG,1e-8);
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
end