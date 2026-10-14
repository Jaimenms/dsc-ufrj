function [Sin,ls,lf,lu] = modelo_intermitente(entradas)
% Função que determina o fechamento do modelo para padrão de escoamento
% intermitente.

alfaG = entradas.alfaG;
u_LS = entradas.v_LS;
u_GS  = entradas.v_GS;
D  = entradas.D;
rugo = entradas.rugo;
angulo_teta = entradas.angulo_teta;
densL = entradas.densL;
densG = entradas.densG;
viscL = entradas.viscL;
viscG = entradas.viscG;
CL = entradas.CL;
CG = entradas.CG;
tensao = entradas.tensao;

% Calculando fração de área de líquido no slug
Rs = fracaoGasSlug(D,rugo,angulo_teta,tensao, densL, densG, viscL, u_LS, u_GS);
alfas = 1-Rs;

% Calculando o comprimento do slug
ls = compSlug(D);

% Determinando a frequencia dos slugs
wS = frequenciaSlugs(u_LS, u_GS, densL, densG, viscL, D, tensao,rugo);

% Calculando velocidade translacional
ut = velTrans(D,angulo_teta, u_LS, u_GS, densL, viscL);

% Calculando comprimentos das regioes
lu = ut./wS;
ls = min(ls,lu);
lf = lu - ls;
lf = max(lf,ls.*1e-3);
lu = lf+ls;

% Calculando holdup na bolha de taylor
alfaf = lu./lf.*alfaG-ls./lf.*alfas;
alfaf = max(1e-6,alfaf);
alfaf = min(1-1e-6,alfaf);
Rf = 1-alfaf;

% Calculando velocidades no slug
uL = (u_LS+ut.*(alfaG-alfas))./Rs;
ub = (u_GS-ut.*(alfaG-alfas))./alfas;

% Calculando velocidades na bolha de Taylor
uf = ut - (ut-uL).*Rs./Rf;
uG = ut - (ut-ub).*alfas./alfaf;

% Modelo Bolhas para o Slug
Area = pi.*D.^2./4;
entradas_Bol = entradas;
entradas_Bol.alfaG = 1-Rs;
entradas_Bol.alfaL = Rs;
entradas_Bol.v_L = uL;
entradas_Bol.v_LS= uL.*Rs;
entradas_Bol.v_G = ub;
entradas_Bol.v_GS = ub.*(1-Rs);
entradas_Bol.AreaL = Area.*Rs;
entradas_Bol.AreaG = Area.*(1-Rs);

[Ss] = modelo_bolhas (entradas_Bol);

% Modelo Estratificado para o Filme
entradas_Est = entradas;
entradas_Est.alfaG = 1-Rf;
entradas_Est.alfaL = Rf;
entradas_Est.v_L = uf;
entradas_Est.v_LS = uf.*Rf;
entradas_Est.v_G = uG;
entradas_Est.v_GS = uG.*(1-Rf);
entradas_Est.AreaL = Area.*Rf;
entradas_Est.AreaG = Area.*(1-Rf);
[hD,beta]=calcula_nivel_estratificado(1-Rf);
entradas_Est.beta = beta;
entradas_Est.hD = hD;
[Sf] = modelo_estratificado (entradas_Est);

Sin.GamaL      =(Ss.GamaL.*ls     +Sf.GamaL.*lf     )./lu;
Sin.GamaG      =(Ss.GamaG.*ls     +Sf.GamaG.*lf     )./lu;
Sin.dPL        =(Ss.dPL  .*ls     +Sf.dPL  .*lf     )./lu;
Sin.dPG        =(Ss.dPG  .*ls     +Sf.dPG  .*lf     )./lu;
Sin.ddPLdP     =(Ss.ddPLdP.*ls    +Sf.ddPLdP.*lf    )./lu;
Sin.ddPGdP     =(Ss.ddPGdP.*ls    +Sf.ddPGdP.*lf    )./lu;
Sin.ddPLdalfaG =(Ss.ddPLdalfaG.*ls+Sf.ddPLdalfaG.*lf)./lu;
Sin.ddPGdalfaG =(Ss.ddPGdalfaG.*ls+Sf.ddPGdalfaG.*lf)./lu;
Sin.dPI        =(Ss.dPI.*ls       +Sf.dPI.*lf       )./lu;
Sin.tensao_G   =(Ss.tensao_G.*ls  +Sf.tensao_G.*lf       )./lu;
Sin.tensao_L   =(Ss.tensao_L.*ls  +Sf.tensao_L.*lf       )./lu;
Sin.hL         =hD.*D;
Sin.beta       =beta;


function wS = frequenciaSlugs(u_LS, u_GS, densL, densG, viscL, D, tensao, rugo)

fL = fator(u_LS,densL,viscL,D,rugo);
Eo = sqrt(9.81.*D.^2.*(densL-densG)./tensao);
Nf = ((D.^1.5).*(densL.*(densL-densG).*9.81).^0.5)./viscL;
n = Nf./(260+0.85.*Nf);
wS = u_GS./D.*(0.0323./2.*u_LS./u_GS.*((densL./(densL-densG)).^0.5).*(fL.^-0.5)./(Eo.^0.2)).^n;


function Rs = fracaoGasSlug(D,rugo,angulo,tensao, densL, densG, viscL, uLS, uGS)
% Função que calcula a fração de área de líquido no slug

uS = uLS + uGS;
ReS = D.*uS.*densL./viscL;

tipo=1;

if tipo==1
    A=(2.457*log(1./(0.27*rugo./D+(7./ReS).^0.9))).^16;
    B=(37530./ReS).^16;
    fS=2.*((8./ReS).^12+1./(A+B).^1.5).^(1./12);
    alfas = 0.058.*(2.*(0.4.*tensao./9.81./(densL-densG)).^0.5.*(2.*fS./D.*uS.^3).^(2/5).*(densL./tensao).^(3/5)-0.725).^2;
    alfas = max(alfas,0);
    alfas = min(alfas,1-0.48);
    Rs = 1-alfas;
else
    angulo=max(0,angulo);
    angulo=min(pi/2,angulo);
    Rs = exp(-0.45.*angulo-2.48e-6.*ReS);
    alfas = 1-Rs;
end


function ut = velTrans(D,angulo, uLS, uGS, densL, viscL)
% Função que calcula a velocidade translacional

% D -> diametro do duto (m)
% angulo -> angulo de inclinação do duto (rad)
% uLS -> velocidade superficial do líquido (m/s)
% uGS -> velocidade superficial do gás (m/s)

% Dukler e Hubbard
ReM=D.*(uLS+uGS).*densL./viscL;
%C=(log10(ReM)+0.089)./(log10(ReM)-0.74);
%ix=find(C<2);
%C(ix)=2;
C = 1.2;


gD = (9.81.*D).^0.5;
ud = 0.35.*gD.*sin(angulo)+0.54.*gD.*cos(angulo); % Bendiksen 1984
ut = C.*(uLS+uGS)+ud;

function ls = compSlug(D)
% Função que calcula o comprimento do slug

% D -> diametros

ls = 18.1.*D; % Nicholson 1978
ix = find(D>0.038);
ls(ix) = exp(-3.287+4.589.*(log(D(ix)./0.0254)).^0.5);