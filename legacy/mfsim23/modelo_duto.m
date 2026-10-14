function [F,saidas,FSS,A,B,ixModel]=modelo_duto(t,x,y,parametros,j)
% Função que gera o vetor de coeficientes da matriz da EDP discretizada
% para o modelo bifásico a 4 equações.
    
% Preparando saídas
saidas=[];
FSS=[];
F=[];
ixModel = [];
if nargin==0
    % Matrizes de coeficientes
    A = [0 0 1 2;...
         0 0 3 4;...
         5 6 7 0;...
         8 9 0 10];
    B = [0;0;11;12];
    ixModel = [1;1;1;1];
    
else

%% Executando cálculos iniciais
    % ny -> 
    % ix -> vetor de ponteiros (1xNk)
    ne = 4; %numero de variaveis de estado (1x1)
    [nx,ny] = size(y); % ny (colunas do vetor de estados)
    ni = nx/ne; % numero de nohs
    nt = length(t); % numero de instantes de tempo
    ix = 1:ne:nx;

    t = ones(ny/nt,1)*t;
    t = ones(ni,1)*t(:)';
    
    g = 9.80665;

    parametrosDuto   = parametros{j};
    parametrosFluido = parametros{j}.Fluido;


%% Extraindo variaveis de entrada
    angulo_teta = interpola_angulo(parametrosDuto.x,parametrosDuto.angulo_teta,x);
    rugo        = parametrosDuto.rugo.*ones(ni,ny);
    D           = parametrosDuto.D.*ones(ni,ny);
    Nk          = parametrosDuto.Nk;
    PR          = parametrosDuto.PR;
    regime      = parametrosDuto.regime;
    
%% Capturando variáveis de estado

    % Desempacotando variaveis de estado:
    alfaG = y(ix,  :);
    PT    = y(ix+1,:);
    q_L   = y(ix+2,:);
    q_G   = y(ix+3,:);

    alfaG = max(1e-3,alfaG);
    alfaG = min(1-1e-3,alfaG);
    
    % Calculando nivel estratificado
    [hD,beta]=calcula_nivel_estratificado(alfaG);


%% Gerando estados modificados
    % alfaL -> fração de área da fase líquida
    % P     -> pressão absoluta em Pa
    % v_L   -> velocidade media do liquido
    % v_LS  -> velocidade superficial media do liquido
    % v_G   -> velocidade media do gas
    % v_GS  -> velocidade superficial media do gas
    % v_I   -> velocidade media da interface

    alfaL = 1-alfaG;
    P = PR.*PT;

    % Calculando propriedades dos fluidos:
    [propL,propG,tensao] = modelo_fluido(P,0,parametrosFluido);

    qVol_L = q_L./propL.dens;
    qVol_G = q_G./propG.dens;

    Area  = pi.*D.^2./4;
    AreaG = Area.*alfaG;
    AreaL = Area.*alfaL;

    v_LS  = q_L./(propL.dens.*Area);
    v_GS  = q_G./(propG.dens.*Area);

    v_L   = v_LS./alfaL;
    v_G   = v_GS./alfaG;
    
%% Montando Struct de variaveis de entrada
    entradas.alfaG   = alfaG;
    entradas.alfaL   = alfaL;
    entradas.v_L     = v_L;
    entradas.v_LS    = v_LS;
    entradas.v_G     = v_G;
    entradas.v_GS    = v_GS;
    entradas.AreaL   = AreaL;
    entradas.AreaG   = AreaG;
    entradas.D       = D;
    entradas.rugo    = rugo;
    entradas.angulo_teta = angulo_teta;
    entradas.densL   = propL.dens;
    entradas.densG   = propG.dens;
    entradas.viscL   = propL.visc;
    entradas.viscG   = propG.visc;
    entradas.CL      = propL.C;
    entradas.CG      = propG.C;
    entradas.tensao  = tensao;
    entradas.beta    = beta;
    entradas.hD      = hD;
    
%% IPE

    if regime==6 % Determinando o regime de escoamento
        [hD_est,hD_anu,alfaG_est,alfaG_anu,alfaG_bol]=determinaAlfaEstavel(entradas);
        entradas.alfaG_est = alfaG_est;
        entradas.alfaG_anu = alfaG_anu;
        entradas.alfaG_bol = alfaG_bol;
        entradas.hD_est    = hD_est;
                
        [fl, reg1, reg2] = identificaRegimeEstavel(entradas);

        reg1C = reg1;
    else % Fixando algum regime predeterminado
        zers = zeros(size(v_L));
        hD_est=zers;        hD_anu=zers;
        alfaG_est=zers;        alfaG_anu=zers;        alfaG_bol=zers;
        [fl, reg1, reg2] = desativaRegime(regime,zers);
        reg1C=zers;
    end
    
    
    
    
%% Modelos
    
    [entradas_mod,check]=filtra_entradas(entradas,fl);
    
    aux = zeros(0,1);
    Saux.GamaL      =aux;    Saux.GamaG      =aux;
    Saux.dPL        =aux;    Saux.dPG        =aux;
    Saux.ddPLdP     =aux;    Saux.ddPGdP     =aux;
    Saux.ddPLdalfaG =aux;    Saux.ddPGdalfaG =aux;
    Saux.dPI        =aux;    Saux.hL         =aux;
    Saux.beta       =aux;    Saux.tensao_G   =aux;
    Saux.tensao_L   =aux;
    
    % Modelo Estratificado
    Ses=Saux;if check.Est, [Ses] = modelo_estratificado (entradas_mod.Est);end
    % Modelo Bolhas
    Sbo=Saux;if check.Bub, [Sbo] = modelo_bolhas (entradas_mod.Bub);end
    % Modelo Anular
    San=Saux;if check.Anu, [San] = modelo_anular (entradas_mod.Anu);end
    % Modelo Intermitente
    Sin=Saux;if check.Int, [Sin] = modelo_intermitente(entradas_mod.Int);end

    % Montando struct de residuos
    [S] = montaRegime(fl,San,Ses,Sbo,Sin,check);

%% Coeficientes do modelo
    
    % Calculando a dissipação    
    FSS.L=(S.GamaL+g.*AreaL.*propL.dens.*sin(angulo_teta))./AreaL;
    FSS.G=(S.GamaG+g.*AreaG.*propG.dens.*sin(angulo_teta))./AreaG;
    
    % Deteminando o vetor de saida do modelo

    auxr= (alfaL./parametrosFluido.gamaL+alfaG./parametrosFluido.gamaG);
    rL  = (alfaL./parametrosFluido.gamaL)./auxr;
    rG  = (alfaG./parametrosFluido.gamaG)./auxr;
    rL2 = (1./parametrosFluido.gamaL)./auxr;
    rG2 = (1./parametrosFluido.gamaG)./auxr;

    F1 = -rG./(Area.*propL.dens);
    F2 =  rL./(Area.*propG.dens);
    F3 =  rL2./(Area.*PR.*propL.C);
    F4 =  rG2./(Area.*PR.*propG.C);
    F5 = -Area.*(S.dPL - alfaL.*S.ddPLdalfaG - (v_L.*abs(v_L)).*propL.dens);
    F6 =  PR.*AreaL.*(1-v_L.*abs(v_L).*propL.C+S.ddPLdP);
    F7 =  2.*abs(v_L);
    F8 =  Area.*(S.dPG + alfaG.*S.ddPGdalfaG - (v_G.*abs(v_G)).*propG.dens);
    F9 =  PR.*AreaG.*(1-v_G.*abs(v_G).*propG.C+S.ddPGdP);
    F10 = 2.*abs(v_G);
    F11 = S.GamaL+g.*AreaL.*propL.dens.*sin(angulo_teta);
    F12 = S.GamaG+g.*AreaG.*propG.dens.*sin(angulo_teta);

    F = [F1;F2;F3;F4;F5;F6;F7;F8;F9;F10;F11;F12];    

%% Calculando saídas
    if nargout>1
        
        saidas.inclinacao   = angulo_teta;
        saidas.diametro     = D;
        saidas.hD           = hD;
        saidas.alfaL        = 1-y(ix,:);
        saidas.alfaG        = y(ix,:);
        saidas.pressao      = y(ix+1,:).*PR./1e5;
        saidas.vazaoMassaL  = y(ix+2,:);
        saidas.vazaoMassaG  = y(ix+3,:);
        saidas.vazaoVolL    = y(ix+2,:)./propL.dens;
        saidas.vazaoVolG    = y(ix+3,:)./propG.dens;
        saidas.vazaoMassa   = y(ix+2,:)+y(ix+3,:);
        saidas.vazaoVol     = y(ix+3,:)./propG.dens + y(ix+2,:)./propL.dens;
        saidas.velocidadeL  = (y(ix+2,:)./propL.dens)./AreaL;
        saidas.velocidadeG  = (y(ix+3,:)./propG.dens)./AreaG;
        saidas.velocidadeSL = (y(ix+2,:)./propL.dens)./Area;
        saidas.velocidadeSG = (y(ix+3,:)./propG.dens)./Area;
        saidas.velocidadeI  = (y(ix+3,:)./propG.dens./AreaG)-(y(ix+2,:)./propL.dens./AreaL);
        saidas.velSomL      = (1./propL.C).^0.5;
        saidas.velSomG      = (1./propG.C).^0.5;
        saidas.velSomM      = ((alfaL./propL.dens+alfaG./propG.dens)./(alfaL.*propL.C./propL.dens+alfaG.*propG.C./propG.dens)).^0.5;
        saidas.dpresL       = S.dPL;
        saidas.dpresG       = S.dPG;
        saidas.cargaL       = FSS.L.*100/1e5;
        saidas.cargaG       = FSS.G.*100/1e5;
        saidas.densidadeL   = propL.dens;
        saidas.densidadeG   = propG.dens;
        saidas.viscosidadeL = propL.visc.*ones(nx./ne,size(y,2));
        saidas.viscosidadeG = propG.visc.*ones(nx./ne,size(y,2));
        saidas.reg1         = reg1;
        
        if regime~=6
            [hD_est,hD_anu,alfaG_est,alfaG_anu,alfaG_bol]=determinaAlfaEstavel(entradas);
            entradas.alfaG_est = alfaG_est;
            entradas.alfaG_anu = alfaG_anu;
            entradas.alfaG_bol = alfaG_bol;
            entradas.hD_est    = hD_est;
            [fl, reg1C] = identificaRegimeEstavel(entradas);
        end
        saidas.reg1C        = reg1C;
        saidas.reg2         = reg2;
        saidas.F1           = F1;
        saidas.F2           = F2;
        saidas.F3           = F3;
        saidas.F4           = F4;
        saidas.F5           = F5;
        saidas.F6           = F6;
        saidas.F7           = F7;
        saidas.F8           = F8;
        saidas.F9           = F9;
        saidas.F10          = F10;
        saidas.F11          = F11;
        saidas.F12          = F12;
        saidas.alfaG_est    = alfaG_est;
        saidas.alfaG_bol    = alfaG_bol;
        saidas.alfaG_anu    = alfaG_anu;

        [nx,ny]= size(saidas.pressao);
        for j=1:nx,
            for k=1:ny,
                A = [0 0 F1(j,k) F2(j,k);...
                     0 0 F3(j,k) F4(j,k);...
                     F5(j,k) F6(j,k) F7(j,k) 0;...
                     F8(j,k) F9(j,k) 0 F10(j,k)];

                A(isnan(A))=0;
                A(isinf(A))=0;
                                 
                E = eig(A,'nobalance');
                [aux,iEaux] = sort(real(E));

                Eaux=E(iEaux);            

                EIG_1s(j,k)=Eaux(1);
                EIG_2s(j,k)=Eaux(2);
                EIG_3s(j,k)=Eaux(3);
                EIG_4s(j,k)=Eaux(4);

            end
        end
        saidas.TipoEDP = 1+2.*((imag(EIG_1s)~=0)|(imag(EIG_2s)~=0)|(imag(EIG_3s)~=0)|(imag(EIG_4s)~=0))./2;
        saidas.Eig_1Real = real(EIG_1s);
        saidas.Eig_2Real = real(EIG_2s);
        saidas.Eig_3Real = real(EIG_3s);
        saidas.Eig_4Real = real(EIG_4s);
        saidas.Eig_1Imag = imag(EIG_1s);
        saidas.Eig_2Imag = imag(EIG_2s);
        saidas.Eig_3Imag = imag(EIG_3s);
        saidas.Eig_4Imag = imag(EIG_4s);
        
    end

end

function yi = interpola_angulo(x,y,xi)
[nxi1,nxi2] = size(xi);
yi = zeros(nxi1,nxi2);
nx = length(x);

for i=1:nxi1
    for j=1:nxi2
        ix2 = find(x==xi(i,j));
        if ~isempty(ix2)
            if ix2==1,      yi(i,j) = y(1);
            elseif ix2==nx, yi(i,j) = y(end);
            else            yi(i,j) = 0.5.*(y(ix2)+y(ix2-1));
            end
        else
            ix2 = find(x<=xi(i,j));
            yi(i,j) = y(ix2(end));
        end
    end
end


function    fl = regimeDiscreto(reg1)

reg1 = round(reg1);

zers = zeros(size(reg1));

fl.Anu=zers;
fl.Bub=zers;
fl.Int=zers;
fl.Est=zers;

fl.Anu(reg1==1)=1;
fl.Bub(reg1==2)=1;
fl.Int(reg1==3)=1;
fl.Est(reg1==4)=1;
