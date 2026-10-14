function [fl,reg1,reg2,fatores,fatoresC] = identificaRegime(alfa,q_L,q_G,parametros)
% Função que executa a IPE e gera as variáveis identificadoras para fraçõe
% de áreas fixadas.
%    [FL] = identificaRegime(ALFA,QL,QG,PARAMETROS), para ALFA, QL e QG
%    matrizes M-por-N correspondentes ao hold-up, vazão mássica de líquido
%    em kg/s e vazão mássica de gás em kg/s, respectivamente e o struct de
%    parametros contendo as seguintes informações:
%      PARAMETROS.angulo_teta -> matriz M-por-N de angulos de inclinaçao em
%      radiano, onde 0 corresponde a duto horizontal
%      PARAMETROS.D           -> diametro do duto (m)
%      PARAMETROS.rugo        -> rugosidade do duto (m)
%      PARAMETROS.h           -> altura de líquido para regime de
%      escoamento estratificado(m)
%      PARAMETROS.viscL       -> viscosidade dinâmica do líquido (Pa.s)
%      PARAMETROS.viscG       -> viscosidade dinâmica do gás (Pa.s)
%      PARAMETROS.densL       -> densidade do líquido (kg/m³)
%      PARAMETROS.densG       -> densidade do gás (kg/m³)
%      PARAMETROS.AreaL       -> Área da seção transversal com líquido (m²)
%      PARAMETROS.AreaG       -> Área da seção transversal com gás (m²)
%      PARAMETROS.tensao      -> Tensao interfacial (Pa.m)
%    Onde a saída FL corresponde a um struct de chaveadores de regime:
%      fl.Est -> matriz M-po-N onde valor 1 indica regime estratificado
%      fl.EsW -> matriz M-po-N onde valor 1 indica regime estratificado
%      com ondas
%      fl.EsS -> matriz M-po-N onde valor 1 indica regime estratificado
%      sem ondas
%      fl.Anu -> matriz M-po-N onde valor 1 indica regime anular
%      fl.Bub -> matriz M-po-N onde valor 1 indica regime bolhas
%      fl.Int -> matriz M-po-N onde valor 1 indica regime intermitente
%
%    [FL,REG1,REG2] = identificaRegime(ALFA,QL,QG,PARAMETROS), para as
%    mesmas entradas gerando as matrizes REG1 e REG2 de ponteiros
%    indicativa do padrao de escoamento existente.
%
%    Os ponteiros de REG1 podem assumir os seguintes valores:
%      Anular ou anular disperso (1)
%      Bolhas ou bolhas dispersas (2)
%      Intermitente (3)
%      Estratificado (4)
%
%    Os ponteiros de REG2 podem assumir os seguintes valores:
%      Anular ou anular disperso (1)
%      Bolhas ou bolhas dispersas (2)
%      Intermitente (3)
%      Estratificado suave (4)
%      Estratificado ondulado (5)
%
%    [FL,REG1,REG2,FATORES] = identificaRegime(ALFA,QL,QG,PARAMETROS), para 
%    mesmas entradas gerando o struct FATORES informa os valores
%    das variáveis identificadoras de padrão de escoamento:
%      FL - numero de Froude do líquido
%      FG - numero de Froude do gás
%      NL - numero de anularidade
%      K - numero de interação onda vento
%      T - numero de interação bolha-intermitente
%      FGcosteta - teste estratificado
%    Segundo as seguinte utilizações para mudanças de regime:
%  	   Stratified to non-stratified (FG/(cos(teta)^0.5), Kelvin-Helmholtz)
%      Smooth to wavy (K - onda de vento, FL - onda de gravidade)
%      Annular to stratified (NL)
%      Bubble to no-bubble (T)
%      Annular to bubble/intermitent (FG)
%      Intermitent to dispersed bubble (alfaG, dmax)
%
%    O seguinte algoritmo foi utilizado
%
%          ESTRATIFICADO
%                |
%                |
%                V
%              Existe?__________
%                |              |
%               Não            Sim
%                |              |
%                |              |
%                V              V
%         ANULAR DISPERSO   Teste SMOOTH ou WAVY______
%                |              |                     |
%                |              |                     |
%                V              V                     V
%      _______Existe?       ESTRATIFICADO        ESTRATIFICADO
%     |          |             SMOOTH!!!             WAVY!!!         
%     |          |
%    Sim        Não_____________________
%     |                                 |
%     V                                 V
%  Teste de   ______                 BUBBLE
%  entrainment      |                   |
%     |             |                   |
%     V             V                   |
%  ANULAR!!!      ANULAR                V
%                DISPERSO!!!         Existe?_________
%                                       |            |
%                                       |            |
%                                      Sim           |
%                                       |            |
%                                       V            V
%                                    BUBBLE!!!   INTERMITENTE!!!

%% (0) Fator de transição: determina a curvatura da função que chaveia os
% regimes. Valores elevados promovem transição instantânea entre os regimes
% de escoamento.
fatorExp = 2e6;

%% (1) Determinando parâmetros de entrada
angulo_teta = parametros.angulo_teta;
D           = parametros.D;
rugo        = parametros.rugo;
hL          = parametros.h;
viscL       = parametros.viscL;
viscG       = parametros.viscG;
densL       = parametros.densL;
densG       = parametros.densG;
AreaL       = parametros.AreaL;
AreaG       = parametros.AreaG;
tensao      = parametros.tensao;
betah       = parametros.betah;

% Calculando velocidades superficiais
vS_L = gerav(q_L,densL,AreaL+AreaG);
vS_G = gerav(q_G,densG,AreaL+AreaG);

% Calculando velocidades reais
v_L = gerav(q_L,densL,AreaL);
v_G = gerav(q_G,densG,AreaG);

% Calculando valores médios
densM = (1-alfa).*densL+alfa.*densG;
v_M = gerav(q_L+q_G,densM,AreaL+AreaG);

% Calculando fração de área de líquido
alfaL = 1-alfa;

% Gerando flag que identifica se fluidos estão ou não em sentido opostos
flag_SentidoOposto = (sign(v_L.*v_G)+1)/2;

% Fatores de atrito de fanning segundo velocidades superficiais
[fS_L,ReS_L]=fator(vS_L,densL,viscL,D,rugo);
[fS_G,ReS_G]=fator(vS_G,densG,viscG,D,rugo);

% Fatores de atrito de fanning segundo velocidades reais
[f_L,Re_L]=fator(v_L,densL,viscL,D,rugo);
[f_G,Re_G]=fator(v_G,densG,viscG,D,rugo);

% Fator de atrito médio
f_M = (1-alfa).*f_L+alfa.*f_G;

%% (2) Calculando parametros

% Calculando grupos adimensionais (FG e FL)
[FL, FG] = geraFroude(densL, densG, D, vS_L, vS_G);

% Calculando FG/(cos(teta))^0.5
FGast = FG./((cos(angulo_teta)).^0.5);

%    Calculando NL, K e T
NL = (fS_L.*FL.^2./cos(angulo_teta)).^0.5;
K  = FG.*(ReS_L./cos(angulo_teta)).^0.5;
T  = (2.*fS_L.*FL.^2./cos(angulo_teta)).^0.5;

EOD = D.*((densL-densG).*9.81./tensao).^0.5;

% Calculando fatores criticos
FG_c = (75./(EOD.^2.*((1+ (EOD./75).^2./f_L).^0.5-1))).^0.5;
FGast_c      = (((1-hL./D).^2).*(alfa.^3).*(pi.*D./4)./(D.*sin(betah./2))).^0.5;
K_c          = 2.*alfaL.^0.5.*alfa./(0.01.^0.5);
FL_c         = 1.5.*(hL./D).^0.5.*alfaL;
NL_c         = 1.0.*(1-hL./D).^0.5.*alfaL./(f_L./fS_L).^0.5;
T_c          = (2.*pi.*(alfaL.^2.*alfa)./sin(betah./2)./(f_L./fS_L)).^0.5;


%% (3) Assumindo-se estratificado, determinando relação hL/D (VÁ PARA ITEM

%% (4) Testando instabilidade Kelvin-Helmholtz
%    Para o caso de FG/(cos(teta))^0.5 maior que crítico -> (VÁ ITEM 8)
%    Para o caso de FG/(cos(teta))^0.5 menor ou igual ao crítico -> (VÁ PARA ITEM 5)

[flag_naoest1] = sigmoidal(fatorExp,FGast,FGast_c);% 0 - se menor e 1 se maior

%% (5) Verificando sentido
%    Fluido subindo (VÁ PARA ITEM 6)
%    Fluido descendo (VÁ PARA ITEM 5.1)
[flag_sobe]=(sign(cos(angulo_teta).*(q_L+q_G))+1)./2; % 0 se desce e 1 - se sobe
flag_desce = 1-flag_sobe;

%% (5.1) Testando a trajetória do filme líquido
%    Para o caso de NL maior que crítico -> bubble, intermitente, anular ou anular disperso (VÁ ITEM 8)
%    Para o caso de NL menor ou igual ao crítico -> estratificado smooth ou wavy (VÁ ITEM 7)
[flag_naoest2] = sigmoidal(fatorExp,NL,NL_c).*flag_desce;% 0 - se menor e 1 se maior

%% (6) Testando iterações de Jeffreus (vento-onda) para escoamento
% estratificado comondas
%    Para o caso de K maior que crítico -> estratificado wavy (FIM)
%    Para o caso de K menor ou igual a crítico -> estratificado smooth (FIM)
[flag_wavy1] = sigmoidal(fatorExp,K,K_c);% 0 - se menor e 1 se maior

%% (7) Testando ondas de gravidade
%    Para o caso de FL maior que crítico e fluido descendo -> estratificado wavy (FIM)
%    Para o caso contrário -> estratificao smooth (FIM)
[flag_wavy2] = sigmoidal(fatorExp,FL,FL_c);% 0 - se menor e 1 se maior

%% (8) (Vem do 5.1) Assumindo-se anular#####################################

%% (9) Testando bridging do líquido acima de abaixo de alfaL = 0.35

%    Se alfaL>0.35 -> bubble ou intermitente (VÁ PARA ITEM 11)
flag_naoanular = sigmoidal(fatorExp,alfaL,0.35);

%    Se teta=90º -> bubble, intermitente, anular ou anular disperso (VÁ PARA ITEM 10)
flag_vertical = (abs(angulo_teta)-90)<(10/180*pi); 

%    Caso contrário -> anular ou anular disperso (FIM)

%% (10) (Vem do 9) Testando estabilidade do filme para escoamento vertical
%    Para o caso de FG menor que crítico -> bubble ou intermitente (VÁ PARA ITEM 11)
%    Para o caso de FG maior ou igual a crítico -> anular ou anular disperso (FIM)
flag_naohorizontal = (abs(angulo_teta))>(10/180*pi); 
flag_anular1 = sigmoidal(fatorExp,FG,FG_c)+flag_naohorizontal;% 0 - se menor e 1 se maior
flag_anular1(find(flag_anular1==2))=1; 

%% (11) Testando bubble ou intermitente###################################

%    Para o caso de |teta|<10º (VÁ PARA ITEM 12)
flag_horizontal = abs(angulo_teta)<(10/180*pi);
%    Se alfaG>0.52 -> intermitente (FIM)
flag_intermitente1 = sigmoidal(fatorExp,alfa,0.52);
%    Se alfaG<0.25 -> dispersed bubble (FIM)
flag_bubble1 = sigmoidal(fatorExp,0.25,alfa);
%    Se 0.25<alfaG<0.52 -> determinar tamanhos das bolhas (ver equações)
dmax = (0.725+4.15.*(vS_G./v_L).^0.5).*(tensao./densL).^(3/5).*(2.*f_M.*v_M.^3./D).^(-2/5);
ddef = 2.*(0.4.*tensao./(densL-densG)./9.81).^0.5;
dmigr = 3./8.*densL./(densL-densG).*f_M.*v_M.^2./9.81./cos(angulo_teta);
flag_checkbolhas = (1-flag_bubble1).*(1-flag_intermitente1);
%    Se dmax<ddef e dmax< dmigr -> dispersed bubble (FIM)
flag_bubble2 = sigmoidal(fatorExp,ddef,dmax).*sigmoidal(fatorExp,dmigr,dmax).*flag_checkbolhas;
%    Se dmax>ddef ou dmax< dmigr -> intermitente (FIM)


%% (12) Testando empuxo vs. flutuações turbulentas

%    Para o caso de T > crítico -> bubble (FIM)
%    Para o caso de T < crítico -> intermitente (FIM)
[flag_bubble3] = sigmoidal(fatorExp,T,T_c);% 0 - se menor e 1 se maior

% Chaveadores principais
fl.Est = (1-flag_naoest1).*flag_sobe + (1-flag_naoest2).*(1-flag_sobe);

fl.EsW = fl.Est.*(flag_wavy1.*flag_sobe+flag_wavy2.*(1-flag_sobe));

fl.EsS = fl.Est.*(1-fl.EsW);

fl.Anu = (1-fl.Est).*...
         (1-flag_naoanular).*...
         (flag_vertical.*flag_anular1);

fl.Bub = (1-fl.Est).*...
         (1-fl.Anu).*...
         (flag_bubble3.*flag_horizontal+(flag_bubble1+flag_bubble2).*(1-flag_horizontal));

fl.Int = (1-fl.Est).*...
         (1-fl.Anu).*...
         (1-fl.Bub);

 % Armazendo Fatores
fatores.FL = FL;
fatores.FG = FG;
fatores.NL = NL;
fatores.K  = K;
fatores.T  = T;
fatores.FGcosteta = FGast;

 % Armazendo Fatores Críticos
fatoresC.FL = FL_c;
fatoresC.NL = NL_c;
fatoresC.K  = K_c;
fatoresC.T  = T_c;

% Gerando marcadores de regime
reg1 = fl.Anu.*1+fl.Bub.*2+fl.Int.*3+fl.Est.*4;
reg2 = fl.Anu.*1+fl.Bub.*2+fl.Int.*3+fl.EsS.*4+fl.EsW.*5;

function v = gerav(q,dens,Area)
% Sub-Função para cálculo da velocidade

% Variáveis de entrada:
% dens -> Densidade (kg/m³)
% q -> vazão mássica (kg/s)
% Area -> área da seção transversal do duto (m²)

v = q./dens./Area;

function [FL, FG] = geraFroude(densL, densG, D, vS_L, vS_G)
% Sub-Função para cálculo dos números adimensionais de Froude

% Variáveis de entrada:
% densL -> Densidade do Líquido (kg/m³)
% densG -> Densidade do Gás (kg/m³)
% D -> diâmetro (m)
% vS_L -> velocidade superficial* do líquido
% vS_G -> velocidade superficial* do gás
%
% * velocidade superficial é definida como a vazão volumétrica dividida
% pela área total da seção transversal do tubo

% Calculando FL (número de Froud da fase líquida)

FL = (densL./(densL-densG)./9.8./D).^0.5.*abs(vS_L);
% Calculando FG (número de Froud da fase gás)

FG = (densG./(densL-densG)./9.8./D).^0.5.*abs(vS_G);

function [F] = sigmoidal(fatorExp,x1,x2)
% Função que gera uma saída sigmoidal para eliminação de descontinuidades
% em chaveamento de modelos. A Função tem o seguintes comportamento:
% Se x1 = x2, então F = 0.5
% Se x1 > X2, então F tende a 1
% Se x1 < x2, então F tende a 0

% variável de entrada:
% x1, x2 -> coordenadas

% variável de saída:
% F -> resultado da função

% Parâmetros:

% Cálculo:
F= 1./(1+exp(-fatorExp.*(x1-x2)));


function [f,Re]=fator(v,dens,visc,D,rugo)
% Geraçao dos fatores de atrito de fanning para as fases, onde:
% tensao_G  = -FI_G.*(f_G.*densG.*(v_G).^2./2).
% Sendo o fator de atrito de fanning (f) corresponde a 1/4 do fator de
% atrito de Darcy.
% Lembrando:
% Em regime laminar, f(Fanning) = 16/Re; f(Darcy) = 64/Re;

inx=find(v==0);
v(inx)=1e-6;
Re = D.*abs(v).*dens./visc;

correlacao = 3;
switch correlacao
    case 1 % Churchill (1977)
        % Churchill, S. W. Friction-Factor Equation Spans All Fluid-Flow
        % Regimes. Chemical Engineering, v. 84, n. 24, 0. 91-92, 1977.
        A=(2.457*log(1./(0.27*rugo./D+(7./Re).^0.9))).^16;
        B=(37530./Re).^16;
        f=2.*((8./Re).^12+1./(A+B).^1.5).^(1./12);
    case 2 % Eck (1973)
        % Eck B., (1973) “Technische Stromungslehre”, Springer,
        % New York, 1973.
        f=0.0625./(log10(15./Re+rugo./D./3.715)).^2;
    case 3 % Haaland (1983)
        % Haaland S. E., (1983) “Simple and Explicit Formulas for the
        % Friction Factor in Turbulent Pipe Flow”. Journal of Fluids
        % Engineering, 105, 89-90.
        f=0.07716./(log10(6.9./Re+(rugo./D./3.7).^1.11)).^2;
    case 4 % Colebrook (1939)
        % Colebrook C. F., (1939) “Turbulent Flow in Pipes, With
        % Particular Reference to the Transition between the Smooth and
        % Rough Pipe Laws”. J. Inst. Civ. Eng., 11, 133-156,London, 1939.
        for i=1:size(Re,1)
            for j=1:size(Re,2)
                f(i,j)=fzero(@(x) 1+2*x.^0.5.*log10(2.51./Re(i,j)./...
                    (x.^0.5)+rugo./3.715./D), 64/Re(i,j));
            end
        end
        f=f./4; % de Darcy para Fanning
        
    case 5 % Moody (1947)
        % Moody L. F., (1947) “An Approximate Formula for Pipe Friction
        % Factors”. Transactions ASME, 69, 1005.
        f=0.001375.*(1+(1E6.*(0.02.*rugo./D+1./Re)).^(1/3));
end
f(inx)=0;
Re(inx)=0;

