function [fl,reg1,reg2,fatores,fatoresC] = identificaRegimeEstavel(entradas)
% Função que executa a IPE e gera as variáveis identificadoras para fraçõe
% de áreas calculadas pelos modelos de Taitel, Pots e Zuber.
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
fatorExp = 1e6;

%% (1) Abrindo entradas

% Frações de área
alfaG_est = entradas.alfaG_est;
alfaG_anu = entradas.alfaG_anu;
alfaG_bol = entradas.alfaG_bol;
hD_est    = entradas.hD_est;

% Estados
v_LS  = entradas.v_LS;
v_GS  = entradas.v_GS;
v_M = v_LS+v_GS;

%Parametros Duto
D     = entradas.D;
rugo  = entradas.rugo;
angulo_teta = entradas.angulo_teta;

% Parametros Fluido
densL = entradas.densL;
densG = entradas.densG;
viscL = entradas.viscL;
viscG = entradas.viscG;
tensao = entradas.tensao;

% Calculando fração de área de líquido
alfaL_est = 1-alfaG_est;
alfaL_anu = 1-alfaG_anu;
alfaL_bol = 1-alfaG_bol;

% Gerando flag que identifica sentidos
flag_sobe          =(sign(cos(angulo_teta).*(v_M+1e-10))+1)./2; % 0 se desce e 1 - se sobe
flag_desce         = 1-flag_sobe;
flag_horizontal    =  abs(angulo_teta)    <(10/180*pi);
flag_vertical      = abs(abs(angulo_teta)-90)<(10/180*pi);%.*flag_sobe; 
flag_naovertical   = 1-flag_vertical; 

% Fatores de atrito de fanning segundo velocidades superficiais
[fS_L,ReS_L]=fator(v_LS,densL,viscL,D,rugo);

% Calculando grupos adimensionais (FG e FL)
[FL, FG] = geraFroude(densL, densG, D, v_LS, v_GS);

%% (2) Analisando estabilidades

% Calculando fatores criticos

% Estabilidade Estratificado - Teste de Kelvin-Helmholtz
FGast = FG./((cos(angulo_teta)).^0.5);
beta = 2.*acos(1-2.*hD_est);
FGast_c      = (((1-hD_est).^2).*(alfaG_est.^3).*(pi.*D./4)./(D.*sin(beta./2))).^0.5;
%    Para o caso de FG/(cos(teta))^0.5 maior que crítico
%    Para o caso de FG/(cos(teta))^0.5 menor ou igual ao crítico
%    0 - se menor e 1 se maior
[flag_naoest1] = sigmoidal(fatorExp,FGast,FGast_c);

% Estabilidade Estratificado - Teste de Trajetória
NL = (fS_L.*FL.^2./cos(angulo_teta)).^0.5;
f_L_est=fator(v_LS./alfaL_est,densL,viscL,D,rugo);
NL_c         = 1.0.*(1-hD_est).^0.5.*alfaL_est./(f_L_est./fS_L).^0.5;
%    Para o caso de NL maior que crítico -> bubble, intermitente, anular ou anular disperso (VÁ ITEM 8)
%    Para o caso de NL menor ou igual ao crítico -> estratificado smooth ou wavy (VÁ ITEM 7)
[flag_naoest2] = sigmoidal(fatorExp,NL,NL_c).*flag_desce;% 0 - se menor e 1 se maior

% Estabilidade Estratificado - Teste de Jeffreys
K  = FG.*(ReS_L./cos(angulo_teta)).^0.5;
K_c          = 2.*alfaL_est.^0.5.*alfaG_est./(0.01.^0.5);
%    Para o caso de K maior que crítico -> estratificado wavy (FIM)
%    Para o caso de K menor ou igual a crítico -> estratificado smooth (FIM)
[flag_wavy1] = sigmoidal(fatorExp,K,K_c);% 0 - se menor e 1 se maior

% Estabilidade Estratificado - Teste de ondas de gravidade
FL_c         = 1.5.*(hD_est).^0.5.*alfaL_est;
%    Para o caso de FL maior que crítico e fluido descendo -> estratificado wavy (FIM)
%    Para o caso contrário -> estratificao smooth (FIM)
[flag_wavy2] = sigmoidal(fatorExp,FL,FL_c);% 0 - se menor e 1 se maior

% Estabilidade Anular - Teste de filme
EOD = D.*((densL-densG).*9.81./tensao).^0.5;
FG_c = (75./(EOD.^2).*((1+ (EOD./75).^2./0.0065).^0.5-1)).^0.5;

%    Para o caso de FG menor que crítico -> bubble ou intermitente (VÁ PARA ITEM 11)
%    Para o caso de FG maior ou igual a crítico -> anular ou anular disperso (FIM)
flag_naoanular = sigmoidal(fatorExp,alfaL_anu,0.35);
flag_anular = sigmoidal(fatorExp,FG,FG_c);

% Estabilidade Bolhas - Teste de turbulência
T  = (2.*fS_L.*FL.^2./cos(angulo_teta)).^0.5;
betahEst = 2.*acos(1-2.*hD_est);
T_c  = (2.*pi.*(alfaL_est.^2.*alfaG_est)./sin(betahEst./2)./(f_L_est./fS_L)).^0.5;

%    Para o caso de T > crítico -> bubble (FIM)
%    Para o caso de T < crítico -> intermitente (FIM)
[flag_bubble3] = sigmoidal(fatorExp,T,T_c);% 0 - se menor e 1 se maior


% Demais teste para Estabilidade Bolhas
densM = alfaL_bol.*densL+alfaG_bol.*densG;
f_M = fator(v_M,densM,viscL,D,rugo);
dmax = (0.725+4.15.*(abs(v_GS./v_M)).^0.5).*(tensao./densL).^(3/5).*(abs(2.*f_M.*v_M.^3./D)).^(-2/5);
ddef = 2.*(0.4.*tensao./abs(densL-densG)./9.81).^0.5;
dmigr = (3/8).*densL./abs(densL-densG).*f_M.*v_M.^2./9.81./cos(angulo_teta);

flag_intermitente1 = sigmoidal(fatorExp,alfaG_bol,0.52);
flag_bubble1       = sigmoidal(fatorExp,0.25,alfaG_bol);
flag_checkbolhas   = (1-flag_bubble1).*(1-flag_intermitente1);
flag_bubble2       = sigmoidal(fatorExp,ddef,dmax).*sigmoidal(fatorExp,dmigr,dmax).*flag_checkbolhas;

% Chaveadores principais
fl.Est = (1-flag_naoest1).*flag_sobe + (1-flag_naoest2).*(1-flag_sobe);
fl.EsW = fl.Est.*(flag_wavy1.*flag_sobe+flag_wavy2.*(1-flag_sobe));
fl.EsS = fl.Est.*(1-fl.EsW);


fl.Anu = (1-fl.Est).*(1-flag_naoanular).*...
         (flag_naovertical+flag_anular.*flag_vertical);
fl.Bub = (1-fl.Est).*...
         (1-fl.Anu).*(...
         flag_bubble1.*(1-flag_horizontal)+...
         flag_bubble2.*(1-flag_horizontal).*(1-flag_intermitente1).*(1-flag_bubble1)+...
         flag_bubble3.*   flag_horizontal);
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

function [FL, FG] = geraFroude(densL, densG, D, v_LS, v_GS)
% Sub-Função para cálculo dos números adimensionais de Froude

% Variáveis de entrada:
% densL -> Densidade do Líquido (kg/m³)
% densG -> Densidade do Gás (kg/m³)
% D -> diâmetro (m)
% v_LS -> velocidade superficial* do líquido
% v_GS -> velocidade superficial* do gás
%
% * velocidade superficial é definida como a vazão volumétrica dividida
% pela área total da seção transversal do tubo

% Calculando FL (número de Froud da fase líquida)

FL = (densL./(densL-densG)./9.8./D).^0.5.*abs(v_LS);
% Calculando FG (número de Froud da fase gás)

FG = (densG./(densL-densG)./9.8./D).^0.5.*abs(v_GS);

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