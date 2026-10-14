function [ZP,saidas,eixos] = modelo_valvula(t,X,E,parametros)
% Função que determina pressão a montante da válvula em função da pressão a
% jusante e das vazões de líquido e de gás

saidas=[];
eixos =[];
ixP  = parametros.valvula.ixP;
ixqL = parametros.valvula.ixqL;
ixqG = parametros.valvula.ixqG;
CV = parametros.valvula.CV;
PR = parametros.PR;
NVal = parametros.NVal;

% Determinando numero de instantes de tempo
nt = size(X,2);

% Capturando variaveis
ix_aux = isnan(ixqL);
ixqL(ix_aux)=1;
qL = X(ixqL,:);
qL(ix_aux,:)=qL(ix_aux,:).*0;

ix_aux = isnan(ixqG);
ixqG(ix_aux)=1;
qG = X(ixqG,:);
qG(ix_aux,:)=qG(ix_aux,:).*0;

P  = E(ixP,:);

% Calculando vazão total
qM = qL+qG;

% Calculando densidade da mistura
[propL, propG] = modelo_fluido(P.*PR,0,parametros.Fluido);
densM = (qL.*propL.dens+qG.*propG.dens)./qM;

% Determinando abertura da válvula
[ab,saidasmalha] = modelo_malha(t,X,parametros);

% Determinando queda de pressão
dP = qM.*abs(qM)./densM./(0.0075.*diag(CV)*ab).^2;

% Determinando pressão a montante
ZP = (P.*PR+dP.*1e5)./PR;
ZP = max(1e-3,ZP);

for i=1:NVal
    saidas{i}.Modo  = saidasmalha.MODO(i,:);
    saidas{i}.SP    = saidasmalha.SP(i,:);
    saidas{i}.PV    = saidasmalha.PV(i,:);
    saidas{i}.Erro  = saidasmalha.Erro(i,:);
    saidas{i}.Ab    = saidasmalha.Ab(i,:);
    saidas{i}.ZP    = ZP(i,:).*PR./1e5;
    saidas{i}.dP    = dP(i,:);
    saidas{i}.qL    = qL(i,:);
    saidas{i}.qG    = qG(i,:);
    eixos.X.valor{i}= t;
    eixos.estilo  = '2D';
    eixos.X.nome  = 'Tempo (s)';
    eixos.X.tipo  = 'T';
end


function [Ab,saidasmalha] = modelo_malha(t,X,parametros)
% Função que modela malhas de controle PID.

% Declarando as informacoes persistentes
persistent tOld iErroOld dErroOld ErroOld

% Capturando Parametros das Malhas
acao = parametros.valvula.acao;
ContP = parametros.valvula.ContP;
ContI = parametros.valvula.ContI;
ContD = parametros.valvula.ContD;
ContBias = parametros.valvula.ContBias;
tEVal = parametros.valvula.tEVal;
Matriz_EVal1=parametros.valvula.Matriz_EVal1;
Matriz_EVal2=parametros.valvula.Matriz_EVal2;
Matriz_EVal3=parametros.valvula.Matriz_EVal3;
ixPV = parametros.valvula.ixPV;
NVal = parametros.NVal;

AbMan=interpola(t,tEVal,Matriz_EVal1);
MODO =interpola(t,tEVal,Matriz_EVal2);
SP   =interpola(t,tEVal,Matriz_EVal3);

% Partindo algoritmo de controle
if isempty(tOld)
     tOld=0;
     ErroOld  = zeros(NVal,1);
     iErroOld = zeros(NVal,1);
     dErroOld = zeros(NVal,1);
     iErroOld = (AbMan(:,1)-ContBias(:))./(ContP(:).*ContI(:));
end

nt = size(X,2);
ix_aux = isnan(ixPV);
ixPV(ix_aux)=1;
PV = X(ixPV,:);
PV(ix_aux,:)=PV(ix_aux,:).*0;


% Determinando Derivadas e Integrais do erro
Erro  = zeros(NVal,nt);
dErro = zeros(NVal,nt);
iErro = zeros(NVal,nt);
AbAuto= zeros(NVal,nt);
if NVal,
        
    Erro = diag(acao)*(PV-SP);

    for j=1:nt
        if t(j)<=tOld
            dErro(:,j)=dErroOld(:);
            iErro(:,j)=iErroOld(:);
        else
            for i=1:NVal
                dErro(i,j) = MODO(i,j).*(Erro(i,j) - ErroOld(i))./(t(j)-tOld);
                iErro(i,j) = MODO(i,j).*(iErroOld(i) + Erro(i,j).*(t(j)-tOld));
                iErro(i,j) = max(iErro(i,j),-1./(ContP(i).*ContI(i)));
                iErro(i,j) = min(iErro(i,j),1./(ContP(i).*ContI(i)));
            end
            if t(j)>tOld
               tOld = t(j);
               ErroOld(:)  = Erro(:,j); 
               dErroOld(:) = dErro(:,j); 
               iErroOld(:) = iErro(:,j); 
            end
        end
    end

    % Aplicando Algoritmo de Controle
    AbAuto = diag(ContP)*(Erro + diag(ContI)*iErro + diag(ContD)*dErro) + ContBias(:)*ones(1,nt);

    % Limitando valvula
    AbAuto = max(AbAuto,0);
    AbAuto = min(AbAuto,1);
end

% Aplicando modo de controle        
Ab = MODO.*AbAuto + (1-MODO).*AbMan;

saidasmalha.MODO = MODO;
saidasmalha.SP = SP;
saidasmalha.PV = PV;
saidasmalha.Erro = Erro;
saidasmalha.Ab = Ab;