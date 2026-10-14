function [parametros]=calculos_iniciais(dados)
% Função que executa cálculos iniciais dos parametros a serem utilizadas 
% pelas demais rotinas 

if ~isfield(dados,'tubo')        ,dados.tubo{1}=[];                 end
if ~isfield(dados,'vertice')     ,dados.vertice{1}=[];              end
if ~isfield(dados,'reservatorio'),dados.reservatorio{1}=[];         end
if ~isfield(dados,'valvula')     ,dados.valvula{1}=[];              end
if ~isfield(dados,'valvulaRet')  ,dados.valvulaRet{1}=[];           end
if ~isfield(dados,'acoplador')   ,dados.acoplador{1}=[];            end


%% 1- Fluidos
parametros.Fluido.densL0 = dados.fluido.densidadeL0;
parametros.Fluido.densG0 = dados.fluido.densidadeG0;
parametros.Fluido.viscL  = dados.fluido.viscosidadeL;
parametros.Fluido.viscG  = dados.fluido.viscosidadeG;
parametros.Fluido.gamaL  = dados.fluido.gamaL;
parametros.Fluido.gamaG  = dados.fluido.gamaG;
parametros.Fluido.P0     = dados.fluido.pressao0;
parametros.Fluido.tensao = dados.fluido.tensao;


%% 2- Simulacao
ti     = dados.simulador.tempoInicial;
dt     = dados.simulador.incrementoTempo;
tf     = dados.simulador.tempoFinal;
PR     = dados.simulador.pressaoR;
lista_n=[3 5 8 12]; n=lista_n(dados.simulador.ordemQuadratura);
parametros.PR = PR;
T=ti:dt:tf;
parametros.T = T;
ne = 4;
parametros.ne = ne;
parametros.n = n;
if ~isfield(dados.simulador,'metodo'),
    parametros.metodo = 1;
else
    parametros.metodo = dados.simulador.metodo;
end

%% 3- Vertices
NVert=length(dados.vertice);
for i=1:NVert
    iVert(i,1)=dados.vertice{i}.tipo;
    V(i,1)=dados.vertice{i}.volume;
end
NVertA=sum(iVert==1);
NVertB=sum(iVert==2);
iVertA = find(iVert==1);
iVertB = find(iVert==2);

tE=[];
aux=0;
for i=1:NVert
    aux=aux+1;
    tE1{aux}= dados.vertice{i}.especificacaoT1;
    ME1{aux}= dados.vertice{i}.especificacao1;
    tE2{aux}= dados.vertice{i}.especificacaoT2;
    ME2{aux}= dados.vertice{i}.especificacao2;
    tE3{aux}= dados.vertice{i}.especificacaoT3;
    ME3{aux}= dados.vertice{i}.especificacao3;
    tE4{aux}= dados.vertice{i}.especificacaoT4;
    ME4{aux}= dados.vertice{i}.especificacao4;
    tE = [tE tE1{aux} tE2{aux} tE3{aux} tE4{aux}];
end
   
tE=unique([ti tE tf ]);
[Matriz_E1] = consolida_E(tE,tE1,ME1);
[Matriz_E2] = consolida_E(tE,tE2,ME2);
[Matriz_E3] = consolida_E(tE,tE3,ME3);
[Matriz_E4] = consolida_E(tE,tE4,ME4);

Matriz_E=[];
for i=1:size(Matriz_E1,1)
    Matriz_E=[Matriz_E;...
              Matriz_E1(i,:);...
              Matriz_E2(i,:);...
              Matriz_E3(i,:);...
              Matriz_E4(i,:)];
end

parametros.V           = V;
parametros.tE         = tE;
parametros.Matriz_E   = Matriz_E;
parametros.iVert       = iVert;
parametros.iVertA      = iVertA;
parametros.iVertB      = iVertB;
parametros.NVert       = NVert;
parametros.NVertA      = NVertA;
parametros.NVertB      = NVertB;

%% 4- Dutos
m=length(dados.tubo);
parametros.m = m;
for i=1:m
    x = dados.tubo{i}.x;
    y = dados.tubo{i}.y;
    z = dados.tubo{i}.z;
    nx = length(x);
    deltaX = x(2:nx)-x(1:(nx-1));
    deltaY = y(2:nx)-y(1:(nx-1));
    deltaZ = z(2:nx)-z(1:(nx-1));
    vNk=dados.tubo{i}.numeroElementos;
    hk=[];
    Ank=[];
    Nohx=x(:,1);
    Nohy=y(:,1);
    Nohz=z(:,1);
    for j=1:(nx-1)
        dL=sqrt(deltaX(j).^2+deltaY(j).^2+deltaZ(j).^2);
        dAn=asin(deltaZ(j)./dL);
        dh = dL./vNk(j);

        hk = [hk dh.*ones(1,vNk(j))];
        Ank = [Ank dAn.*ones(1,vNk(j))];

        Nohx=[Nohx x(j)+deltaX(j).*[(1/vNk(j)):(1/vNk(j)):1]];
        Nohy=[Nohy y(j)+deltaY(j).*[(1/vNk(j)):(1/vNk(j)):1]];
        Nohz=[Nohz z(j)+deltaZ(j).*[(1/vNk(j)):(1/vNk(j)):1]];
    end
    N{i}=vNk(:);
    Nk(i,1)=sum(vNk);
    An{i}=Ank(:);
    h{i}=hk(:);
    L(i,1)=sum(hk);
    NohX{i}=Nohx(:);
    NohY{i}=Nohy(:);
    NohZ{i}=Nohz(:);
    D(i,1) = dados.tubo{i}.diametro;
    rugo(i,1)=dados.tubo{i}.rugosidade./1e6;
    SegL{i}=sqrt(deltaX.^2+deltaY.^2+deltaZ.^2);
    SegZ{i}=z;
    xx{i}=cumsum([0;hk(:)]);

    parametros.D(i,1) = D(i);
    parametros.L(i,1) = L(i);
    parametros.rugo(i,1) = rugo(i);
    parametros.Duto{i}.x  = xx{i};
    parametros.Duto{i}.PR = PR;
    parametros.Duto{i}.D = D(i);
    parametros.Duto{i}.angulo_teta = An{i};
    parametros.Duto{i}.rugo = rugo(i);
    parametros.Duto{i}.Nk = Nk(i);
    parametros.Duto{i}.L = L(i);
    parametros.Duto{i}.regime = dados.tubo{i}.regime;
    parametros.regime(i,1) = dados.tubo{i}.regime;
    parametros.Duto{i}.N=N{i};
    parametros.Duto{i}.angulo_teta=An{i};
    parametros.Duto{i}.h=h{i};
    parametros.Duto{i}.NohX=NohX{i};
    parametros.Duto{i}.NohY=NohY{i};
    parametros.Duto{i}.NohZ=NohZ{i};
    vIn = dados.tubo{i}.verticeOrigem;
    vOut = dados.tubo{i}.verticeDestino;
    parametros.Duto{i}.vIn=vIn;
    parametros.Duto{i}.vOut=vOut;
    parametros.Duto{i}.Fluido = parametros.Fluido;
    auxZ = NohZ{i};
    parametros.Vertice{vIn}.z =auxZ(1);
    parametros.Vertice{vOut}.z =auxZ(end);
    parametros.Z(vIn,1)=auxZ(1);
    parametros.Z(vOut,1)=auxZ(end);
    
end
parametros.NohX = NohX;
parametros.NohY = NohY;
parametros.NohZ = NohZ;
parametros.SegL = SegL;
parametros.SegZ = SegZ;
parametros.Nk = Nk;
parametros.h = h;


%% 5- Rede
ME  = zeros(NVert,m);
MS  = zeros(NVert,m);
for i=1:NVert
    for j=1:m
        iDestino = dados.tubo{j}.verticeDestino;
        iOrigem = dados.tubo{j}.verticeOrigem; 
        ME(i,j)=(iDestino==i);
        MS(i,j)=(iOrigem==i);
    end
end
parametros.MS = MS;
parametros.ME = ME;


%% 7- Matrizes Seletoras

% Vetores Importantes:
%   W{k} -> vetor de estados reduzidos do duto
%   Y{k} -> vetor de estados do duto
%   Ya   -> vetor de estados do vertice tipo A
%   X    -> vetor de estados da rede
%   U    -> vetor de estados aumentados da rede

% Matrizes seletoras
%   Sw{k}-> transforma o vetor de estados do duto no vetor de estados
%           reduzidos do duto
%   Sxa  -> transforma o vetor de estados da rede no vetor de estados de
%           vertices do tipo A
%   Sya  -> transforma o vetor de estados dos vertices no vetor de estados da
%           rede

%   Seb  -> transforma especificaçoes de vertices do tipo B no vetor de estados
%           da rede


% Indicadores do vetor de estados do duto
%   IndY{k}.tipo -> indica qual o estado do duto usando o seguinte código
%                1: se é uma fração de área de gás em um nó do duto
%                2: se é uma pressão em um nó do duto
%                3: se é uma vazão de líquido em um nó do duto
%                4: se é uma vazão de gás em um nó do duto
%   IndY{k}.indice -> indica qual o índice do nó correspondente ao estado do duto
%   IndY{k}.CC   -> indica qual o tipo de condição de contorno associada ao estado
%                1: ausência de C.C.
%                2: C.C. do tipo Dirichlet
%                3: C.C. do tipo von Neumann
%   IndY{k}.CC_vertice -> indica qual o vertice que transfere a condição de contorno associada ao estado
%   IndY{k}.inlet -> indica se as variaveis correspondem a estados da
%   entrada dos dutos. 1 se sim e 0 se não.
%   IndY{k}.outlet -> indica se as variaveis correspondem a estados da
%   entrada dos dutos. 1 se sim e 0 se não.

% Determinando numero de estados por duto
Ny = (Nk(:)+1).*ne;

for k=1:m
    IndY{k}.tipo(1:4:Ny(k),1) = 1;
    IndY{k}.tipo(2:4:Ny(k),1) = 2;
    IndY{k}.tipo(3:4:Ny(k),1) = 3;
    IndY{k}.tipo(4:4:Ny(k),1) = 4;
end
for k=1:m
    IndY{k}.indice(1:4:Ny(k),1) = 1:(Nk(k)+1);
    IndY{k}.indice(2:4:Ny(k),1) = 1:(Nk(k)+1);
    IndY{k}.indice(3:4:Ny(k),1) = 1:(Nk(k)+1);
    IndY{k}.indice(4:4:Ny(k),1) = 1:(Nk(k)+1);
end
for k=1:m
    IndY{k}.CC = ones(Ny(k),1);
    IndY{k}.CC(1,1)          = dados.tubo{k}.tipocondicaoContornoE1;
    IndY{k}.CC(2,1)          = dados.tubo{k}.tipocondicaoContornoE2;
    IndY{k}.CC(3,1)          = dados.tubo{k}.tipocondicaoContornoE3;
    IndY{k}.CC(4,1)          = dados.tubo{k}.tipocondicaoContornoE4;
    IndY{k}.CC(Nk(k)*ne+1,1) = dados.tubo{k}.tipocondicaoContornoS1;
    IndY{k}.CC(Nk(k)*ne+2,1) = dados.tubo{k}.tipocondicaoContornoS2;
    IndY{k}.CC(Nk(k)*ne+3,1) = dados.tubo{k}.tipocondicaoContornoS3;
    IndY{k}.CC(Nk(k)*ne+4,1) = dados.tubo{k}.tipocondicaoContornoS4;
    
    if ~isfield(dados.tubo{k},'modelo')
        dados.tubo{k}.modelo = 1;
    end
        
    switch dados.tubo{k}.modelo
    case 2
        IndY{k}.CC(1,1) = 1;
        IndY{k}.CC(Nk(k)*ne+1,1) = 1;
    case 3
        IndY{k}.CC(1,1) = 1;
        IndY{k}.CC(Nk(k)*ne+1,1) = 1;
        IndY{k}.CC(4,1) = 1;
        IndY{k}.CC(Nk(k)*ne+4,1) = 1;
    case 4
        IndY{k}.CC(1,1) = 1;
        IndY{k}.CC(Nk(k)*ne+1,1) = 1;
        IndY{k}.CC(3,1) = 1;
        IndY{k}.CC(Nk(k)*ne+3,1) = 1;
    end
    
end

for k=1:m
    IndY{k}.inlet (1:Ny(k),1)       = 0;
    IndY{k}.inlet (1:4,1)           = 1;
    IndY{k}.outlet(1:Ny(k),1)       = 0;
    IndY{k}.outlet((Ny(k)-3):end,1) = 1;
end


for k=1:m
    IndY{k}.CC_vertice(1:Ny(k),1)=0;
    IndY{k}.CC_verticeA(1:Ny(k),1)=0;
    IndY{k}.CC_verticeB(1:Ny(k),1)=0;
    ix = find(MS(:,k));
    for j = 1:4,           if IndY{k}.CC(j,1)==2, IndY{k}.CC_vertice(j,1)=ix; end, end
    for j = 1:4,           if IndY{k}.CC(j,1)==2&&sum(iVertA==ix)>0, IndY{k}.CC_verticeA(j,1)=find(iVertA==ix); end, end
    for j = 1:4,           if IndY{k}.CC(j,1)==2&&sum(iVertB==ix)>0, IndY{k}.CC_verticeB(j,1)=find(iVertB==ix); end, end
    ix = find(ME(:,k));
    for j = 4*Nk(k)+(1:4), if IndY{k}.CC(j,1)==2, IndY{k}.CC_vertice(j,1)=ix; end, end
    for j = 4*Nk(k)+(1:4), if IndY{k}.CC(j,1)==2&&sum(iVertA==ix)>0, IndY{k}.CC_verticeA(j,1)=find(iVertA==ix); end, end
    for j = 4*Nk(k)+(1:4), if IndY{k}.CC(j,1)==2&&sum(iVertB==ix)>0, IndY{k}.CC_verticeB(j,1)=find(iVertB==ix); end, end
end


for k=1:m % modificando matriz seletora de CC para retirar estados não utilizados pelos modelos
    switch dados.tubo{k}.modelo
    case 2
        for i=1:(Nk(k)+1)
            IndY{k}.CC(4*(i-1)+1,1) = 2;
        end
    case 3
        for i=1:(Nk(k)+1)
            IndY{k}.CC(4*(i-1)+1,1) = 2;
            IndY{k}.CC(4*(i-1)+4,1) = 2;
        end
    case 4
        for i=1:(Nk(k)+1)
            IndY{k}.CC(4*(i-1)+1,1) = 2;
            IndY{k}.CC(4*(i-1)+3,1) = 2;
        end
    end
end

parametros.IndY = IndY;

% Indicadores do vetor de estados aumentados da rede
%   IndU.elemento -> indica qual elemento correspondente ao estado da rede
%                1: se corresponde a uma variável de duto
%                2: se corresponde a uma variável de vértice tipo A
%   IndU.tipo -> indica qual o estado da rede usando o seguinte código
%                1: se é uma fração de área de gás em um nó do duto
%                2: se é uma pressão em um nó do duto
%                3: se é uma vazão de líquido em um nó do duto
%                4: se é uma vazão de gás em um nó do duto
%                5: se é um hold-up de vértice tipo A
%                6: se é uma pressão de vértice tipo A
%   IndU.indice -> indica qual o índice do elemento correspondente ao estado da rede
%   IndU.noh    -> indica qual o noh do estado da rede (caso seja variavel
%   de linha)
%   IndU.CC   -> indica qual o tipo de condição de contorno associada ao estado
%                1: ausência de C.C.
%                2: C.C. do tipo Dirichlet
%                3: C.C. do tipo von Neumann

% Determinando o numero total de estados aumentados da rede
Nu = sum(Ny) + 2.*NVertA;

IndU.elemento = zeros(Nu,1);
IndU.elemento(1:sum(Ny),1)=1;
IndU.elemento(sum(Ny)+(1:(2*NVertA)),1)=2;
IndU.tipo = zeros(Nu,1);
IndU.noh = zeros(Nu,1);
for i=1:ne:Nu,
    IndU.tipo(i+(1:ne)-1,1)=[1;2;3;4];
end
for i=1:2:(2*NVertA),
    IndU.tipo(sum(Ny)+i-1+(1:2) ,1)=[5;6];
end
aux = 0;
for i=1:m,
    IndU.indice(aux+(1:Ny(i)),1)=i;
    aux = aux+Ny(i);
end
for i=1:NVertA,
    IndU.indice(aux+(1:2),1)=i;
    aux = aux+2;
end
aux = 0;
for i=1:m,
    for j=1:(Nk(i)+1)
        IndU.noh(aux+(1:ne),1)=j;
        aux = aux+ne;
    end
end

aux = 0;
IndU.CC = zeros(Nu,1);
for k=1:m
    IndU.CC(aux+(1:Ny(k)),1) = IndY{k}.CC;
    aux = aux+Ny(k);
end
parametros.IndU = IndU;


% Indicadores do vetor de estados da rede
%   IndX.elemento -> indica qual elemento correspondente ao estado da rede
%                1: se corresponde a uma variável de duto
%                2: se corresponde a uma variável de vértice tipo A
%   IndX.tipo -> indica qual o estado da rede usando o seguinte código
%                1: se é uma fração de área de gás em um nó do duto
%                2: se é uma pressão em um nó do duto
%                3: se é uma vazão de líquido em um nó do duto
%                4: se é uma vazão de gás em um nó do duto
%                5: se é um hold-up de vértice tipo A
%                6: se é uma pressão de vértice tipo A
%   IndX.indice -> indica qual o índice do elemento correspondente ao estado da rede
%   IndX.noh    -> indica qual o noh do estado da rede (caso seja variavel
%   de linha)

% Determinando numero de estados reduzidos por duto
NDir=[];
for k=1:m
    NDir(k,1)= sum(IndY{k}.CC==2);
end
NNeu=[];
for k=1:m
    NNeu(k,1) = sum(IndY{k}.CC==3);
end
Nw = Ny-NDir-NNeu;

% Determinando o numero total de estados
Nx = sum(Nw) + 2.*NVertA;

ix = IndU.CC~=2&IndU.CC~=3;
IndX.elemento(:,1) = IndU.elemento(ix);
IndX.tipo(:,1) = IndU.tipo(ix);
IndX.indice(:,1) = IndU.indice(ix);
IndX.noh(:,1) = IndU.noh(ix);
parametros.IndX=IndX;

% Gerando Sx{k}
j=1;
for k=1:m
    Sxk=zeros(Nw(k),Nx);
    for i=1:Nw(k)
        Sxk(i,j)=1;
        j=j+1;
    end
    Sx{k}=Sxk;
end

% Gerando Sw{k}
for k=1:m
    Swk=eye(Ny(k));
    Swk(:,IndY{k}.CC==2|IndY{k}.CC==3)=[];
    Sw{k}=Swk;
end

% Gerando Se
for k=1:m
    aux=0;
    Sek = zeros(Ny(k),NVert*4);
    for i=1:Ny(k)
        ix = IndY{k}.CC_vertice(i);
        iy = IndY{k}.tipo(i);
        if ix~=0
            Sek(i,4*(ix-1)+iy)=1;
        end
    end
    Se{k}=Sek;
end

% Gerando Sneu
for k=1:m
    Sneuk = zeros(Ny(k),Ny(k));
    for i=1:4,
        if IndY{k}.CC(i,1)==3, Sneuk(i,i+4)=1;end
    end
    for i=Nk(k)*ne+(1:4)
        if IndY{k}.CC(i,1)==3, Sneuk(i,i-4)=1;end
    end
    ix = IndY{k}.CC==2|IndY{k}.CC==3;
    Sneuk(:,ix)=[];
    Sneu{k}=Sneuk;
end

% Gerando Sxa
Sxa = eye(Nx);
Sxa(IndX.elemento~=2,:)=[];

% Gerando SVert
SVert1 = zeros(4*NVert,Nx);
SVert2 = eye(4*NVert,4*NVert);
for i = 1:NVert
    if iVert(i)==1
        aux = find(i==iVertA);
        SVert1((i-1)*4+1,sum(Nw)+(aux-1)*4+1) = 1;
        SVert1((i-1)*4+2,sum(Nw)+(aux-1)*4+2) = 1;
        SVert2((i-1)*4+1,(i-1)*4+1) = 0;
        SVert2((i-1)*4+2,(i-1)*4+2) = 0;
    end
end
parametros.SVert1 =SVert1 ;
parametros.SVert2 =SVert2 ;



%% 8- Vetores de seleção

% Vetor de seleção completa de estados
for k=1:m
    SYk = [(Sw{k}*Sx{k}+Sneu{k}*Sx{k}) Se{k}]; %(Ny(k) x (Nx + 4*NvertB)
    SY{k} =SYk;
end

% Vetor de seleção para evitar fluxo reverso
ixCheck =[];
for i=1:m
   if ~isfield(dados.tubo{i},'check'),
       dados.tubo{i}.check = 1;
   end
   if dados.tubo{i}.check==1
       aux = sum(Nw(1:i));
       ixCheck =[ixCheck;aux-1]; % Líquido
       ixCheck =[ixCheck;aux];  % Gás
   end
end

% Empacotando parametros
parametros.SY=SY;
parametros.Sw=Sw;
parametros.Sxa=Sxa;
parametros.ixCheck = ixCheck;
parametros.Sneu = Sneu;

for k=1:m
    switch dados.tubo{k}.modelo
        case 1
        parametros.modelo{k}=@modelo_duto;
        case 2
        parametros.modelo{k}=@modelo_duto_BB;
        case 3
        parametros.modelo{k}=@modelo_duto_L;
        case 4
        parametros.modelo{k}=@modelo_duto_G;
        case 5
        parametros.modelo{k}=@modelo_duto_BB2;            
    end
end



%% 9- Especificacoes Valvulas
if ~isfield(dados,'valvula'), dados.valvula{1}=[]; end
NVal = length(dados.valvula);
if isempty(dados.valvula{end}), NVal = NVal-1; end
indice = [];
CV=[];
tEVal=[];
SVal1 =  zeros(NVert.*4,NVal);
SVal2 =  eye(NVert.*4);
ixP = [];
ixPV= [];
ixqL=[];
ixqG=[];
linha=[];
tipoPV = [];
nohPV  = [];
linhaPV= [];
acao   = [];
ContP  = [];
ContI  = [];
ContD  = [];
ContBias = [];
for i=1:NVal
    indice(i,1)=dados.valvula{i}.vertice;
    linha(i)=find(ME(indice(i),:));
    CV(i,1)  =dados.valvula{i}.CV;
    tipoPV(i,1) =dados.valvula{i}.tipoPV;
    if isempty(dados.valvula{i}.nohPV),    nohPV(i,1)  =5;   else  nohPV(i,1)   =dados.valvula{i}.nohPV;     end
    if isempty(dados.valvula{i}.linhaPV),  linhaPV(i,1)  =1; else  linhaPV(i,1) =dados.valvula{i}.linhaPV;   end
    if isempty(dados.valvula{i}.acao),     acao(i,1)  =1;    else  acao(i,1)    =dados.valvula{i}.acao;      end
    if isempty(dados.valvula{i}.ContP),    ContP(i,1)  =0;   else  ContP(i,1)   =dados.valvula{i}.ContP;     end
    if isempty(dados.valvula{i}.ContI),    ContI(i,1)  =0;   else  ContI(i,1)   =dados.valvula{i}.ContI;     end
    if isempty(dados.valvula{i}.ContD),    ContD(i,1)  =0;   else  ContD(i,1)   =dados.valvula{i}.ContD;     end
    if isempty(dados.valvula{i}.ContBias), ContBias(i,1)  =0;else  ContBias(i,1) =dados.valvula{i}.ContBias; end
    
    tE1Val{i}=dados.valvula{i}.especificacaoT1;
    ME1Val{i}=dados.valvula{i}.especificacao1;
    tE2Val{i}=dados.valvula{i}.especificacaoT2;
    ME2Val{i}=dados.valvula{i}.especificacao2;
    tE3Val{i}=dados.valvula{i}.especificacaoT3;
    ME3Val{i}=dados.valvula{i}.especificacao3;
    
    tEVal = [tEVal tE1Val{i} tE2Val{i} tE3Val{i}];
    
    SVal1((indice(i)-1)*4+2,i)=1;
    SVal2((indice(i)-1)*4+2,(indice(i)-1)*4+2)=0;
    ixP(i,1)  = (indice(i)-1)*4+2;
    
    auxL  = IndX.noh==(Nk(linha(i))+1)&IndX.tipo==3&IndX.elemento==1&IndX.indice==linha(i);
    auxG  = IndX.noh==(Nk(linha(i))+1)&IndX.tipo==4&IndX.elemento==1&IndX.indice==linha(i);
    auxPV  = IndX.noh==nohPV(i,1)&IndX.tipo==tipoPV(i,1)&IndX.elemento==1&IndX.indice==linhaPV(i,1);
    if sum(auxL)==0, ixqL(i,1) = nan; else ixqL(i,1) = find(auxL); end
    if sum(auxG)==0, ixqG(i,1) = nan; else ixqG(i,1) = find(auxG); end
    if sum(auxPV)==0, ixPV(i,1) = nan; else ixPV(i,1) = find(auxPV); end
   
end

tEVal=unique([ti tEVal tf ]);
if NVal>0
    [Matriz_EVal1] = consolida_E(tEVal,tE1Val,ME1Val);
    [Matriz_EVal2] = consolida_E(tEVal,tE2Val,ME2Val);
    [Matriz_EVal3] = consolida_E(tEVal,tE3Val,ME3Val);
else
    tEVal=[];
    [Matriz_EVal1] =[];
    [Matriz_EVal2] =[];
    [Matriz_EVal3] =[];
end
parametros.NVal=NVal;
parametros.valvula.tEVal = tEVal;
parametros.valvula.Matriz_EVal1  = Matriz_EVal1;
parametros.valvula.Matriz_EVal2  = Matriz_EVal2;
parametros.valvula.Matriz_EVal3  = Matriz_EVal3;
parametros.valvula.indice=indice;
parametros.valvula.CV = CV;
parametros.valvula.SVal1=SVal1;
parametros.valvula.SVal2=SVal2;
parametros.valvula.ixP=ixP;
parametros.valvula.ixPV=ixPV;
parametros.valvula.ixqL=ixqL;
parametros.valvula.ixqG=ixqG;
parametros.valvula.tipoPV=tipoPV;
parametros.valvula.nohPV=nohPV;
parametros.valvula.linhaPV=linhaPV;
parametros.valvula.acao=acao;
parametros.valvula.ContP=ContP;
parametros.valvula.ContI=ContI;
parametros.valvula.ContD=ContD;
parametros.valvula.ContBias=ContBias;


%% 10- Especificacoes Valvulas de Retenção
if ~isfield(dados,'valvulaRet'), dados.valvulaRet{1}=[]; end
NValRet = length(dados.valvulaRet);
if isempty(dados.valvulaRet{end}), NValRet = NValRet-1; end
indice=[];
CV=[];
SValRet1 =  zeros(NVert.*4,NValRet);
SValRet2 =  zeros(NVert.*4,NValRet);
SValRet3 =  eye(NVert.*4);
ixP = [];
ixH=[];
ixPext=[];
linha=[];
dPMin = [];
for i=1:NValRet
    if isfield(dados.valvulaRet{i},'dPMin')
        dPMin(i,1) = dados.valvulaRet{i}.dPMin;
    else
        dPMin(i,1) = 0.;
    end

    indice(i,1)=dados.valvulaRet{i}.vertice;
    linha(i)=find(ME(indice(i),:));
    CV(i,1)  =dados.valvulaRet{i}.CV;
    SValRet1((indice(i)-1)*4+3,i)=1;
    SValRet2((indice(i)-1)*4+4,i)=1;
    SValRet3((indice(i)-1)*4+3,(indice(i)-1)*4+3)=0;
    SValRet3((indice(i)-1)*4+4,(indice(i)-1)*4+4)=0;
    
    aux = (indice(i)==iVertA);
    ixH(i,1)     = sum(Nw)+(aux-1)*2+1;
    ixP(i,1)     = sum(Nw)+(aux-1)*2+2;
    ixPext(i,1)  = (indice(i)-1)*4+2;
            
end
parametros.NValRet=NValRet;
parametros.valvulaRet.indice=indice;
parametros.valvulaRet.CV = CV;
parametros.valvulaRet.SValRet1=SValRet1;
parametros.valvulaRet.SValRet2=SValRet2;
parametros.valvulaRet.SValRet3=SValRet3;
parametros.valvulaRet.ixP=ixP;
parametros.valvulaRet.ixH=ixH;
parametros.valvulaRet.ixPext=ixPext;
parametros.valvulaRet.dPMin=dPMin;


%% 11- Especificacoes Reservatorio
if ~isfield(dados,'reservatorio'), dados.reservatorio{1}=[]; end
NRes = length(dados.reservatorio);
if isempty(dados.reservatorio{end}), NRes = NRes-1; end
vertice=[];
ixP = [];
IPL=[];
IPG=[];
PRes=[];
linha=[];
SRes1 =  zeros(NVert.*4,NRes);
SRes2 =  zeros(NVert.*4,NRes);
SRes3 =  eye(NVert.*4);
for i=1:NRes
    vertice(i,1)=dados.reservatorio{i}.vertice;
    IPL(i,1)=dados.reservatorio{i}.IPL;
    IPG(i,1)=dados.reservatorio{i}.IPG;
    PRes(i,1) =dados.reservatorio{i}.P;
    linha(i)=find(MS(vertice(i),:));
    SRes1((vertice(i)-1)*4+3,i)=1;
    SRes2((vertice(i)-1)*4+4,i)=1;
    SRes3((vertice(i)-1)*4+3,(vertice(i)-1)*4+3)=0;
    SRes3((vertice(i)-1)*4+4,(vertice(i)-1)*4+4)=0;
    IndX.noh==1&IndX.tipo==2&IndX.elemento==1&IndX.indice==linha(i)
    ixP(i,1) = find(IndX.noh==1&IndX.tipo==2&IndX.elemento==1&IndX.indice==linha(i));
end
parametros.NRes=NRes;
parametros.reservatorio.vertice=vertice;
parametros.reservatorio.ixP=ixP;
parametros.reservatorio.IPL=IPL;
parametros.reservatorio.IPG=IPG;
parametros.reservatorio.PRes = PRes;
parametros.reservatorio.SRes1=SRes1;
parametros.reservatorio.SRes2=SRes2;
parametros.reservatorio.SRes3=SRes3;


%% 12- Acoplador

% Gerando seletor de especificações de acoplamento
Nacop = length(dados.acoplador);
if isempty(dados.acoplador{end}), Nacop = Nacop-1; end

verticeE=[];
verticeS=[];
linhaE=[];
linhaS=[];
for i=1:Nacop
    verticeE(i)=dados.acoplador{i}.verticeE;
    verticeS(i)=dados.acoplador{i}.verticeS;
    linhaE(i)=find(ME(verticeE(i),:));
    linhaS(i)=find(MS(verticeS(i),:));
end
parametros.acoplador.verticeE=verticeE;
parametros.acoplador.verticeS=verticeS;
parametros.acoplador.linhaE=linhaE;
parametros.acoplador.linhaS=linhaS;

Sc1 =  zeros(NVert.*4,Nx);
Sc2 =  eye(NVert.*4);
for i=1:Nacop
    % Forçando pressão no vertice E igual a pressão no noh 1 da linha S
    ix = IndX.noh==1&IndX.tipo==2&IndX.elemento==1&IndX.indice==linhaS(i);
    Sc1((verticeE(i)-1)*4+2,ix)=1;
    Sc2((verticeE(i)-1)*4+2,(verticeE(i)-1)*4+2)=0;
end
parametros.Sc1=Sc1;
parametros.Sc2=Sc2;

Sc3 =  zeros(NVert.*4,Nx);
Sc4 =  eye(NVert.*4);
for i=1:Nacop
    % Forçando vazões no vertice S igual a vazões no noh final da linha E
    ix = IndX.noh==(Nk(linhaE(i))+1)&IndX.tipo==3&IndX.elemento==1&IndX.indice==linhaE(i);
    Sc3((verticeS(i)-1)*4+3,ix)=1;
    Sc4((verticeS(i)-1)*4+3,(verticeS(i)-1)*4+3)=0;
    ix = IndX.noh==(Nk(linhaE(i))+1)&IndX.tipo==4&IndX.elemento==1&IndX.indice==linhaE(i);
    Sc3((verticeS(i)-1)*4+4,ix)=1;
    Sc4((verticeS(i)-1)*4+4,(verticeS(i)-1)*4+4)=0;
end
parametros.Sc3=Sc3;
parametros.Sc4=Sc4;


%% Subfunções

function [E] = consolida_E(tE,tE1,ME1)
% Organizador da matriz de especificacoes

tEMax = max(tE);
tEMin = min(tE);
NV = length(tE1);
E=[];
for i=1:NV
    auxt1 = tE1{i};
    auxE1 = ME1{i};
    if isempty(auxE1)
        auxt1 = -1;
        auxE1 = -1;
    end
    if length(auxE1)==1
        auxt1 = [tEMin tEMax];
        auxE1 = [auxE1 auxE1];
    else
        if auxt1(end)<tEMax
            auxt1(end+1) = tEMax;
            auxE1(end+1) = auxE1(end);
        end
        if auxt1(1)<tEMin
            auxt1(1) = tEMin;
        end
    end
    E = [ E; interpola(tE,auxt1,auxE1)];
end
