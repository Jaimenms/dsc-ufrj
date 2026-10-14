function [X0]=simulador_SS(parametros,dados)
% Função que executa a simulação bifásica estacionária em redes de dutos.


% Preparando a rede estacionária em função do nível de discretização
[M,E,e,N,S] = monta_Rede(parametros);

% Desempacotando variaveis
parametros.M = M;
parametros.E = E;
parametros.e = e;
parametros.N = N;
parametros.S = S;

% Configurando Newton-Raphson
options.maxIter = 50;
options.Delta = 1e-3;
options.Delta2 = 1e-6;
options.Tol = 1e-9;

% Fornecendo estimativas iniciais para alfa
alfa0 = [0.1 0.25 0.5 0.75 0.9 0.01 0.99];

nalfa=length(alfa0);


% Executando busca de estados estacionarios
isComplexo=1;isOut=1;aux=0;
flag=1;
while flag||isComplexo || isOut
    % Atualiza Contador
    if aux>1,close,end
    aux=aux+1;
    if aux>nalfa,disp('Não possui SS'),break,end

    % Estimativas iniciais:
    %   Alfa
    parametros.alfa0 = alfa0(aux);
    %   Demais variáveis
    [F,x]=modelo_rede_SS([],[],parametros);

    % Executa newton-raphson
    [x,aux2,flag]=solver_sea(options,'modelo_rede_SS',[],x,parametros);
    flag=1-flag;

    % Gera novos alfas
    [F,x,alfa]=modelo_rede_SS([],x,parametros);

    % Testa novos alfas
    isOut=~isempty(find(alfa<0|alfa>1,1));
    isComplexo=~isreal(alfa);
end

% Gerando vetor de estados da rede
[Wk0,Ya0] = gera_CI(x,parametros,dados);
X0=[Wk0;Ya0];


function [Wk0,Ya0] = gera_CI(x,parametros,dados)
% Gera o vetor de estados da rede a partir da saída do simulador
% estacionario

% Desempacotando variaveis
Sw = parametros.Sw;
m = parametros.m;
NVertA = parametros.NVertA;
iVertA = parametros.iVertA;
Nk = parametros.Nk;
N = parametros.N;
S = parametros.S;

% Separando variaveis
alfa_SS=x(0*N+0*S+(1:S));
qL_SS  =x(0*N+1*S+(1:S));
qG_SS  =x(0*N+2*S+(1:S));
p_SS   =x(0*N+3*S+(1:N));

Wk0=[];
Ya0=[];
for i=1:m
    Wk0_aux=[];
    Yk0=[];
    vIn = dados.tubo{i}.verticeOrigem;
    vOut = dados.tubo{i}.verticeDestino;
    ix = 0:4:4*Nk(i);
    Yk0(ix+1,1)=ones(Nk(i)+1,1).*alfa_SS(i);
    Yk0(ix+2,1)=linspace(p_SS(vIn),p_SS(vOut),Nk(i)+1);
    Yk0(ix+3,1)=ones(Nk(i)+1,1).*qL_SS(i);
    Yk0(ix+4,1)=ones(Nk(i)+1,1).*qG_SS(i);
    Wk0_aux=Sw{i}'*Yk0;
    Wk0=[Wk0;Wk0_aux];
end
for i=1:NVertA
    alfa_aux=[];
    for j=1:m
        vIn = dados.tubo{j}.verticeOrigem;
        vOut = dados.tubo{j}.verticeDestino;
        if vIn==iVertA(i)||vOut==iVertA(i)
            alfa_aux=[alfa_aux alfa_SS(j)];
        end
    end
    Ya0((i-1)*2+1,:)=mean(alfa_aux);
    Ya0((i-1)*2+2,:)=p_SS(iVertA(i));
end


function [AG,PT,QL,QG] = busca_Ei(parametros)

Matriz_E = parametros.Matriz_E;

AG = Matriz_E(1:4:end,1);
PT = Matriz_E(2:4:end,1);
QL = Matriz_E(3:4:end,1);
QG = Matriz_E(4:4:end,1);

function [M,E,e,N,S] = monta_Rede(parametros)
% Prepara as matrizes e variaveis utilizadas na simulacao estacionaria

% Captura variaveis
NVert = parametros.NVert;
iVert = parametros.iVert;
iVertA = parametros.iVertA;
iVertB = parametros.iVertB;
MS = parametros.MS;
ME = parametros.ME;
m = parametros.m;

% Determina tamanhos da rede
N = NVert; %Numero de Vértices
S = m; %Numero de tubos

M = ME-MS; % Matriz de incidência

% Determinando a matriz de especificaçoes do simulador
E=zeros(0,3*(N+S));

% Determina Especificações
[AG,PT,QL,QG] = busca_Ei(parametros);

% Ativa especificações
iy=1;
for i=1:NVert
    % Especificando vertices do tipo A quanto a Vazão
    if iVert(i)==1
        % Liquido
        E(iy,3*S+1*N+i)=1;
        if QL(i)==-1
            e(iy,1)=0;
        else
            e(iy,1)=QL(i);
        end
        iy = iy+1;
        % Gas
        E(iy,3*S+2*N+i)=1;        
        if QG(i)==-1
            e(iy,1)=0;
        else
            e(iy,1)=QG(i);
        end
        iy = iy+1;        
    end
    % Especificando vertices do tipo B quanto a Pressão
    if iVert(i)==2
        if PT(i)>0
            E(iy,3*S+0*N+i)=1;
            e(iy,1)=PT(i);
            iy = iy+1;
        end
    end
    % Especificando vertices do tipo B quanto a Hold-up
    if iVert(i)==2
        if PT(i)>0
            aux=find(M(i,:)==-1);
            for k=1:length(aux)
                E(iy,aux(k))=1;
                e(iy,1)=AG(i);
                iy = iy+1;
            end
        end
    end
    % Especificando vertices do tipo B quanto a Vazao
    if iVert(i)==2
        if PT(i)<=0
            % Liquido
            E(iy,3*S+1*N+i)=1;
            e(iy,1)=QL(i);
            iy = iy+1;
            % Gas
            E(iy,3*S+2*N+i)=1;        
            e(iy,1)=QG(i);
            iy = iy+1;        
        end
    end
end    