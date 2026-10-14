function [dWdt]=solver_edp_galerkin(t,Y,Nk,n,h,Swk,modelo,parametros)
% Função que transforma sistemas de EDP em EDO segundo galerkin.

persistent IX teta invMk Mbjj Mbjjm1 pesos ne ixMODEL aux0 ne_0
persistent I_LAMBDAjj ix_LAMBDAjj I_LAMBDAjjm1 ix_LAMBDAjjm1
persistent I_CSIjj ix_CSIjj I_CSIjjm1 ix_CSIjjm1

nt = size(Y{1},2);
m = length(Y);

if isempty(invMk)
    
    % PESOS E ABCISSAS DE QUADRATURA
    % teta -> coordenadas modificadas de quadratura (nx1)
    % teta_ast -> coordenadas modificadas de quadratura do tipo asterisco
    % (nx1) 
    % teta2 -> fator de peso de quadratura (nx1)
    % teta2_ast -> fator de peso de quadratura do tipo asterisco (nx1)

    [abcissa,peso] = fatores_quadratura;

    % Escolhendo os pesos da quadratura
    abcissas=abcissa{n};
    pesos=peso{n};
    
    % Ativando tetas
    teta=-0.5+0.5.*abcissas;
    
    
    for i = 1 : m

        [aux1,aux2,aux3,A,B,ixModel]= feval(modelo{i});
        aux1=[];
        aux2=[];
        aux3=[];
        ne(i) = size(A,1);
        ne_0 = length(ixModel);

        ixMODELk=[];
        ni = size(Y{i},1);
        for j=1:length(ixModel)
            ixMODELk(j:ne_0:ni)=ixModel(j);
        end
        ixMODEL{i} = ixMODELk;
        
        % Ponteiros (4Nkx1)
        IX_K=1:ne_0*Nk(i)';

        % GERANDO MATRIZ DE DISCRETIZAÇÃO de GALERKIN
        % Mk -> matriz de discretização (ne*(Nk+1)xne*(Nk+1))
        [invMkK] = gera_Mk(Nk(i),ne(i),h{i});
        
        [MbjjK,Mbjjm1K] = gera_Mbk(Nk(i),ne(i),h{i});
        
        % GERANDO MATRIZ DE MONTAGEM DA MATRIZ DE COEFICIENTES LAMBDA
        % I_LAMBDAjj -> Matriz de ponteiros de geraçao da matriz LAMBDAjj
        % I_LAMBDAjjm1 -> Matriz de ponteiros de geraçao da matriz LAMBDAjjm1
        [I_LAMBDAjjK,ix_LAMBDAjjK,I_LAMBDAjjm1K,ix_LAMBDAjjm1K] = ...
            gera_LAMBDA(A,Nk(i),ne(i));
        % GERANDO MATRIZ DE MONTAGEM DA MATRIZ DE COEFICIENTES CSI
        % I_CSIjj -> Vetor de ponteiros de geraçao da matriz CSIjj
        % I_CSIjjm1 -> Vetor de ponteiros de geraçao da matriz CSIjjm1
        [I_CSIjjK,ix_CSIjjK,I_CSIjjm1K,ix_CSIjjm1K] = ...
            gera_CSI(B,Nk(i),ne(i));

        IX{i} = IX_K;
        invMk{i} = invMkK;
        Mbjj{i} = MbjjK;
        Mbjjm1{i} = Mbjjm1K;
        I_LAMBDAjj{i} = I_LAMBDAjjK;
        ix_LAMBDAjj{i} = ix_LAMBDAjjK;
        I_LAMBDAjjm1{i} = I_LAMBDAjjm1K;
        ix_LAMBDAjjm1{i} = ix_LAMBDAjjm1K;
        I_CSIjj{i} = I_CSIjjK;
        ix_CSIjj{i} = ix_CSIjjK;
        I_CSIjjm1{i} = I_CSIjjm1K;
        ix_CSIjjm1{i} = ix_CSIjjm1K;

        aux0{i} = zeros(ne(i).*(Nk(i)+1),ne(i));
        
    end
    
end

global tubo_ativo
dWdt = [];
ix = 0;
for j = 1:m

    tubo_ativo = j;
    hk = h{j};
    Yk = Y{j};
    IX_K = IX{j};
    SwkK = Swk{j};
    aux0k = aux0{j};
    invMkK = invMk{j};
    ixMODELk = ixMODEL{j};
        
    I_LAMBDAjjK = I_LAMBDAjj{j};
    I_LAMBDAjjm1K = I_LAMBDAjjm1{j};
    I_CSIjjK = I_CSIjj{j};
    I_CSIjjm1K = I_CSIjjm1{j};

    dWkdt = [];

    CT=[];
    XT=[];
    for i = 1:nt
        
        % Quadratura Gaussiana

        % C -> matriz de estados para cada ponto de quadratura (4Nkxn)
        % C mapeia interpolações dentro do elemento da esquerda para a
        % direita       
        C=Yk(IX_K+ne_0,i)*ones(1,n)+(Yk(IX_K+ne_0,i)-Yk(IX_K,i))*teta';
        

        if isempty(parametros{j}.x)
            x=[];
        else
            x_1K =parametros{j}.x(1:(end-1),1);
            x_2K =parametros{j}.x(2:(end  ),1);
            x = x_2K*ones(1,n)+(x_2K-x_1K)*teta';
            XT=[XT x];
        end
        CT=[CT C];
        
    end

    FT=feval(modelo{j},t,XT,CT,parametros,j);
    
    aux=0;
    FI=[];
    for i = 1:nt

        C = CT(:,aux+(1:n));        
        F = FT(:,aux+(1:n));        
        
        aux=aux+n;
                
        % teta  : [-1 0]
        % tetam1: [ 0 1]
        tetam1 = teta + 1;

        Fjjm1= 0.5.*F*((1-tetam1).*pesos);
        Fjj  = 0.5.*F*((1+teta  ).*pesos);
        
        % Determinando LAMBDA
        LAMBDAaux = zeros(size(I_LAMBDAjjK));
        
        LAMBDAaux(ix_LAMBDAjj{j})= Fjj(I_LAMBDAjjK(ix_LAMBDAjj{j}));
        LAMBDAaux(ix_LAMBDAjjm1{j}) = Fjjm1(I_LAMBDAjjm1K(ix_LAMBDAjjm1{j}));
                
        % Determinando CSI
        CSIjj = I_CSIjjK;
        CSIjjm1 = I_CSIjjm1K;
        CSIjj(ix_CSIjj{j}) = Fjj(I_CSIjjK(ix_CSIjj{j}));
        CSIjjm1(ix_CSIjjm1{j}) = Fjjm1(I_CSIjjm1K(ix_CSIjjm1{j}));

        CSI = Mbjj{j}*CSIjj + Mbjjm1{j}*CSIjjm1;
                        
        LAMBDA = [aux0k LAMBDAaux]-[LAMBDAaux aux0k];
        
        % Obtendo derivadas de estado reduzido do tubo j:
        FI(:,i)= (LAMBDA*Yk(find(ixMODELk),i)+CSI);
        
    end
           
    dWdt = [dWdt;-SwkK(find(ixMODELk),:)'*(invMkK*FI)];
end



function [invMk,Mk] = gera_Mk(Nk,ne,hk)
% Gera matriz inversa de Mk

uns = eye(ne);

MJk=zeros(ne.*(Nk+1));
MHk=zeros(ne.*(Nk+1));

for i=1:Nk
    MJk((i-1).*ne+[1:ne],(i-1).*ne+[1:ne])=2.*hk(i).*uns./6;
    MJk((i-0).*ne+[1:ne],(i-1).*ne+[1:ne])=hk(i).*uns./6;
end

for i=1:Nk
    MHk((i-0).*ne+[1:ne],(i-0).*ne+[1:ne])=2.*hk(i).*uns./6;
    MHk((i-1).*ne+[1:ne],(i-0).*ne+[1:ne])=hk(i).*uns./6;
end

Mk = sparse(MJk+MHk);

invMk=Mk\speye(size(Mk));

function [Mbjj,Mbjjm1] = gera_Mbk(Nk,ne,hk)

Mbjj  =zeros(ne.*(Nk+1),1);
Mbjjm1=zeros(ne.*(Nk+1),1);

for i=1:Nk
    Mbjj((i-0).*ne+[1:ne],1)=hk(i);
    Mbjjm1((i-1).*ne+[1:ne],1)=hk(i);
end

Mbjj=diag(Mbjj);
Mbjjm1=diag(Mbjjm1);


function [I_jj,ix_jj,I_jjm1,ix_jjm1] = gera_LAMBDA(A,Nk,ne)
% Gera matrizes que compoem a matriz LAMBDA

ixA = A>0;
I_aux=[];
for j=1:Nk
    aux = ixA.*((A-1).*Nk+j);
    I_aux = blkdiag(I_aux,aux);
end
I_jj=sparse([zeros(ne,ne*Nk);I_aux]);
I_jjm1=sparse([I_aux;zeros(ne,ne*Nk)]);
ix_jj=find(I_jj>0);
ix_jjm1=find(I_jjm1>0);

function [I_jj,ix_jj,I_jjm1,ix_jjm1] = gera_CSI(B,Nk,ne)
% Gera matrizes que compoem a matriz CSI

ixB = B>0;
I_aux=[];
for j=1:Nk
    aux = ixB.*((B-1).*Nk+j);
    I_aux = [I_aux;aux];
end
I_jj=sparse([zeros(ne,1);[I_aux]]);
I_jjm1=sparse([[I_aux];zeros(ne,1)]);
ix_jj=find(I_jj>0);
ix_jjm1=find(I_jjm1>0);

function [abcissa,peso] = fatores_quadratura
% Gera pesos e abcissas da quadratura gaussiana

abcissa{1}=0;
peso{1}=2;
abcissa{2}=[...
    -1/(3.^(0.5));...
    1/(3.^(0.5))];
peso{2}=[...
    1;...
    1];
abcissa{3}=[...
    -((3/5).^0.5);...
    0;...
    ((3/5).^0.5)];
peso{3}=[...
    5/9;...
    8/9;...
    5/9];
abcissa{5}=[...
    -0.906179845939;...
    -0.538469310106;...
    0;...
    0.538469310106;...
    0.906179845939];
peso{5}=   [...
    0.236926885056;...
    0.478628670499 ;...
    0.568888888889;...
    0.478628670499;...
    0.236926885056];
abcissa{8}=[...
    -0.960289856497536;...
    -0.796666477413627;...
    -0.525532409916329;...
    -0.183434642495650;...
    0.183434642495650;...
    0.525532409916329;...
    0.796666477413627;...
    0.960289856497536];
peso{8}=[0.101228536290376;...
    0.222381034453374;...
    0.313706645877887;...
    0.362683783378362;...
    0.362683783378362;...
    0.313706645877887;...
    0.222381034453374;...
    0.101228536290376];
abcissa{12}=[-0.981560634246719;...
    -0.904117256370475;...
    -0.769902674194305;...
    -0.587317954286617;...
    -0.367831498998180;...
    -0.125233408511469;...
    0.125233408511469;...
    0.367831498998180;...
    0.587317954286617;...
    0.769902674194305;...
    0.904117256370475;...
    0.981560634246719];
peso{12}=[...
    0.047175336386512;...
    0.106939325995318;...
    0.160078328543346;...
    0.203167426723066;...
    0.233492536538355;...
    0.249147045813403;...
    0.249147045813403;...
    0.233492536538355;...
    0.203167426723066;...
    0.160078328543346;...
    0.106939325995318;...
    0.047175336386512];
