function [dWdt]=solver_edp_diferencas_upwind(t,Y,Nk,n,h,Swk,modelo,parametros)
% Função que transforma sistemas de EDP em EDO segundo diferenças finitas
% upwind.

persistent invMk Mbjj ne ne_0 ixMODEL
persistent I_LAMBDAjj1 ix_LAMBDAjj1 I_LAMBDAjj2 ix_LAMBDAjj2
persistent I_CSIjj ix_CSIjj

nt = size(Y{1},2);
m = length(Y);

if isempty(invMk)
            
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
        

        % GERANDO MATRIZ DE DISCRETIZAÇÃO de GALERKIN
        % Mk -> matriz de discretização (ne*(Nk+1)xne*(Nk+1))
        [invMkK] = gera_Mk(Nk(i),ne(i),h{i});
        [MbjjK] = gera_Mbk(Nk(i),ne(i),h{i});
        
        % GERANDO MATRIZ DE MONTAGEM DA MATRIZ DE COEFICIENTES LAMBDA
        % I_LAMBDAjj -> Matriz de ponteiros de geraçao da matriz LAMBDAjj
        % I_LAMBDAjjm1 -> Matriz de ponteiros de geraçao da matriz LAMBDAjjm1
        [I_LAMBDAjjK1,ix_LAMBDAjjK1,I_LAMBDAjjK2,ix_LAMBDAjjK2] = gera_LAMBDA(A,Nk(i),ne(i));

        % GERANDO MATRIZ DE MONTAGEM DA MATRIZ DE COEFICIENTES CSI
        % I_CSIjj -> Vetor de ponteiros de geraçao da matriz CSIjj
        % I_CSIjjm1 -> Vetor de ponteiros de geraçao da matriz CSIjjm1
        [I_CSIjjK,ix_CSIjjK] = gera_CSI(B,Nk(i),ne(i));

        invMk{i} = invMkK;
        Mbjj{i} = MbjjK;
        I_LAMBDAjj1{i} = I_LAMBDAjjK1;
        ix_LAMBDAjj1{i} = ix_LAMBDAjjK1;
        I_LAMBDAjj2{i} = I_LAMBDAjjK2;
        ix_LAMBDAjj2{i} = ix_LAMBDAjjK2;
        I_CSIjj{i} = I_CSIjjK;
        ix_CSIjj{i} = ix_CSIjjK;
        
    end
    
end


dWdt = [];
ix = 0;
for j = 1:m

    hk = h{j};
    Yk = Y{j};
    SwkK = Swk{j};
    invMkK = invMk{j};
    ixMODELk = ixMODEL{j};
        
    I_LAMBDAjjK1 = I_LAMBDAjj1{j};
    I_LAMBDAjjK2 = I_LAMBDAjj2{j};
    I_CSIjjK = I_CSIjj{j};

    dWkdt = [];

    XT=[];
    for i = 1:nt
        
        if isempty(parametros{j}.x)
            x=[];
        else
            x = parametros{j}.x;
            XT=[XT x];
        end
        
    end

    FT=feval(modelo{j},t,XT,Yk,parametros,j);
    
    FI=[];
    for i = 1:nt

        C = Yk(:,i);
        F = FT(:,i);
        
                
        % Determinando LAMBDA
        LAMBDA1 = zeros(size(I_LAMBDAjjK1));
        LAMBDA2 = zeros(size(I_LAMBDAjjK2));
        LAMBDA1(ix_LAMBDAjj1{j})= F(I_LAMBDAjjK1(ix_LAMBDAjj1{j}));
        LAMBDA2(ix_LAMBDAjj2{j})= F(I_LAMBDAjjK2(ix_LAMBDAjj2{j}));
                
        % Determinando CSI
        CSIjj = I_CSIjjK;
        CSIjj(ix_CSIjj{j}) = F(I_CSIjjK(ix_CSIjj{j}));

        CSI = Mbjj{j}*CSIjj;
        
        
        % Obtendo derivadas de estado reduzido do tubo j:

        LAMBDA = LAMBDA1-LAMBDA2;
        
        % Obtendo derivadas de estado reduzido do tubo j:
        FI(:,i)= (LAMBDA*Yk(find(ixMODELk),i)+CSI);
        

    end
             
    dWdt = [dWdt;-SwkK(find(ixMODELk),:)'*(invMkK*FI)];
end



function [invMk,Mk] = gera_Mk(Nk,ne,hk)
% Gera matriz inversa de Mk

uns = eye(ne);

Mk=zeros(ne.*(Nk+1));

Mk(1:ne,1:ne)= hk(1).*uns;

for i=1:Nk
    Mk(i.*ne+[1:ne],i.*ne+[1:ne])=hk(i).*uns;
end

Mk = sparse(Mk);

invMk=Mk\speye(size(Mk));

function [Mbjj] = gera_Mbk(Nk,ne,hk)

Mbjj  =zeros(ne.*(Nk+1),1);
Mbjj(1:ne,1)= hk(1);

for i=1:Nk
    Mbjj((i-0).*ne+[1:ne],1)=hk(i);
end

Mbjj = diag(Mbjj);

function [I_jj1,ix_jj1,I_jj2,ix_jj2] = gera_LAMBDA(A,Nk,ne)
% Gera matrizes que compoem a matriz LAMBDA

ixA = A>0;
I_aux=[];
I_jj1 = zeros((Nk+1).*ne);
I_jj2 = zeros((Nk+1).*ne);
for j=1:(Nk+1)
    aux = ixA.*((A-1).*(Nk+1)+j);
    if j==1
        I_aux1 = aux;
    elseif j==(Nk+1)
        I_auxNkM1 = aux;
    else
        I_aux = blkdiag(I_aux,aux);
    end
end
I_jj1([1:ne],ne+[1:ne])=I_aux1;
I_jj1(1*ne+[1:(ne*(Nk-1))],1*ne+[1:(ne*(Nk-1))])=I_jj1(1*ne+[1:(ne*(Nk-1))],2*ne+[1:(ne*(Nk-1))])+I_aux;
I_jj1(ne.*Nk+[1:ne],ne.*Nk+[1:ne])=I_auxNkM1;
I_jj1 = sparse(I_jj1);
ix_jj1=find(I_jj1>0);

I_jj2([1:ne],[1:ne])=I_aux1;
I_jj2(ne.*Nk+[1:ne],ne.*(Nk-1)+[1:ne])=I_auxNkM1;
I_jj2(1*ne+[1:(ne*(Nk-1))],0*ne+[1:(ne*(Nk-1))])=I_jj2(1*ne+[1:(ne*(Nk-1))],0*ne+[1:(ne*(Nk-1))])+I_aux;
I_jj2 = sparse(I_jj2);
ix_jj2=find(I_jj2>0);


function [I_jj,ix_jj] = gera_CSI(B,Nk,ne)
% Gera matrizes que compoem a matriz CSI

ixB = B>0;
I_aux=[];
for j=1:(Nk+1)
    aux = ixB.*((B-1).*(Nk+1)+j);
    I_aux = [I_aux;aux];
end
I_jj=sparse(I_aux);
ix_jj=find(I_jj>0);