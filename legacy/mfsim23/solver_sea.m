function [X1,saidas,flag]=solver_sea(options,FUN,t,X1,parametros)
% Função que resolve um sist. alg. linear (Newton-Raphson).

global F2 X2

% Tol -> Tolerância do método (norma do vetor de residuos)
Tol=options.Tol;
% Numero máximo de iterações
maxIter=options.maxIter;

% Calculando valores iniciais
[F1]=feval(FUN,t,X1,parametros);

% Iniciando variáveis
cont=0; % contador
Res=norm(F1); % Resíduo
flag = 1; % Flag de convergência

% Iniciando painel
disp(['it.:     ' num2str(cont) '    norma :  ' num2str(norm(F1))])

status = odeprogressbar([0 1],[],'init');

% Executando laço de convergência
while Res>Tol&status==0

    if ~isreal(X1)
        disp('Parou devido à número complexo.')
        flag = 0;
        break
    end

    if cont==maxIter
        disp('Parou por atingir número máximo de iterações.')
        flag = 0;
        break
    end
    
    % Determinando jacobiana
    dF=jacobiana(options,F1,FUN,t,X1,parametros);

    % Busca unidimensional por passo ótimo
    passo=dF\F1;
    [fator_passo,Res] = fminbnd(@(fator_passo) veros(fator_passo,passo,FUN,t,X1,parametros),0,1);

    % Atualização dos estados
    X2=X1-fator_passo.*passo;

    % Cálculo do vetor de residuos
    [F2]=feval(FUN,t,X2,parametros);

    % Atualização das variaveis
    X1 = X2;
    F1 = F2;
    Res =norm(F1);
    cont=cont+1;    

    % Atualização do painel
    disp(['it.:     ' num2str(cont) '    norma :  ' num2str(norm(F1))])
    
    % Barra de Status
    lRes = -log(Res);
    lRes = max(0,lRes);
    lTol = -log(Tol);
    t = lRes./lTol;
    status = odeprogressbar(t,[],'on');
    
end

% Check final de convergência
if sum(sum(isnan(X1)))
    disp('Não Convergiu')
    flag = 0;
end

% Check final
if status==1
    disp('Parado pelo usuário')
end
odeprogressbar(t,[],'done');


%Empacotando saidas
saidas.cont=cont;
saidas.F=F1;
saidas.X=X1;
saidas.Res=Res;

function [Res] = veros(fator_passo,passo,FUN,t,X1,parametros)
% Função que faz a busca unidimensional pelo passo potimo
global X2 F2
X2=X1-fator_passo.*passo;   
[F2]=feval(FUN,t,X2,parametros);
Res =norm(F2);

function [dF]=jacobiana(options,F0,FUN,t,X1,parametros)
% Função para cálculo da matriz Jacobiana.

% Delta e Delta2 -> Parametros para calculo da Jacobiana numérica
Delta=options.Delta;
Delta2=options.Delta2;
% F0 -> vetor de resíduos no X0 médio da Jacobiana

dF=[];
for j=1:length(X1)
    X2=X1;
    X2(j)=(1+Delta)*X1(j)+Delta2;
    dF=[dF (feval(FUN,t,X2,parametros)-F0)./(X2(j)-X1(j))];
end


