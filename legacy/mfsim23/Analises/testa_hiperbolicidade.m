function Hiperb = testa_hiperbolicidade(plota)
% Funcao que verifica se o sistema de equacoes diferenciais é hiperbolico

global dados saidasRede saidasDuto saidasDuto2D

[parametros]=calculos_iniciais(dados);

Nk = parametros.Nk;
ne = parametros.ne;
m  = parametros.m;
A  = parametros.A;
B  = parametros.B;

ixA = A>0;

if ~exist('plota'),
    plota=0;
end   


for i=1:m
    % ix -> vetor de ponteiros (1xNk)
    
    ix = 1:ne:(Nk(i)*ne);
    PR = parametros.Duto{i}.PR;
    y=[];    
    y(ix+3,:) = saidasDuto{i}.vazaoMassaG;
    y(ix+2,:) = saidasDuto{i}.vazaoMassaL;
    y(ix+1,:) = saidasDuto{i}.pressao./PR.*1e5;
    y(ix,:) = saidasDuto{i}.holdup;
    
    nt = size(y,2);
    
    [F]=modelo_duto(y,parametros.Duto{i},parametros.Fluido);
    
    
    
    for j=1:nt
        
        Ft = F(:,j);
        
        for k=1:Nk(i)
        
            Iaux = ixA.*((A-1).*Nk(i)+k);
            ix=find(Iaux);

            Mt = zeros(size(A));
            Mt(ixA) = Ft(Iaux(ix));
            
            Et = sort(eig(Mt));
            E1(k,j) = Et(1);
            E2(k,j) = Et(2);
            E3(k,j) = Et(3);
            E4(k,j) = Et(4);
            
            isHiper(k,j) = isreal(Et);
        end                
    end
    
    Hiperb{i}.E1=E1;
    Hiperb{i}.E2=E2;
    Hiperb{i}.E3=E3;
    Hiperb{i}.E4=E4;
    Hiperb{i}.isHiper=isHiper;
    
    if plota
        figure
        surf(Hiperb{i}.E1)
        hold on
        surf(Hiperb{i}.E2)
        surf(Hiperb{i}.E3)
        surf(Hiperb{i}.E4)
        shading interp
        title('Autovalores da matriz do sistemas de EDPs')
        ylabel('Instantes')
        xlabel('Elementos')
        zlabel('Autovalores')

        figure
        surf(abs(Hiperb{i}.E1))
        hold on
        surf(abs(Hiperb{i}.E2))
        surf(abs(Hiperb{i}.E3))
        surf(abs(Hiperb{i}.E4))
        shading interp
        title('Autovalores da matriz do sistemas de EDPs')
        ylabel('Instantes')
        xlabel('Elementos')
        zlabel('abs(Autovalores)')

        figure
        subplot(2,2,1)
        surf(Hiperb{i}.E1)
        shading interp
        title('Autovalores da matriz do sistemas de EDPs')
        subplot(2,2,2)
        surf(Hiperb{i}.E2)
        shading interp
        subplot(2,2,3)
        surf(Hiperb{i}.E3)
        shading interp
        subplot(2,2,4)
        surf(Hiperb{i}.E4)
        shading interp
        ylabel('Instantes')
        xlabel('Elementos')
        zlabel('Autovalores')

    end

end
    
    
