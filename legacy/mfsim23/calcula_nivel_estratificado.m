function [hD,beta]=calcula_nivel_estratificado(alfaG)
% Função que determinando altura de líquido /diâmetro para escoamento
% estratificado de forma rigorosa

% Polinomios
P_h=[0.00000000000197;...
       -3.66048527474641;...
       9.15121318685793;...
       -8.89104962298284;...
       4.18536124761860;...
       -1.75567780293538;...
       0.98531913309307]';

P_beta=1.0e+002.*...
    [0.00000000000056;...
     -0.67451330764088;...
     1.68628326909991;...
     -1.58733054088785;...
     0.69471254223251;...
     -0.17280444315654;...
     0.05824216671205]';

% Calculando hD
hD=polyval(P_h,alfaG);

% Calculando beta
beta=polyval(P_beta,alfaG);

% Tratando insonsistências
beta = max(0,beta);
beta = min(2.*pi,beta);

Tol=1e-9;
maxIter=100;
[nx,ny] = size(alfaG);

for i=1:nx
    for j=1:ny
        alfaLi = (1-alfaG(i,j));
        if alfaLi>(1-1e-6) % Caso apenas Líquido
            betai=2*pi.*alfaLi;
        elseif alfaLi<(1e-6) % Caso apenas Gás
            betai = 2*pi.*alfaG(i,j);
        else % Caso intermediário
            % Preparando método
            cont=0;
            erro=1;
            betai=beta(i,j);
            % Executando Newton-Raphson
            while erro>Tol
                if cont>=maxIter;break,end
                F0=sin(betai)-betai+2*pi*alfaLi;
                dF=cos(betai)-1;
                passo = F0/dF;
                betai = betai - passo;
                erro  = abs(passo);
                cont=cont+1;
            end
        end
        beta(i,j) = betai;
    end
end

% Calculando hD a partir do angulo de molhamento
hD=1/2*(1-cos(beta/2));
