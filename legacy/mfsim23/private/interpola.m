function ek=interpola(t,tEk,Ek)
% Função auxiliar que fisga de uma S-T um determinado conjunto de variaveis
% para um valor de tempo especificado

% Entradas:
% t  ->  vetor de tempos a serem fisgados (nt x 1) ou (1 x nt)
% tEk ->  vetor de tempos da serie temporal (ntp x 1) ou (1 x ntp)
% Ek ->  matriz de entradas ao longo da serie temporal (nent x ntp)

% Saidas:
% ek ->  matriz de entradas fisgadas (nt x nent)

nt = length(t);
ek=zeros(0,nt);
for i=1:nt
    it_inf=max(find(tEk<=t(i)));
    it_sup=min(find(tEk>=t(i)));
    if it_inf==it_sup
        ek=[ek Ek(:,it_inf)];
    else
        t_inf=tEk(it_inf);
        t_sup=tEk(it_sup);
        delta_t=(t_sup-t_inf);
        ek=[ek Ek(:,it_inf)+(t(i)-t_inf)/delta_t.*(Ek(:,it_sup)-Ek(:,it_inf))];
    end
end