function [entradas_mod,check]=filtra_entradas(entradas,fl)
% Função que filtra os valores gerados pelos modelos de fechamento em
% função do padrão de escoamento calculado ou estipulado.

entradas_mod=entradas;
campos=fieldnames(entradas);
padroes=fieldnames(fl);

for i=1:length(padroes)
    ix = find(fl.(padroes{i})>0.1|isnan(fl.(padroes{i})));
    if isempty(ix)
        check.(padroes{i})=0;
    else
        check.(padroes{i})=1;
        for j=1:length(campos)
            entradas_mod.(padroes{i}).(campos{j}) = entradas.(campos{j})(ix);
        end
    end
end
