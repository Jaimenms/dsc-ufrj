function [S] = montaRegime(fl,San,Ses,Sbo,Sin,check)
% Função que consolida as variaveis de fechamento em função do padrão de
% escoamento.

campos=fieldnames(Ses);
aux = zeros(size(fl.Est));
for j=1:length(campos)
    S.(campos{j})=aux;
end

S=monta(S,Ses,fl.Est,campos,check.Est);
S=monta(S,San,fl.Anu,campos,check.Anu);
S=monta(S,Sin,fl.Int,campos,check.Int);
S=monta(S,Sbo,fl.Bub,campos,check.Bub);

S.ddPLdalfaG(find(isnan(S.ddPLdalfaG)))=0;
S.ddPGdalfaG(find(isnan(S.ddPGdalfaG)))=0;
S.dPL(find(isnan(S.dPL)))=0;
S.dPG(find(isnan(S.dPG)))=0;
S.dPI(find(isnan(S.dPI)))=0;
S.ddPLdP(find(isnan(S.ddPLdP)))=0;
S.ddPGdP(find(isnan(S.ddPGdP)))=0;    

function S=monta(S,Si,flagi,campos,checki)

if checki
    ix = find(flagi>0.1|isnan(flagi));
    for j=1:length(campos)
        S.(campos{j})(ix) = S.(campos{j})(ix)+flagi(ix).*Si.(campos{j});
    end
end