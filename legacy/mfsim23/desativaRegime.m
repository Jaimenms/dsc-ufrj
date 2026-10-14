function [fl, reg1, reg2] = desativaRegime(regime,zers)
% Função que desativa o regime de escoamento forçando um regime previamente
% estipulado pelo usuário.

fl.Anu=zers;
fl.Bub=zers;
fl.Int=zers;
fl.Est=zers;
fl.EsS=zers;
fl.EsW=zers;
switch regime
    case 1 % Fixando regime anular
        fl.Anu(:,:)=1;
    case 2 % Fixando regime bolhas
        fl.Bub(:,:)=1;
    case 3 % Fixando regime intermitente
        fl.Int(:,:)=1;
    case 4 % Fixando regime estratificado suave
        fl.Est(:,:)=1;
        fl.EsS(:,:)=1;
    case 5 % Fixando regime estratificado ondulado
        fl.Est(:,:)=1;
        fl.EsW(:,:)=1;        
end
% Recalculando indicadores de regime
reg1 = fl.Anu.*1+fl.Bub.*2+fl.Int.*3+fl.Est.*4;
reg2 = fl.Anu.*1+fl.Bub.*2+fl.Int.*3+fl.EsS.*4+fl.EsW.*5;