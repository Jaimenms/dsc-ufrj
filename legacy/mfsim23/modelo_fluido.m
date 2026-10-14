function [propL, propG, tensao] = modelo_fluido(P,T,parametros)
% Função que descreve as propriedades dos fluidos.

%    [PROPL,PROPG, TENSAO] = modelo_fluido(P,T,parametros), para a
%    matriz M-por-N P correspondente aos estados de pressão, para a matriz
%    M-por-N T correspondente aos estados de temperatura e para o struct
%    parametros retorna O escalar TENSAO que corresponde à tensao
%    interfacial dos fluidos e os structs PROPL e PROPG com as seguintes
%    estruturas:
%      PROPL.dens    -> densidade da fase líquida
%      PROPG.dens    -> densidade da fase gas
%      PROPL.C       -> derivada da densidade pela pressão da fase líquida
%      PROPG.C       -> derivada da densidade pela pressão da fase gás
%      PROPL.ddensdT -> derivada da densidade pela temp. da fase líquida
%      PROPG.ddensdT -> derivada da densidade pela temp. da fase gás
%      PROPL.visc    -> viscosidade da fase líquida
%      PROPG.visc    -> viscosidade da fase gás
%    Sendo o struct de parametros de composto por:
%      viscG  -> viscosidade do gas
%      gamaG  -> constante politropica do gas
%      densG0 -> densidade do gas no estado padrao
%      viscL  -> viscosidade do liquido
%      gamaL  -> constante politropica do liquido
%      densL0 -> densidade do liquido no estado padrao
%      P0     -> pressão no estado padrão
%      tensao -> tensao interfacial

% Considerando expansão adiabática
propL.dens=parametros.densL0.*(P./parametros.P0).^(1/parametros.gamaL);
propG.dens=parametros.densG0.*(P./parametros.P0).^(1/parametros.gamaG);
propL.C=propL.dens./(P.*parametros.gamaL);
propG.C=propG.dens./(P.*parametros.gamaG);

% TODO - Definir derivada com temperatura
PROPL.ddensdT = P.*0;
PROPG.ddensdT = P.*0;

uns = ones(size(P));
tensao = parametros.tensao.*uns;
propL.visc = parametros.viscL.*uns;
propG.visc = parametros.viscG.*uns;

propL.velSom = (propL.dens./parametros.gamaL./P).^(-0.5);
propG.velSom = (propG.dens./parametros.gamaG./P).^(-0.5);