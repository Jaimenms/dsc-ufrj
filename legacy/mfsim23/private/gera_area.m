function gera_area(hfig, parametros)
% Função auxiliar que gera diferentes areas da figura

% Captura parametros
posicaoFator1 = parametros.posicao(1);
posicaoFator2 = parametros.posicao(2);
tamanhoFator1 = parametros.posicao(3);
tamanhoFator2 = parametros.posicao(4);
nomeFrame = parametros.nomeFrame;
corFrame = parametros.cor;
corFonte = parametros.corFonte;

% Busca Figura
tamanhoFigura = get(hfig,'position');

% Determina tamanhos
tamanhoFrame = [ tamanhoFator1.*tamanhoFigura(3) ...
    tamanhoFator2.*tamanhoFigura(4) ];
posicaoFrame = [ posicaoFator1.*tamanhoFigura(3) ...
    posicaoFator2.*tamanhoFigura(4) ];

% Gera objetos
uicontrol(hfig,'style','frame','position',[posicaoFrame tamanhoFrame],...
    'BackGroundColor',corFrame);
uicontrol(hfig,'style','text','FontWeight','bold',...
    'position',[posicaoFrame(1)+1 posicaoFrame(2)+tamanhoFrame(2)-22 ...
    tamanhoFrame(1)-2 20],'string',nomeFrame,'BackGroundColor',corFrame,...
    'ForeGroundColor',corFonte);
