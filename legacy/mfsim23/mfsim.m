function mfsim
% Interface gráfica para tela de abertura do programa.

% Definindo parametros de configurção da tela
parametrosBanner.posicao = [0.02, 0.85, 0.96, 0.6];
parametrosBanner.nomeModulo = 'PAINEL DE CONTROLE';
parametrosBanner.cor = [0 0.5 0.5];
parametrosBanner.corFonte = [0 0 0];
parametrosBanner.nomeSoftware = 'SDMPFLOW';
parametrosMenu.posicao = [0.275, 0.3, 0.45, 0.2];
parametrosMenu.nomeModulo = ...
    {'ENTRADA','PADROES','SAIDA'};
parametrosMenu.fontSize = 11; 
nome = 'SDMPFLOW';
texto = 'desenvolvido por Jaime Souza';

hfig = findall(0,'name',nome);
% Gerando tela
if isempty(hfig)
    gera_figura('SDMPFLOW');
    image(imread('image','jpg')), axis off
    gera_banner(gcf, parametrosBanner);
    menu(parametrosMenu);
    uicontrol(gcf,'Style','text','String',texto,'FontSize',8,...
        'backgroundcolor',[1 1 1],'position',[330 20 200 20]);
else
    figure(hfig(1))
end


function menu(parametros)
%Menu com os botões de ação.
%
% Essa subfunction recebe os parametros de configuração de um struct e gera
% um conjunto de botoes devidamente posicionados na forma de um menu.
% Essa subfunctionn trabalha associada com uma subfunction de callback.
% PARAMETROS é um struct que possui os seguintes campos:
%   PARAMETROS.posicao - vetor de quatro elementos que determina a posição
%   do menu de botoes. Os valores variam de 0 a 1.
%   PARAMETROS.nomeModulo - cell array com os strings dos nomes dos botões.
%   PARAMETROS.fontsize - tamanho da fonte do nome dos botoes

% Capturando parametros
posicaoFator1 = parametros.posicao(1);
posicaoFator2 = parametros.posicao(2);
tamanhoFator1 = parametros.posicao(3);
tamanhoFator2 = parametros.posicao(4);
nomeBotoes = parametros.nomeModulo;
fontSize = parametros.fontSize;

% Determinando tamanhos
tamanhoFigura = get(gcf,'position');
numeroBotoes = length(nomeBotoes);
tamanhoBotoes = [ tamanhoFator1.*tamanhoFigura(3) ...
    tamanhoFator2.*tamanhoFigura(4)/numeroBotoes];

% Gerando objetos
for i=1:numeroBotoes
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',nomeBotoes{i},...
        'UserData','menu1','Callback',@callback1,'FontSize',fontSize,...
        'Callback',{@callback,i});
end

function callback(src,eventdata,action)
%Callbacks dos botoes do menu.
%
% Executa as ações para cada botao.
% SRC e EVENT não são utilizados nessa subfunction.
% ACTION é o flag o indice que determina qual o botão que está acionando o
% callback.

switch action
    case 1
        % Função de organização de dados de entrada e execução.
        mfsim_entrada
    case 2
        % Função de visuazação de resultados.
        mfreg
    case 3
        % Função de visuazação de resultados.
        mfsim_saida
end
