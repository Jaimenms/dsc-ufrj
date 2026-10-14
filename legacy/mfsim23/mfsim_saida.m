function mfsim_saida
% Interface gráfica para resultados de escoamento bifásico.

global dados detalhesVariaveis nomes saidas eixos

% Capturando estrutura de dados e saidas    
if isempty(dados)|isempty(saidas),
    [detalhesVariaveis, dados] = arvore_dados;
end
[nomes] = arvore_saidas;

% Apresentando parametros de configuração da tela
parametrosBanner.posicao = [0.02, 0.94, 0.96, 0.6];
parametrosBanner.nomeModulo = 'RESULTADOS';
parametrosBanner.cor = [0 0.5 0.5];
parametrosBanner.corFonte = [0 0 0];
parametrosBanner.nomeSoftware = 'SDMPFLOW';

parametrosArea1.posicao = [0.02, 0.02, 0.10, 0.92];
parametrosArea1.nomeFrame = 'ELEMENTOS';
parametrosArea1.cor = [0 0.5 0.5];
parametrosArea1.corFonte = [1 1 1];

parametrosArea2.posicao = [0.14, 0.02, 0.10, 0.92];
parametrosArea2.nomeFrame = 'OBJETOS';
parametrosArea2.cor = [0 0.5 0.5];
parametrosArea2.corFonte = [1 1 1];

parametrosArea3.posicao = [0.26, 0.02, 0.72, 0.92];
parametrosArea3.nomeFrame = 'GRÁFICOS';
parametrosArea3.cor = [0.8 0.8 0.8];
parametrosArea3.corFonte = [0 0 0];

% Gerando tela
nome = 'SAIDA' ;
hfig = findall(0,'name',nome);
if isempty(hfig)
    gera_figura(nome,5);
    gera_banner(gcf, parametrosBanner)
    gera_area(gcf, parametrosArea1)
    gera_area(gcf, parametrosArea2)
    gera_area(gcf, parametrosArea3)
    
    gera_menu1
    gera_menu2
    if ~isempty(saidas),     gera_menu3, end
    gera_menubar
else
    figure(hfig)
end

function gera_menu1
%Menu com os botões do menu principal.
%
% Essa subfunction recebe os parametros de configuração de um struct e gera
% um conjunto de botoes devidamente posicionados na forma de um menu.
% Essa subfunctionn trabalha associada com uma subfunction de callback.
% PARAMETROS é um struct que possui os seguintes campos:
%   PARAMETROS.posicao - vetor de quatro elementos que determina a posição
%   do menu de botoes. Os valores variam de 0 a 1.
%   PARAMETROS.nomeModulo - cell array com os strings dos nomes dos botões.
%   PARAMETROS.fontsize - tamanho da fonte do nome dos botoes

% Resgatado informação dos dados
global saidas nomes

% Captura parametros
posicaoFator1 = 0.03;
posicaoFator2 = 0.05;
tamanhoFator1 = 0.08;
tamanhoFator2 = 0.86;

% Limpando Objetos
limpa_objetos({'menu1','menu2','menu3'});

% Posicionando Objetos
tamanhoFigura = get(gcf,'position');
tamanhoBotoes = [ tamanhoFator1.*tamanhoFigura(3) 20];

% Capturando niveis possiveis dos dados
if isempty(saidas),
    nomesNivel1=[];
else
    nomesNivel1= fieldnames(saidas);
end
botoesNumero = length (nomesNivel1);

% Gerando botoes
for i=1:botoesNumero
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2)...
       .*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],...
        'string',nomes.(nomesNivel1{i}),...
        'UserData','menu1',...
        'Callback',{@callback1, nomesNivel1{i}});
end

function gera_menu2(nomesNivel1)
%Menu com os botões do menu secundário.
%
% Essa subfunction recebe os parametros de configuração de um struct e gera
% um conjunto de botoes devidamente posicionados na forma de um menu.
% Essa subfunctionn trabalha associada com uma subfunction de callback.

% Resgatado informação dos dados
global saidas nomes

% Captura parametros
posicaoFator1 = 0.15;
posicaoFator2 = 0.05;
tamanhoFator1 = 0.08;
tamanhoFator2 = 0.86;

% Limpando Objetos
limpa_objetos({'menu2','menu3'});

% Posicionando Objetos
tamanhoFigura = get(gcf,'position');
tamanhoBotoes = [ tamanhoFator1.*tamanhoFigura(3) 20];

if ~exist('nomesNivel1','var'),
    nomesNivel1aux = fieldnames(saidas);
    nomesNivel1 =nomesNivel1aux{1};
end

% Capturando niveis possiveis dos dados
botoesNumero = length (saidas.(nomesNivel1));

% Gerando botoes
for i=1:botoesNumero
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',...
        [nomes.(nomesNivel1) ' - ' num2str(i)],...
        'UserData','menu2','Callback',{@callback1, nomesNivel1,i});
end


function gera_menu3(nomesNivel1,k)
%Menu com os botões do menu terciário.
%
% Essa subfunction recebe os parametros de configuração de um struct e gera
% um conjunto de botoes devidamente posicionados na forma de um menu.
% Essa subfunctionn trabalha associada com uma subfunction de callback.

% Resgatado informação dos dados
global saidas nomes eixos

% Limpando Objetos
limpa_objetos({'menu3'});

% Posicionando Objetos
tamanhoFigura = get(gcf,'position');

if ~exist('nomesNivel1','var'),
    nomesNivel1aux = fieldnames(saidas);
    nomesNivel1    = nomesNivel1aux{1};
end
if ~exist('k','var'),
    k =1;
end

[ tiposBotao ] = tipos_botao;

%Tamanhos
posicaoFator1Texto = 0.28;
posicaoFator2Texto = 0.05;
tamanhoFator1Texto = 0.23;
tamanhoFator2Texto = 0.86;
posicaoFator1Botao = 0.5;
tamanhoFator1Botao = 0.10;

tamanhoTexto = [ tamanhoFator1Texto.*tamanhoFigura(3)...
    20];
tamanhoBotao = [ tamanhoFator1Botao/length(tiposBotao).*tamanhoFigura(3)...
    20];
posicaoTexto(1)=posicaoFator1Texto.*tamanhoFigura(3);
posicaoTexto(2)=(posicaoFator2Texto+tamanhoFator2Texto).*tamanhoFigura(4);

% Captura nomes
nomesNivel2 = fieldnames(saidas.(nomesNivel1){k});
botoesNumero = length (nomesNivel2);

% Check do tipo do eixo
[eh2d,eh3d,ehmap] = check_eixos(eixos,nomesNivel1);

% Gerando botoes
for i=1:botoesNumero
    if i==30 % Quebrando em 2 colunas
        posicaoFator1Botao = posicaoFator1Botao + tamanhoFator1Botao...
            + tamanhoFator1Texto;
        posicaoTexto(1)=(posicaoFator1Texto+tamanhoFator1Texto+tamanhoFator1Botao)...
            .*tamanhoFigura(3);
        posicaoTexto(2)=(posicaoFator2Texto+tamanhoFator2Texto)...
            .*tamanhoFigura(4);
    end
    posicaoTexto(2)=posicaoTexto(2)-tamanhoTexto(2);

    if isfield(nomes,nomesNivel2{i})
        texto = nomes.(nomesNivel2{i});
    else
        texto = nomesNivel2{i};
    end
    
    uicontrol(gcf,'style','text','HorizontalAlignment','left',...
        'position',[posicaoTexto tamanhoTexto],'string',...
        texto,'UserData','menu3')

    for j=1:length(tiposBotao)
        posicaoBotao(1)=posicaoFator1Botao.*tamanhoFigura(3)+tamanhoBotao(1).*(j-1);
        posicaoBotao(2)=posicaoTexto(2);
        tipo = 'pushbutton';
        status = 'off';
        if     j == 1 && (eh2d||eh3d)
            tipo = 'pushbutton';
            status = 'on';
        elseif j == 2 && eh3d
            tipo = 'pushbutton';
            status = 'on';
        elseif j == 3 && eh3d
            tipo = 'togglebutton';
            status = 'on';
        elseif j == 4 && (eh3d||ehmap)
            tipo = 'pushbutton';
            status = 'on';
        end
        uicontrol(gcf,'style',tipo,...
            'position',[posicaoBotao tamanhoBotao],'string',...
            tiposBotao{j},'enable',status,'UserData','menu3',...
            'Callback',{@callback1, nomesNivel1,k,nomesNivel2{i},j});
    end
end    


function gera_menubar
%Gera menu de comandos
%
% Determina as ações do menu do simulador

% Apagando menus antigos
delete(findobj('type','menubar'))

% Gerando menu principal
h_1(1)=uimenu('Label','Topologia');
h_1(2)=uimenu('Label','Gráficos Especiais');
h_1(3)=uimenu('Label','Animações');

% Gerando submenus
if 1==2,plota_resultados(1), end
    
h_2(1)  = uimenu(h_1(1),'Label','Topologia 1','Callback','plota_resultados(1)');
h_2(2)  = uimenu(h_1(1),'Label','Topologia 2','Callback','plota_resultados(2)');
h_2(3)  = uimenu(h_1(2),'Label','Topologia','Callback','plota_resultados(6)');
h_2(4)  = uimenu(h_1(2),'Label','Radiografia','Callback','plota_resultados(3)');
h_2(5)  = uimenu(h_1(2),'Label','Radiografia (todos)','Callback','plota_resultados(7)');
h_2(6)  = uimenu(h_1(3),'Label','Topologia','Callback','plota_resultados(5)');
h_2(7)  = uimenu(h_1(3),'Label','Radiografia','Callback','plota_resultados(4)');
h_2(8)  = uimenu(h_1(3),'Label','Radiografia (todos)','Callback','plota_resultados(8)');

% TODO % Gerar descrição de help about
% TODO % Gerar help
set([h_1(:);h_2(:)],'userdata','menubar')


function callback1(src,eventdata,nomesNivel1,k, nomesNivel2,j)

if exist('nomesNivel1','var'), tipo = 1; end
if exist('k','var'),           tipo = 2; end
if exist('nomesNivel2','var'), tipo = 3; end

switch tipo
    case 1
        gera_menu2(nomesNivel1)
        gera_menu3(nomesNivel1)
    case 2
        gera_menu3(nomesNivel1,k)
    case 3
        plotador(k, nomesNivel1, nomesNivel2, j)
end
        

function plotador(k, nomesNivel1, nomesNivel2,j)
global nomes eixos saidas

valores = saidas.(nomesNivel1){k}.(nomesNivel2);

[eh2d,eh3d] = check_eixos(eixos,nomesNivel1);

titulo = [nomes.(nomesNivel1) ' - ' num2str(k)];

if j==1
    if eh2d
        y = valores(:);
        nomeX = eixos.(nomesNivel1).X.nome;
        x = eixos.(nomesNivel1).X.valor{k};
        if isfield(nomes,nomesNivel2)
            nomeY = nomes.(nomesNivel2);
        else
            nomeY = nomesNivel2;
        end
        if ~isempty(x)&~isempty(y),
            figure, h =plot(x,y,'o-');
            ht = title([titulo]);
            he(1) = xlabel(nomeX);
            he(2) = ylabel(nomeY);
            set(h,'Linewidth',2)
            set(ht, 'fontsize', 14,'FontWeight','bold')
            set(he, 'fontsize', 12)
            set(gca,'fontsize',12)
        end
    elseif eh3d
        y = valores(1,:);
        nomeX = eixos.(nomesNivel1).Y.nome;
        dados.x2= eixos.(nomesNivel1).X.valor{k};
        dados.nomeX2= eixos.(nomesNivel1).X.nome;
        x = eixos.(nomesNivel1).Y.valor{k};
        if isfield(nomes,nomesNivel2)
            nomeY = nomes.(nomesNivel2);
        else
            nomeY = nomesNivel2;
        end
        dados.x = x;

        dados.valores = valores;
        dados.status = 1;
        dados.titulo = titulo;
        titulo2=[': ' dados.nomeX2 ' = ' num2str(dados.x2(1))];
        if ~isempty(x)&~isempty(y),
            figure, h =plot(x,y,'o-');
            ht = title([titulo titulo2]);
            set(gca,'Userdata',dados)
            he(1) = xlabel(nomeX);
            he(2) = ylabel(nomeY);
            set(h,'Linewidth',2)
            set(ht, 'fontsize', 14,'FontWeight','bold')
            set(he, 'fontsize', 12)
            set(gca,'fontsize',12)
            gera_acao_grafico(j)
        end
    end
    
elseif j==2
    
    if eh2d
        x = eixos.(nomesNivel1).X.valor{k};
        y = valores(:);
        nomeX = eixos.(nomesNivel1).X.nome;
        if isfield(nomes,nomesNivel2)
            nomeY = nomes.(nomesNivel2);
        else
            nomeY = nomesNivel2;
        end
        if ~isempty(x)&~isempty(y),        
            figure, h =plot(x,y);
            ht = title([nomes.(nomesNivel1) ' - ' num2str(k)]);
            he(1) = xlabel(nomeX);
            he(2) = ylabel(nomeY);
            set(h,'Linewidth',2)
            set(ht, 'fontsize', 14,'FontWeight','bold')
            set(he, 'fontsize', 12)
            set(gca,'fontsize',12)
        end
    elseif eh3d
        x = eixos.(nomesNivel1).X.valor{k};
        y = valores(:,1);
        nomeX = eixos.(nomesNivel1).X.nome;
        if isfield(nomes,nomesNivel2)
            nomeY = nomes.(nomesNivel2);        
        else
            nomeY = nomesNivel2;
        end
        dados.x = x;
        dados.x2= eixos.(nomesNivel1).Y.valor{k};
        dados.nomeX2= eixos.(nomesNivel1).Y.nome;
        dados.valores = valores;
        dados.status = 1;
        dados.titulo = titulo;
        if ~isempty(x)&~isempty(y),        
            figure, h =plot(x,y);
            ht = title([nomes.(nomesNivel1) ' - ' num2str(k)]);
            he(1) = xlabel(nomeX);
            he(2) = ylabel(nomeY);
            set(h,'Linewidth',2)
            set(ht, 'fontsize', 14,'FontWeight','bold')
            set(he, 'fontsize', 12)
            set(gca,'Userdata',dados)
            set(gca,'fontsize',12)
            gera_acao_grafico(j)
        end
    end

    
elseif j==3
        
    htoggle = findall(gcf,'style','togglebutton');
    hativo = findobj(htoggle,'string','ZZ','value',1);
    hfig = findall(0,'userdata','fig');
    haxes = findall(hfig,'type','axes');
    dados = get(haxes,'Userdata');
    dados.x2= eixos.(nomesNivel1).Y.valor{k};
    dados.nomeX2= eixos.(nomesNivel1).Y.nome;
    
    if length(hativo) == 1
        
        if eh2d
            y = valores(:);
            x = y;
            if isfield(nomes,nomesNivel2)
                nomeY = nomes.(nomesNivel2);
            else
                nomeY = nomesNivel2;
            end
            nomeX = nomeY;
            dados =[];
        elseif eh3d
            y = valores(:,1);
            x = y;
            if isfield(nomes,nomesNivel2)
                nomeY = nomes.(nomesNivel2);
            else
                nomeY = nomesNivel2;
            end
            nomeX = nomeY;
            dados.valoresx = valores;
            dados.valores = valores;
            dados.status = 1;
        end
        if ~isempty(x)&~isempty(y),
            figure('userdata','fig')
            h =plot(x,y);
            ht = title(['Comparação para ' dados.nomeX2 ' = ' num2str(dados.x2(1))]);
            he(1) = xlabel(nomeX);
            he(2) = ylabel(nomeY);
            set(h,'Linewidth',2)
            set(ht, 'fontsize', 14,'FontWeight','bold')
            set(he, 'fontsize', 12)
            set(gca,'fontsize',12)

            if ~isempty (dados)
                gera_acao_grafico(j)
            end

            set(gca,'Userdata',dados)
        end
    end
    if length(hativo) == 2

        set(hativo,'value',0) 
        
        if ~eh3d
            y = valores(:);
            if isfield(nomes,nomesNivel2)
                nomeY = nomes.(nomesNivel2);
            else
                nomeY = nomesNivel2;
            end
        else
            y = valores(:,1);
            if isfield(nomes,nomesNivel2)
                nomeY = nomes.(nomesNivel2);
            else
                nomeY = nomesNivel2;
            end
            dados.valores = valores;
        end        
        
        hgraf = get(haxes,'children');
        heixoy = get(haxes,'Ylabel');
                
        set(hgraf,'YData',y)
        set(heixoy,'string',nomeY)
        set(haxes,'Userdata',dados)
    end
       
    
elseif j==4
    
    x = eixos.(nomesNivel1).X.valor{k};
    y = eixos.(nomesNivel1).Y.valor{k};
    nx = length(x);
    ny = length(y);
    x = x(:)*ones(1,ny);
    y = [y(:)*ones(1,nx)]';
    if isfield(nomes,nomesNivel2)
        nomeY = nomes.(nomesNivel2);
    else
        nomeY = nomesNivel2;
    end
    
    if ~isempty(x)&~isempty(y),    
        [valores]=filtra_z(valores);
        figure, surfc(x, y, valores);
        ht = title([nomes.(nomesNivel1) ' - ' num2str(k)]);
        he(1) = xlabel(eixos.(nomesNivel1).X.nome);
        he(2) = ylabel(eixos.(nomesNivel1).Y.nome);
        he(3) = zlabel(nomeY);
        shading interp
        set(ht, 'fontsize', 14,'FontWeight','bold')
        set(he, 'fontsize', 12)
        set(gca,'fontsize',12)
    end    
end

function [valores]=filtra_z(valores)
ix=find(imag(valores)~=0);
valores(ix)=nan;

function [eh2d,eh3d,ehmap] = check_eixos(eixos,nomesNivel1)
eh3d  = 0;
eh2d  = 0;
ehmap = 0;

if strcmp(eixos.(nomesNivel1).estilo,'3D')
    eh3d = 1;
end
if strcmp(eixos.(nomesNivel1).estilo,'2D')    
    eh2d = 1;
end
if strcmp(eixos.(nomesNivel1).estilo,'Map')    
    ehmap = 1;
end

function [ tiposBotao ] = tipos_botao
tiposBotao = {'XZ','YZ','ZZ','3D'};


function gera_acao_grafico(j)
uicontrol(gcf,'style','pushbutton','string','<<<',...
    'position',[20 5 30 20],'callback',{@callbackPlot,1,j});
uicontrol(gcf,'style','pushbutton','string',' <<',...
    'position',[50 5 30 20],'callback',{@callbackPlot,5,j});
uicontrol(gcf,'style','pushbutton','string','  <',...
    'position',[80 5 30 20],'callback',{@callbackPlot,2,j});


uicontrol(gcf,'style','pushbutton','string','>  ',...
    'position',[110 5 30 20],'callback',{@callbackPlot,3,j});
uicontrol(gcf,'style','pushbutton','string','>> ',...
    'position',[140 5 30 20],'callback',{@callbackPlot,6,j});
uicontrol(gcf,'style','pushbutton','string','>>>',...
    'position',[170 5 30 20],'callback',{@callbackPlot,4,j});

uicontrol(gcf,'style','pushbutton','string','RR',...
    'position',[400 5 40 20],'callback',{@callbackPlot,10,j});
uicontrol(gcf,'style','pushbutton','string','PLAY',...
    'position',[440 5 40 20],'callback',{@callbackPlot,7,j});
uicontrol(gcf,'style','pushbutton','string','FF',...
    'position',[480 5 40 20],'callback',{@callbackPlot,8,j});
uicontrol(gcf,'style','pushbutton','string','PAUSE',...
    'position',[520 5 40 20],'callback',{@callbackPlot,12,j});
uicontrol(gcf,'style','pushbutton','string','fit',...
    'position',[20 25 30 40],'callback',{@callbackPlot,11,j});



% TODO - Os gráficos 2D estão com eixo x considerando Nk quando deveria ser
% Nk+1 assim perde-se a informação da última célula

function callbackPlot(src,eventdata,flag,j)
global parar roda

hAxe = findall(gcf,'type','axes');

dados = get(hAxe,'userdata');

if flag==10|flag==7|flag==8,
    parar=0;
end

roda=1;
status = dados.status;
while roda

    if parar, break,end
    
    switch flag
        case 1
            status = 1;
            roda = 0;
        case 2
            status = max(1,dados.status - 1);
            roda = 0;
        case 3
            status = min(length(dados.x2),dados.status + 1);
            roda = 0;
        case 4
            status = length(dados.x2);
            roda = 0;
        case 5
            status = max(1,dados.status - round(length(dados.x2)./10));
            roda = 0;
        case 6
            status = min(length(dados.x2),dados.status + round(length(dados.x2)./10));
            roda = 0;
        case 7
            status = status+1;
            if status<length(dados.x2) 
                roda = 1;
            else
                roda =0;
                status = length(dados.x2);
            end
        case 8
            status = status+5;
            if status<length(dados.x2) 
                roda = 1;
            else
                status = length(dados.x2);
                roda =0;
            end
        case 10
            status = status-5;
            if status>1
                roda = 1;
            else
                status = 1;
                roda =0;
            end
    end
    
    dados.status = status;
    set(hAxe,'Userdata',dados)
    valorX2 = dados.x2(status);
    if j==1
        titulo = dados.titulo;
        titulo2=[': ' dados.nomeX2 ' = ' sprintf('%8.5g',valorX2)];
        y = dados.valores(status,:);
        x = get(get(hAxe,'children'),'XData');
    elseif j==2
        titulo = dados.titulo;
        titulo2=[': ' dados.nomeX2 ' = ' sprintf('%8.5g',valorX2)];
        y = dados.valores(:,status);
        x = get(get(hAxe,'children'),'XData');
    elseif j==3
        titulo = [];
        titulo2 = ['Comparação para ' dados.nomeX2 ' = '  sprintf('%8.5g',dados.x2(status))];
        y = dados.valores(:,status);
        x = dados.valoresx(:,status);
    end
    y = y(:)';
    x = x(:)';

    set(get(hAxe,'children'),'YData',y)
    set(get(hAxe,'children'),'XData',x)
    set(get(hAxe,'title'),'string',[titulo titulo2])
    ylim([min(min(dados.valores)) max(max(dados.valores))])
    if isfield(dados,'valoresx')
        xlim([min(min(dados.valoresx)) max(max(dados.valoresx))])
    end
    drawnow
    if flag == 11
        ylim([min(min(y)) max(max(y))])
    end
end

parar=0;