function mfreg
% Interface gráfica para análise dos padrões de escoamento.

warning off
global dadosPadroes detalhesPadroes

% Criando Arvore
aux=cd;
aux=[aux '\Analises'];
addpath(aux)

% Apresentando parametros do banner
parametrosBanner.posicao = [0.02, 0.94, 0.96, 0.6];
parametrosBanner.nomeModulo = 'ANALISE DO ESCOAMENTO';
parametrosBanner.cor = [0 0.5 0.5];
parametrosBanner.corFonte = [0 0 0];
parametrosBanner.nomeSoftware = 'SDMPFLOW';

% Apresentando parametros da área
parametrosArea.posicao = [0.02, 0.02, 0.96, 0.92];
parametrosArea.nomeFrame = 'Dados de Entrada';
parametrosArea.cor = [0.8 0.8 0.8];
parametrosArea.corFonte = [0 0 0];

% Apresentando parametros do menu
parametrosMenu.posicao = [0.1, 0.1, 0.8, 0.15];
parametrosMenu.nomeModulo = ...
    {'EXECUTAR'};
parametrosMenu.fontSize = 10; 

% Gerando tela
nome = 'PADROES';
hfig = findall(0,'name',nome);
if isempty(hfig)
    gera_figura(nome,10);
    gera_banner(gcf, parametrosBanner)
    gera_area(gcf, parametrosArea)
    gera_lista('padrao')
    menu(parametrosMenu)
    
else
    figure(hfig(1))
end


function analisaRegimesEstavel(dados)
% Função que executa o CASO B

% Capturando dados
dados = dados.padrao;

% Desempacotando variaveis dos dutos
parametros.angulo_teta = dados.angulo_teta./180.*pi;
parametros.D           = dados.diametro;
parametros.rugo        = dados.rugosidade;
parametros.tensao      = dados.tensao;
parametros.densL       = dados.densidadeL;
parametros.densG       = dados.densidadeG;
parametros.viscL       = dados.viscosidadeL;
parametros.viscG       = dados.viscosidadeG;

Area = pi.*parametros.D.^2./4;

np = dados.np; % Numero de pontos

% Vetores de mapeamento
vg=10.^linspace(log10(dados.vGMin),log10(dados.vGMax),np);
vl=10.^linspace(log10(dados.vLMin),log10(dados.vLMax),np);
VG = vg(:)*ones(1,np);
VL = ones(np,1)*(vl(:)');

% Estados
entradas.v_LS = VL;
entradas.v_GS = VG;

%Parametros Duto
entradas.D     = dados.diametro.*ones(size(VL));
entradas.rugo  = dados.rugosidade;
entradas.angulo_teta = dados.angulo_teta./180.*pi.*ones(size(VL));

% Parametros Fluido
entradas.densL = dados.densidadeL.*ones(size(VL));
entradas.densG = dados.densidadeG.*ones(size(VL));
entradas.viscL = dados.viscosidadeL.*ones(size(VL));
entradas.viscG = dados.viscosidadeG.*ones(size(VL));
entradas.tensao = dados.tensao;

[hD_est,hD_anu,alfaG_est,alfaG_anu,alfaG_bol]=determinaAlfaEstavel(entradas);
entradas.alfaG_est = alfaG_est;
entradas.alfaG_anu = alfaG_anu;
entradas.alfaG_bol = alfaG_bol;
entradas.hD_est    = hD_est;
[fl, reg1] = identificaRegimeEstavel(entradas);

% Gerando Números de Froude
FL = (parametros.densL./(parametros.densL-parametros.densG)./...
    9.81./parametros.D).^0.5.*VL;
FG = (parametros.densG./(parametros.densL-parametros.densG)./...
    9.81./parametros.D).^0.5.*VG;

% Plotando regimes em função das velocidades
figure
CMAP = [0 0.7 0.25;0 0.3 0.6;0 0.3 0.9;0.5 0.5 0.5];
surf(VG,VL,real(reg1))
axis tight
set(gca,'View', [0 90],'CLim',[1 4])
colormap(CMAP)
a=colorbar;
set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
shading flat
set(gca,'YScale','log','XScale','log')
xlabel('Vel. Sup. Gás (m/s)','FontSize',12)
ylabel('Vel. Sup. Líquido (m/s)','FontSize',12)
title('Flowmap','Fontsize',14)
xlim([dados.vGMin dados.vGMax])
ylim([dados.vLMin dados.vLMax])

figure
CMAP = [0 0.7 0.25;0 0.3 0.6;0 0.3 0.9;0.5 0.5 0.5];
surf(VG.*Area.*parametros.densG,VL.*Area.*parametros.densL,real(reg1))
axis tight
set(gca,'View', [0 90],'CLim',[1 4])
colormap(CMAP)
a=colorbar;
set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
shading flat
set(gca,'YScale','log','XScale','log')
xlabel('Vazão Gás (kg/s)','FontSize',12)
ylabel('Vazão Líquido (kg/s)','FontSize',12)
title('Flowmap','Fontsize',14)

% Plotando regimes em função dos Numeros de Froude
figure
CMAP = [0 0.7 0.25;0 0.3 0.6;0 0.3 0.9;0.5 0.5 0.5];
surf(FG,FL,real(reg1))
axis tight
set(gca,'View', [0 90],'CLim',[1 4])
colormap(CMAP)
a=colorbar;
set(a,'YLim',[1 4],'YTickLabel',['             ';'Anular       ';'Bolhas       ';'             ';'Intermitente ';'Estratificado'])
shading flat
set(gca,'YScale','log','XScale','log')
xlabel('Froude Gás (-)','FontSize',12)
ylabel('Froude Líquido (-)','FontSize',12)
title('Flowmap','Fontsize',14)
FLmin = min(min(FL));
FGmin = min(min(FG));
FLmax = max(max(FL));
FGmax = max(max(FG));
xlim([FGmin FGmax])
ylim([FLmin FLmax])

figure
surf(VL,VG,alfaG_est)
xlabel('Vel. Sup. Líquido (m/s)','Fontsize',12)
ylabel('Vel. Sup. Gás (m/s)','Fontsize',12)
zlabel('Alfa_G Est. (-)','Fontsize',12)
title('Fração de área de gás (estratificado)','Fontsize',14)
shading interp

figure
surf(VL,VG,alfaG_anu)
xlabel('Vel. Sup. Líquido (m/s)','Fontsize',12)
ylabel('Vel. Sup. Gás (m/s)','Fontsize',12)
zlabel('Alfa_G Anular (-)','Fontsize',12)
title('Fração de área de gás (anular)','Fontsize',14)
shading interp

figure
surf(VL,VG,alfaG_bol)
xlabel('Vel. Sup. Líquido (m/s)','Fontsize',12)
ylabel('Vel. Sup. Gás (m/s)','Fontsize',12)
zlabel('Alfa_G Bolhas (-)','Fontsize',12)
title('Fração de área de gás (bolhas)','Fontsize',14)
shading interp


%% Funçoes Auxiliares

function menu(parametros)
% Function Auxiliar que gera o menu
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
        'UserData','menu1','FontSize',fontSize,...
        'Callback',{@callback2,i});
end

function callback2(src,eventdata,arg1)
% Function Auxiliar que gera os Callbacks
global dadosPadroes
switch arg1
    case 1
        armazena_dados
        gera_lista('padrao')
        analisaRegimesEstavel(dadosPadroes);
end

function h = gera_figura(nome,pos)
% Function Auxiliar que gera a figura
if nargin==0, nome = ''; end
if nargin<2,  pos = [] ;end
fig_= zeros(10,4);
y = .45;                        %   screen dependent constant
fig_(1,:) = [0,  y, .5,   y];   %   upper left
fig_(2,:) = [.5, y, .5,   y];   %   upper right
fig_(3,:) = [0,  0, .5,   y];   %   lower left
fig_(4,:) = [.5, 0, .5,   y];   %   lower right
fig_(5,:) = [0,  0.06,  1, 2*y];   %   full screen
fig_(6,:) = [0,  0, .5, 2*y];   %   left  half screen
fig_(7,:) = [.5, 0, .5, 2*y];   %   right half screen
fig_(8,:) = [0,  y,  1,   y];   %   upper half screen
fig_(9,:) = [0,  0,  1,   y];   %   lower half screen
fig_(10,:) = [.25, 0.1*y, .5, 2*(0.9*y)];   %   center half screen
h=[];
if length(pos)==1,     pos = fig_(pos,:); end
if isempty(findall(0,'name',nome))
    units = get(0,'Units');
    h = figure('name',nome,'numbertitle','off',...
    'color',[1 1 1],'menubar','none','Units','normalized',...
    'toolbar','none');
    if ~isempty(pos)
        set(h,'Position',pos)
    end
    set(h,'Units',units)
end

function gera_area(hfig, parametros)
% Function Auxiliar que gera a area
posicaoFator1 = parametros.posicao(1);
posicaoFator2 = parametros.posicao(2);
tamanhoFator1 = parametros.posicao(3);
tamanhoFator2 = parametros.posicao(4);
nomeFrame = parametros.nomeFrame;
corFrame = parametros.cor;
corFonte = parametros.corFonte;
tamanhoFigura = get(hfig,'position');
tamanhoFrame = [ tamanhoFator1.*tamanhoFigura(3) ...
    tamanhoFator2.*tamanhoFigura(4) ];
posicaoFrame = [ posicaoFator1.*tamanhoFigura(3) ...
    posicaoFator2.*tamanhoFigura(4) ];
uicontrol(hfig,'style','frame','position',[posicaoFrame tamanhoFrame],...
    'BackGroundColor',corFrame);
uicontrol(hfig,'style','text','FontWeight','bold',...
    'position',[posicaoFrame(1)+1 posicaoFrame(2)+tamanhoFrame(2)-22 ...
    tamanhoFrame(1)-2 20],'string',nomeFrame,'BackGroundColor',corFrame,...
    'ForeGroundColor',corFonte);

function gera_banner(hfig,parametros)
% Function Auxiliar que gera o banner
posicaoFator1 = parametros.posicao(1);
posicaoFator2 = parametros.posicao(2);
tamanhoFator1 = parametros.posicao(3);
tamanhoFator2 = parametros.posicao(4);
nomeModulo = parametros.nomeModulo;
corFrame = parametros.cor;
corFonte = parametros.corFonte;
nomeSoftware = parametros.nomeSoftware;
tamanhoFigura = get(hfig,'position');
tamanhoFrame = [ tamanhoFator1.*tamanhoFigura(3) 10 ];
posicaoFrame = [ posicaoFator1.*tamanhoFigura(3) ...
    posicaoFator2.*tamanhoFigura(4)+tamanhoFrame(2)/2 ];
uicontrol(hfig,'style','text','FontSize',20,'FontWeight','bold',...
    'position',[posicaoFrame(1) posicaoFrame(2)+tamanhoFrame(2) ...
    tamanhoFrame(1)/2 30],'string',nomeSoftware,...
    'HorizontalAlignment','left','BackGroundColor',[1 1 1],...
    'ForeGroundColor',corFonte);
uicontrol(hfig,'style','frame','position',[posicaoFrame tamanhoFrame],...
    'BackGroundColor',corFrame);
uicontrol(hfig,'style','text','FontSize',14,'FontWeight','bold',...
    'position',[posicaoFrame(1)+tamanhoFrame(1)/2 posicaoFrame(2)+tamanhoFrame(2) ...
    tamanhoFrame(1)/2 20],'string',nomeModulo,...
    'HorizontalAlignment','right','BackGroundColor',[1 1 1],...
    'ForeGroundColor',corFonte);

function gera_lista(nomeObj)
% Function Auxiliar que gera a lista
global dadosPadroes detalhesPadroes
posicaoFator1 = 0.05;
posicaoFator2 = 0.02;
tamanhoFator2 = 0.92;
tamanhoFigura = get(gcf,'position');
if isempty(detalhesPadroes)
    [detalhesPadroes, dadosPadroes] = arvore_dados;
end
objNames=fieldnames(dadosPadroes.(nomeObj));
objNumero=length(objNames);
for i=1:objNumero
    objValue = dadosPadroes.(nomeObj).(objNames{i});
    objDescricao = detalhesPadroes.(nomeObj).(objNames{i}).descricao;
    objUnidade = detalhesPadroes.(nomeObj).(objNames{i}).unidade;
    objNome = detalhesPadroes.(nomeObj).(objNames{i}).nome;
    objTipo = detalhesPadroes.(nomeObj).(objNames{i}).tipo;
    objTamanho = detalhesPadroes.(nomeObj).(objNames{i}).tamanho;
    if isfield(detalhesPadroes.(nomeObj).(objNames{i}),'opcoes')
        objPop =1;
        objOpcoes = detalhesPadroes.(nomeObj).(objNames{i}).opcoes;
    else
        objPop =0; 
        objOpcoes = '';
    end
    if strcmp(objTipo,'fixo')
        objEnable = 'off';
        objColor = [0.7 0.7 0.7];
    else
        objEnable = 'on';
        objColor = [1 1 1];
    end
    tamanho = [ 200 20 ];
    posicao = [ posicaoFator1.*tamanhoFigura(3) ...
        (posicaoFator2+tamanhoFator2)...
        .*tamanhoFigura(4)-(i+1)*tamanho(2).*1.2 ];
    nomeFrame = '';
    if i==1
        uicontrol(gcf,'style','text','FontWeight','bold',...
            'position',[posicao(1) posicao(2)+tamanho(2) tamanho(1) ...
            tamanho(2)],'string',nomeFrame,...
            'UserData',nomeObj,'BackgroundColor',[0.8 0.8 0.8]);
    end
    h1(i)=uicontrol(gcf,'style','text',...
        'HorizontalAlignment','left',...
        'position',[posicao tamanho],...
        'string',[objNome ' [ ' objUnidade ' ]:'],...
        'UserData',nomeObj);
    if strcmp(objTamanho,'escalar'),
        tamanho = [ 100 20 ];
        posicao(1) = posicao(1) + 200 + 10;    
    elseif strcmp(objTamanho,'vetor'),
        tamanho = [ 150 20 ];
        posicao(1) = posicao(1) + 200 + 10;    
    end
    if objPop,
        h2(i)=uicontrol(gcf,'style','popupmenu','BackGroundColor',objColor,...
            'position',[posicao tamanho],'string',objOpcoes,...
            'Value',objValue,'Enable', objEnable,'UserData',nomeObj);
    else
        h2(i)=uicontrol(gcf,'style','edit','BackGroundColor',objColor,...
            'position',[posicao tamanho],'string',num2str(objValue),...
            'Enable', objEnable,'UserData',nomeObj);
    end
    posicao(1) = posicao(1) + tamanho(1) + 10;
    tamanho = [ 350-tamanho(1) 20 ];    
    h3(i)=uicontrol(gcf,'style','text','HorizontalAlignment','left',...
        'position',[posicao tamanho],'string',objDescricao,...
        'UserData',nomeObj);
    set([h1(i);h3(i)],'BackGroundColor',[0.7 0.7 0.7])
end


%% Funçoes de Especificação e Captura de Dados

function [detalhesVariaveis, valoresDefault] = arvore_dados
%Compositor da estrutura dos dados de entrada.

% Valor padrao
valoresDefault.padrao.densidadeL=860.;
valoresDefault.padrao.densidadeG=90;
valoresDefault.padrao.viscosidadeL=0.008;
valoresDefault.padrao.viscosidadeG=2e-5;
valoresDefault.padrao.tensao=0.03;
valoresDefault.padrao.rugosidade=0;
valoresDefault.padrao.diametro=0.1;
valoresDefault.padrao.angulo_teta=0;
valoresDefault.padrao.np=100;
valoresDefault.padrao.vGMin=0.001;
valoresDefault.padrao.vLMin=0.01;
valoresDefault.padrao.vGMax=50;
valoresDefault.padrao.vLMax=10;

% Detalhes
detalhesVariaveis.padrao.nome = 'Padrão de Escoamento';
detalhesVariaveis.padrao.tamanho = 'escalar';

detalhesVariaveis.padrao.rugosidade.unidade='10^-6 m';
detalhesVariaveis.padrao.rugosidade.nome='Rugosidade';
detalhesVariaveis.padrao.rugosidade.descricao=...
    'Valor da rugosidade na superficie interna do duto.';
detalhesVariaveis.padrao.rugosidade.tipo='usuario';
detalhesVariaveis.padrao.rugosidade.tamanho='escalar';

detalhesVariaveis.padrao.diametro.unidade='m';
detalhesVariaveis.padrao.diametro.nome='Diâmetro';
detalhesVariaveis.padrao.diametro.descricao=...
    'Valor do diametro interno da tubulação.';
detalhesVariaveis.padrao.diametro.tipo='usuario';
detalhesVariaveis.padrao.diametro.tamanho='escalar';

detalhesVariaveis.padrao.angulo_teta.unidade='graus';
detalhesVariaveis.padrao.angulo_teta.nome='Inclinação';
detalhesVariaveis.padrao.angulo_teta.descricao=...
    'Inclinação do duto (+ -> ascendente) [-90 90]';
detalhesVariaveis.padrao.angulo_teta.tipo='usuario';
detalhesVariaveis.padrao.angulo_teta.tamanho='escalar';

detalhesVariaveis.padrao.densidadeL.unidade='kg/m³';
detalhesVariaveis.padrao.densidadeL.nome='Densidade (líquido)';
detalhesVariaveis.padrao.densidadeL.descricao=...
    'Densidade média da fase líquida.';
detalhesVariaveis.padrao.densidadeL.tipo='usuario';
detalhesVariaveis.padrao.densidadeL.tamanho='escalar';

detalhesVariaveis.padrao.densidadeG.unidade='kg/m³';
detalhesVariaveis.padrao.densidadeG.nome='Densidade (gás)';
detalhesVariaveis.padrao.densidadeG.descricao=...
    'Densidade média da fase gasosa.';
detalhesVariaveis.padrao.densidadeG.tipo='usuario';
detalhesVariaveis.padrao.densidadeG.tamanho='escalar';

detalhesVariaveis.padrao.viscosidadeL.unidade='Pa.s';
detalhesVariaveis.padrao.viscosidadeL.nome='Viscosidade Dinâmica (líquido)';
detalhesVariaveis.padrao.viscosidadeL.descricao=...
    'Viscosidade média da fase líquida.';
detalhesVariaveis.padrao.viscosidadeL.tipo='usuario';
detalhesVariaveis.padrao.viscosidadeL.tamanho='escalar';

detalhesVariaveis.padrao.viscosidadeG.unidade='Pa.s';
detalhesVariaveis.padrao.viscosidadeG.nome='Viscosidade Dinâmica (gás)';
detalhesVariaveis.padrao.viscosidadeG.descricao=...
    'Viscosidade média da fase gasosa.';
detalhesVariaveis.padrao.viscosidadeG.tipo='usuario';
detalhesVariaveis.padrao.viscosidadeG.tamanho='escalar';

detalhesVariaveis.padrao.tensao.unidade='Pa.m';
detalhesVariaveis.padrao.tensao.nome='Tensao interfacial';
detalhesVariaveis.padrao.tensao.descricao=...
    'Tensao da interface gás-líquido.';
detalhesVariaveis.padrao.tensao.tipo='usuario';
detalhesVariaveis.padrao.tensao.tamanho='escalar';

detalhesVariaveis.padrao.vGMin.unidade='Pa.m';
detalhesVariaveis.padrao.vGMin.nome='Vel. sup. Mín. Gás';
detalhesVariaveis.padrao.vGMin.descricao=...
    'Velocidade superficial mínima do gás.';
detalhesVariaveis.padrao.vGMin.tipo='usuario';
detalhesVariaveis.padrao.vGMin.tamanho='escalar';

detalhesVariaveis.padrao.vGMax.unidade='Pa.m';
detalhesVariaveis.padrao.vGMax.nome='Vel. sup. Máx. Gás';
detalhesVariaveis.padrao.vGMax.descricao=...
    'Velocidade superficial máxima do gás.';
detalhesVariaveis.padrao.vGMax.tipo='usuario';
detalhesVariaveis.padrao.vGMax.tamanho='escalar';

detalhesVariaveis.padrao.vLMin.unidade='m/s';
detalhesVariaveis.padrao.vLMin.nome='Vel. sup. Mín. Líquido';
detalhesVariaveis.padrao.vLMin.descricao=...
    'Velocidade superficial mínima do líquido.';
detalhesVariaveis.padrao.vLMin.tipo='usuario';
detalhesVariaveis.padrao.vLMin.tamanho='escalar';

detalhesVariaveis.padrao.vLMax.unidade='m/s';
detalhesVariaveis.padrao.vLMax.nome='Vel. sup. Máx. Líquido';
detalhesVariaveis.padrao.vLMax.descricao=...
    'Velocidade superficial máxima do líquido.';
detalhesVariaveis.padrao.vLMax.tipo='usuario';
detalhesVariaveis.padrao.vLMax.tamanho='escalar';

detalhesVariaveis.padrao.np.unidade='-';
detalhesVariaveis.padrao.np.nome='Numero de pontos';
detalhesVariaveis.padrao.np.descricao=...
    'Numero de pontos de pesquisa.';
detalhesVariaveis.padrao.np.tipo='usuario';
detalhesVariaveis.padrao.np.tamanho='escalar';

function armazena_dados
% Function Auxiliar que executa o armazenamento das variaveis
global dadosPadroes
% Gerando lista de niveis possiveis
ListaNomeObj = fieldnames(dadosPadroes);
% Executando loop para determinar qual lista esta disponivel
nomeObj = [];
for i=1:length(ListaNomeObj)
    nomeObjTeste = ListaNomeObj{i};
    hObjTodos = findall(gcf,'userdata',nomeObjTeste);
    if ~isempty(hObjTodos)
        nomeObj = nomeObjTeste;
    end
end    
if ~isempty(nomeObj)
    % Gerando vetores de handles
    hObjTodos = findall(gcf,'userdata',nomeObj);
    hObjEdit = findall(gcf,'userdata',nomeObj,'style','edit');
    hObjPop = findall(gcf,'userdata',nomeObj,'style','popupmenu');
    hObj = sort([hObjEdit;hObjPop]);
    hObjNumero = length(hObj);
    % Determinando o tipo de objeto
    ehCell = iscell(dadosPadroes.(nomeObj));
    if ehCell 
        objNames=fieldnames(dadosPadroes.(nomeObj){1});
    else
        objNames=fieldnames(dadosPadroes.(nomeObj));
    end
    % Capturando informações da lista
    j =1;
    for i=1:hObjNumero
        if strcmp(get(hObj(i),'Style'),'edit')
            valores.(objNames{i})=str2num(get(hObj(i),'String'));
            j=j+1;
        end
        if strcmp(get(hObj(i),'Style'),'popupmenu')
            valores.(objNames{j})=get(hObj(i),'value');
            j=j+1;
        end
    end
    % Atualizando dados
    if ehCell
        k=valores.indice;
        dadosPadroes.(nomeObj){k}=valores;
    else
        dadosPadroes.(nomeObj)=valores;
    end
    % Limpando objetos
    delete(hObjTodos)
end