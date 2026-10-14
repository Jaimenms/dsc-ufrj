function mfsim_entrada
% Interface gráfica para simulação de redes de escoamento bifásico.

global dados detalhesVariaveis nomes saidas estados valoresDefault

% Capturando estrutura de dados e saidas    
[detalhesVariaveis, valoresDefault] = arvore_dados;

if isempty(dados)
    dados = coloca_dados(dados,valoresDefault);
end
if isempty(nomes)
    nomes = arvore_saidas(saidas,nomes);
end

% Apresentando parametros do banner
parametrosBanner.posicao = [0.02, 0.94, 0.96, 0.6];
parametrosBanner.nomeModulo = 'ENTRADA DE DADOS';
parametrosBanner.cor = [0 0.5 0.5];
parametrosBanner.corFonte = [0 0 0];
parametrosBanner.nomeSoftware = 'SDMPFLOW';

% Apresentando parametros da área 1
parametrosArea1.posicao = [0.02, 0.02, 0.10, 0.92];
parametrosArea1.nomeFrame = 'ELEMENTOS';
parametrosArea1.cor = [0 0.5 0.5];
parametrosArea1.corFonte = [1 1 1];

% Apresentando parametros da área 2
parametrosArea2.posicao = [0.14, 0.02, 0.10, 0.92];
parametrosArea2.nomeFrame = 'OBJETOS';
parametrosArea2.cor = [0 0.5 0.5];
parametrosArea2.corFonte = [1 1 1];

% Apresentando parametros da área 3
parametrosArea3.posicao = [0.26, 0.02, 0.72, 0.92];
parametrosArea3.nomeFrame = '';
parametrosArea3.cor = [0.8 0.8 0.8];
parametrosArea3.corFonte = [0 0 0];

% Gerando tela
nome = 'ENTRADA';
hfig = findall(0,'name',nome);
if isempty(hfig)
    gera_figura(nome,5);
    gera_banner(gcf, parametrosBanner)
    gera_area(gcf, parametrosArea1)
    gera_area(gcf, parametrosArea2)
    gera_area(gcf, parametrosArea3)
    gera_menu1
    gera_menu2([])
    gera_lista([],[])
    gera_menubar
else
    figure(hfig(1))
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

global dados detalhesVariaveis

% Capturando parametros
posicaoFator1 = 0.03;
posicaoFator2 = 0.05;
tamanhoFator1 = 0.08;
tamanhoFator2 = 0.86;

% Limpando Objetos
limpa_objetos({'menu1'});

% Posicionando Objetos
tamanhoFigura = get(gcf,'position');
tamanhoBotoes = [ tamanhoFator1.*tamanhoFigura(3) 20];

% Capturando niveis possiveis dos dados
nomesNivel= fieldnames(dados);
botoesNumero = length (nomesNivel);

% Gerando botoes
for i=1:botoesNumero
    if isfield(detalhesVariaveis,(nomesNivel{i}))
        botaoNome = detalhesVariaveis.(nomesNivel{i}).nome;
    else
        botaoNome = nomesNivel{i};
    end
    nomeObj=nomesNivel{i};
    k = iscell(dados.(nomesNivel{i}));
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',botaoNome,...
        'UserData','menu1','Callback',{@callback1, nomeObj,-k});
end

function gera_menu2(nomeObj)
%Menu com os botões do menu secundário.
%
% Essa subfunction recebe os parametros de configuração de um struct e gera
% um conjunto de botoes devidamente posicionados na forma de um menu.
% Essa subfunctionn trabalha associada com uma subfunction de callback.
% NOMEOBJ é um string do nome do campo.
% DADOS é o struct com a arvore de dados.
% DETALHESCARIAVEIS é o struct com os detalhes da arvore de dados.

global dados detalhesVariaveis

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

% Ativando o nome primário
if isempty(nomeObj)
   nomeObjaux = fieldnames(dados);
   nomeObj = nomeObjaux{1};
end

% Calcula numero de botoes
if iscell(dados.(nomeObj)),
    botoesNumero = length (dados.(nomeObj));
else
    botoesNumero = 0;
end

% Gera objetos
for i=1:botoesNumero
    if ~isempty(dados.(nomeObj){i})
        nomeBotao = [detalhesVariaveis.(nomeObj).nome ' ' num2str(i)];
        posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
           (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
        uicontrol(gcf,'style','pushbutton',...
            'position',[posicaoBotoes tamanhoBotoes],'string',nomeBotao,...
            'UserData','menu2','Callback',{@callback1, nomeObj, i});
    end
end
if iscell(dados.(nomeObj))
    if isempty(dados.(nomeObj){i}), i = i-1; end
end

if botoesNumero
    % Gera objetos de comando
    i=i+1;
    nomeBotao = 'Adicionar';
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',nomeBotao,...
        'UserData','menu3','Callback',{@callback1, nomeObj,[]});
    i=i+1;
    nomeBotao = 'Excluir';
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',nomeBotao,...
        'UserData','menu3','Callback',{@callback1, nomeObj,[]});
    i=i+1;
    nomeBotao = 'Salvar';
    posicaoBotoes = [ posicaoFator1.*tamanhoFigura(3) ...
       (posicaoFator2+tamanhoFator2).*tamanhoFigura(4)-i*tamanhoBotoes(2)];
    uicontrol(gcf,'style','pushbutton',...
        'position',[posicaoBotoes tamanhoBotoes],'string',nomeBotao,...
        'UserData','menu3','Callback',{@callback1, nomeObj,[]});
end

function gera_menubar
%Gera menu de comandos
%
% Determina as ações do menu do simulador

% Apagando menus antigos
delete(findobj('type','menubar'))

% Gerando menu principal
h_1(1)=uimenu('Label','Arquivo');
h_1(2)=uimenu('Label','Executar');
h_1(3)=uimenu('Label','Resultados');
h_1(4)=uimenu('Label','Analisar');
h_1(5)=uimenu('Label','Ajuda');

% Gerando submenus
h_2(1)  = uimenu(h_1(1),'Label','Abrir Simulacao',             'Callback',{@callback2,1 });
h_2(1)  = uimenu(h_1(1),'Label','Abrir Tudo',                  'Callback',{@callback2,100});
h_2(2)  = uimenu(h_1(1),'Label','Salvar Simulacao',            'Callback',{@callback2,2 });
h_2(2)  = uimenu(h_1(1),'Label','Salvar Tudo',                 'Callback',{@callback2,200});
h_2(4)  = uimenu(h_1(1),'Label','Sair',                        'Callback',{@callback2,4 });
h_2(5)  = uimenu(h_1(2),'Label','Estacionaria Incompresivel',  'Callback',{@callback2,5 });
h_2(6)  = uimenu(h_1(2),'Label','Estacionaria Compressivel',   'Callback',{@callback2,6 });
h_2(6)  = uimenu(h_1(2),'Label','Dinamica',                    'Callback',{@callback2,7 });
h_2(7)  = uimenu(h_1(2),'Label','Completa',                    'Callback',{@callback2,8 });
h_2(71) = uimenu(h_1(2),'Label','Atualizar Coef. Arraste Est.','Callback',{@callback2,15 });
h_2(8)  = uimenu(h_1(5),'Label','Documentação',                'Callback',{@callback2,9 });
h_2(9)  = uimenu(h_1(3),'Label','Calcular',                    'Callback',{@callback2,10});
h_2(10) = uimenu(h_1(3),'Label','Visualizar',                  'Callback',{@callback2,11});
h_2(10) = uimenu(h_1(4),'Label','Padroes de Escoamento',       'Callback',{@callback2,12});
h_2(11) = uimenu(h_1(4),'Label','Matriz Jacobiana das EDOs',   'Callback',{@callback2,13});
h_2(12) = uimenu(h_1(4),'Label','Autovalores da Jacobiana das EDOs',   'Callback',{@callback2,14});

% TODO % Gerar descrição de help about
% TODO % Gerar help
set([h_1(:);h_2(:)],'userdata','menubar')

function gera_lista(nomeObj, numeroBotao)
%Lista com os dados de entrada do objeto NOMEOBJ.
%
% Essa subfunction gera uma lista com todas as variaveis de entrada para
% o objeto selevionado.

global dados detalhesVariaveis

% Captura posicao dos objetos
posicaoFator1 = 0.28;
posicaoFator2 = 0.02;
tamanhoFator2 = 0.92;
tamanhoFigura = get(gcf,'position');

% Gerando lista primária
if isempty(nomeObj)
   nomeObjaux = fieldnames(dados);
   nomeObj = nomeObjaux{1};
   ehCell=iscell(dados.(nomeObj));
   if ehCell
       numeroBotao = 1;
   else
       numeroBotao = 0;
   end
end

% Ajustando tela ao tipo de variavel
if numeroBotao>0 
    if isempty(dados.(nomeObj){numeroBotao})
        objNames=[];
        nomeFrame = ['PARAMETROS - ' detalhesVariaveis.(nomeObj).nome];
    else
        objNames=fieldnames(dados.(nomeObj){numeroBotao});
        nomeFrame = ['PARAMETROS - ' detalhesVariaveis.(nomeObj).nome ' - ' num2str(numeroBotao)];
    end
else
    objNames=fieldnames(dados.(nomeObj));
    nomeFrame = ['PARAMETROS - ' detalhesVariaveis.(nomeObj).nome];
end
objNumero=length(objNames);

% Loop para inserir todos os objetos
for i=1:objNumero
    % Capturando valor
    if numeroBotao>0
        objValue = dados.(nomeObj){numeroBotao}.(objNames{i});
    else
        objValue = dados.(nomeObj).(objNames{i});
    end
    % Capturando detalhes
    if isfield(detalhesVariaveis,(nomeObj))&&isfield(detalhesVariaveis.(nomeObj),objNames{i})
        objNome = detalhesVariaveis.(nomeObj).(objNames{i}).nome;
        objDescricao = detalhesVariaveis.(nomeObj).(objNames{i}).descricao;
        objUnidade = detalhesVariaveis.(nomeObj).(objNames{i}).unidade;
        objTipo = detalhesVariaveis.(nomeObj).(objNames{i}).tipo;
        objTamanho = detalhesVariaveis.(nomeObj).(objNames{i}).tamanho;
        % Verificando tipo de unicontrol
        if isfield(detalhesVariaveis.(nomeObj).(objNames{i}),'opcoes')
            objPop =1;
            objOpcoes = detalhesVariaveis.(nomeObj).(objNames{i}).opcoes;
        else
            objPop =0; 
            objOpcoes = '';
        end
    else
        objNome = objNames{i};
        objDescricao = objNames{i};
        objUnidade = '-';
        objTipo = 'usuario';
        objTamanho = 'escalar';
        objPop =0;
        objOpcoes = '';
    end
            
    % Verificando disponibilidade de unicontrol
    if strcmp(objTipo,'fixo')
        objEnable = 'off';
        objColor = [0.7 0.7 0.7];
    else
        objEnable = 'on';
        objColor = [1 1 1];
    end

    % Posicoes primarias
    tamanho = [ 650 20 ];
    posicao = [ posicaoFator1.*tamanhoFigura(3) ...
        (posicaoFator2+tamanhoFator2)...
        .*tamanhoFigura(4)-(i+1)*tamanho(2).*1.2 ];
    
    % Gerando titulo
    if i==1
        uicontrol(gcf,'style','text','FontWeight','bold',...
            'position',[posicao(1) posicao(2)+tamanho(2) tamanho(1) ...
            tamanho(2)],'string',nomeFrame,...
            'UserData',nomeObj,'BackgroundColor',[0.8 0.8 0.8]);
    end
    
    % Gerando objetos    
    h1(i)=uicontrol(gcf,'style','text',...
        'HorizontalAlignment','left',...
        'position',[posicao tamanho],...
        'string',[objNome ' [ ' objUnidade ' ]:'],...
        'UserData',nomeObj);
    
    % Determinando tamanho da editbox
    if strcmp(objTamanho,'escalar'),
        tamanho = [ 100 20 ];
        posicao(1) = posicao(1) + 200 + 10;    
    elseif strcmp(objTamanho,'vetor'),
        tamanho = [ 300 20 ];
        posicao(1) = posicao(1) + 200 + 10;    
    end
    
    % Inserindo objeto de entrada de dados
    if objPop,
        h2(i)=uicontrol(gcf,'style','popupmenu','BackGroundColor',objColor,...
            'position',[posicao tamanho],'string',objOpcoes,...
            'Value',objValue,'Enable', objEnable,'UserData',nomeObj);
    else
        h2(i)=uicontrol(gcf,'style','edit','BackGroundColor',objColor,...
            'position',[posicao tamanho],'string',num2str(objValue),...
            'Enable', objEnable,'UserData',nomeObj);
    end

    % Gerando objeto de descricao
    posicao(1) = posicao(1) + tamanho(1) + 10;
    tamanho = [ 470-tamanho(1) 20 ];    
    h3(i)=uicontrol(gcf,'style','text','HorizontalAlignment','left',...
        'position',[posicao tamanho],'string',objDescricao,...
        'UserData',nomeObj);
    
    set([h1(i);h3(i)],'BackGroundColor',[0.7 0.7 0.7])

end


function armazena_dados
%Armazenamento dos dados da tela de configuração
%
% Salva no arquivo de DADOS todas as variáveis listadas na tela. 

global dados

% Gerando lista de niveis possiveis
ListaNomeObj = fieldnames(dados);

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
    ehCell = iscell(dados.(nomeObj));
    if ehCell 
        objNames=fieldnames(dados.(nomeObj){1});
    else
        objNames=fieldnames(dados.(nomeObj));
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
        dados.(nomeObj){k}=valores;
    else
        dados.(nomeObj)=valores;
    end

    % Limpando objetos
    delete(hObjTodos)
end


function callback1(src,eventdata,nomeObj,k)
%Callbacks de execuçao dos botoes
%
% Ações ds botões do menu1, menu2 e menu3.

global dados valoresDefault

% Armazenando dados
armazena_dados

% Coletando informacoes dos botoes
tipoBotao = get(gcbo,'Userdata');
nomeBotao = get(gcbo,'String');

% Executa comandos do menu1
if strcmp(tipoBotao,'menu1')
    % Lipando menus dependentes
    limpa_objetos({'menu2','menu3'});
    if k==-1 %Caso objeto de menu1 tenha depedentes
        delete(findall(gcf,'Userdata',nomeObj))
        gera_menu2(nomeObj)
        gera_lista(nomeObj,1)
    else % caso objeto de menu1 não tenha dependentes
        gera_lista(nomeObj,k)
    end
end

% Executa comandos do menu2
if strcmp(tipoBotao,'menu2')
    gera_lista(nomeObj,k)
end

% Executa comandos do menu3
if strcmp(tipoBotao,'menu3')
    %Calcula numero de elementos
    n = length(dados.(nomeObj));
    % Executa ações
    if strcmp(nomeBotao,'Adicionar')
        if isempty(dados.(nomeObj){n}), n=n-1; end
        dados.(nomeObj){n+1}=valoresDefault.(nomeObj){1};
        dados.(nomeObj){n+1}.indice = n+1;
        gera_menu2(nomeObj)
        gera_lista(nomeObj,n+1)
    elseif strcmp(nomeBotao,'Excluir')
        if n>1
            for i=1:n-1
                aux{i} = dados.(nomeObj){i};
            end
            dados.(nomeObj)=aux;
            gera_menu2(nomeObj)
            gera_lista(nomeObj,n-1)
        elseif n==1
            dados.(nomeObj){1}=[];
            gera_menu2(nomeObj)
            gera_lista(nomeObj,1)
        end
    elseif strcmp(nomeBotao,'Salvar')
        armazena_dados
    end
end

function callback2(src,eventdata,arg1)
%Callbacks de execução do menubar.
% 
% Ações do menubar.

%Lista de Callbacks para executar os comandos do menubar
global dados saidas estados eixos detalhesVariaveis

switch arg1
    case 1
        raiz=cd;
        cd files;
        [filename, pathname] = uigetfile({'*.mf1','Arquivos de simulacao (*.mf1)'},'Escolha um arquivo de dados');
        if isequal(filename,0)||isequal(pathname,0),
        else cd(pathname),load(filename,'-mat'),end,
        cd(raiz)
        close(findall(0,'name','ENTRADA'))
        close(findall(0,'name','SAIDA'))
        mfsim_entrada
    case 100
        raiz=cd;
        cd files;
        [filename, pathname] = uigetfile({'*.mf2','Arquivos completos (*.mf2)'},'Escolha um arquivo pacote');
        if isequal(filename,0)||isequal(pathname,0),
        else cd(pathname),load(filename,'-mat'),end,
        cd(raiz)
        close(findall(0,'name','ENTRADA'))
        close(findall(0,'name','SAIDA'))
        mfsim_entrada
    case 2
        armazena_dados
        gera_menu1
        gera_menu2([])
        gera_lista([],[])
        raiz=cd;
        cd files;
        [filename, pathname] = uiputfile('*.mf1','Salvar simulacao');
        if isequal(filename,0)||isequal(pathname,0),
        else
            cd(pathname)
            if sum(filename(1,end-3:end)=='.mf1')==4,
                save(filename,'dados','detalhesVariaveis','estados','-mat')
            else
                save([filename '.mf1'],'dados','detalhesVariaveis','estados','-mat')
            end
        end
        cd(raiz)
    case 200
        armazena_dados
        gera_menu1
        gera_menu2([])
        gera_lista([],[])
        raiz=cd;
        cd files;
        [filename, pathname] = uiputfile('*.mf2','Salvar tudo');
        if isequal(filename,0)||isequal(pathname,0),
        else
            cd(pathname)
            if sum(filename(1,end-3:end)=='.mf2')==4,
                save(filename,'dados','detalhesVariaveis','estados','saidas','eixos','-mat')
            else
                save([filename '.mf2'],'dados','detalhesVariaveis','estados','saidas','eixos','-mat')
            end
        end
        cd(raiz)
    case 4
        close
    case 5
        armazena_dados
        [dados,estados,saidas,eixos] = simulador(dados,estados,1,0,0,0);
        hfig = findall(0,'name','ENTRADA');
        if ~isempty(hfig)
            figure(hfig)
            gera_menu1
            gera_menu2([])
            gera_lista([],[])
        else
            mfsim_entrada
        end
    case 6
        armazena_dados
        [dados,estados,saidas,eixos] = simulador(dados,estados,0,1,0,0);
        hfig = findall(0,'name','ENTRADA');
        if ~isempty(hfig)
            figure(hfig)
            gera_menu1
            gera_menu2([])
            gera_lista([],[])
        else
            mfsim_entrada
        end
    case 7
        armazena_dados
        tic
        [dados,estados,saidas,eixos] = simulador(dados,estados,0,0,1,0);
        t_aux = toc;
        disp(['Tempo de Simulação: ' num2str(t_aux) ' seg.'])
        hfig = findall(0,'name','ENTRADA');
        if ~isempty(hfig)
            figure(hfig)
            gera_menu1
            gera_menu2([])
            gera_lista([],[])
        else
            mfsim_entrada
        end
    case 8
        armazena_dados
        [dados,estados,saidas,eixos] = simulador(dados,estados,1,1,1,0);
        hfig = findall(0,'name','ENTRADA');
        if ~isempty(hfig)
            figure(hfig)
            gera_menu1
            gera_menu2([])
            gera_lista([],[])
        else
            mfsim_entrada
        end
    case 9
        web(['file:' cd '\Docs\index.html'])
    case 10
        [dados,estados,saidas,eixos] = simulador(dados,estados,0,0,0,1);
    case 11
        mfsim_saida;
    case 12
        aux=cd;
        aux=[aux '\Analises'];
        addpath(aux)
        analisaRegimes(dados,estados,saidas);
        
    case 13
        analisador(dados,estados,1)
    case 14
        analisador(dados,estados,2)
    case 15
        armazena_dados        
        [dados] = simulador(dados,estados,0,0,0,0,1);
        hfig = findall(0,'name','ENTRADA');
        figure(hfig)
        gera_menu1
        gera_menu2([])
        gera_lista([],[])

end

function dados2 = coloca_dados(dados,valoresDefault,qual)

if isempty(dados)
    names = fieldnames(valoresDefault);
    for i=1:length(names)
        if ~iscell(valoresDefault.(names{i}))
            dados2.(names{i}) = valoresDefault.(names{i});
        else
            dados2.(names{i}){1} = [];
        end
    end
end

