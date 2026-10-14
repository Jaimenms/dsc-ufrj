function [detalhesVariaveis, valoresDefault] = arvore_dados
% Função que compõe a estrutura dos dados de entrada e a arvore de dados
% contendo as características de todas as informações necessárias para
% executar o simulador

% Valor padrao para o simulador
valoresDefault.simulador.tempoInicial=0;
valoresDefault.simulador.tempoFinal=200;
valoresDefault.simulador.incrementoTempo=2;
valoresDefault.simulador.pressaoR=100e5;
valoresDefault.simulador.tipoOde=4;
valoresDefault.simulador.metodo=1;        
valoresDefault.simulador.maxStep=1;        
valoresDefault.simulador.absTol=1e-1;        
valoresDefault.simulador.relTol=1e-4;        
valoresDefault.simulador.ordemQuadratura=1;
valoresDefault.simulador.tolerancia=1e-6;
valoresDefault.simulador.maxIter=50;
valoresDefault.simulador.delta=1e-6;
valoresDefault.simulador.delta2=1e-10;

% Valor padrao para o fluido
valoresDefault.fluido.pressao0=100e5;
valoresDefault.fluido.densidadeL0=860.;
valoresDefault.fluido.densidadeG0=80.;
valoresDefault.fluido.gamaL=1000;
valoresDefault.fluido.gamaG=1.2;
valoresDefault.fluido.viscosidadeL=0.008;
valoresDefault.fluido.viscosidadeG=2e-5;
valoresDefault.fluido.tensao=0.03;

% Valor padrao para o vertice
valoresDefault.vertice{1}.indice=1;
valoresDefault.vertice{1}.tipo=1;
valoresDefault.vertice{1}.volume=0.1^3;
valoresDefault.vertice{1}.especificacaoT1=[];
valoresDefault.vertice{1}.especificacaoT2=[];
valoresDefault.vertice{1}.especificacaoT3=[0 10 20 100 110 200];
valoresDefault.vertice{1}.especificacaoT4=[];
valoresDefault.vertice{1}.especificacao1=[];
valoresDefault.vertice{1}.especificacao2=[];
valoresDefault.vertice{1}.especificacao3=[1 1 10 10 1 1];
valoresDefault.vertice{1}.especificacao4=[1];
valoresDefault.vertice{1}.condicaoInicial1=[];
valoresDefault.vertice{1}.condicaoInicial2=[];

% Valor padrao para os tubos
valoresDefault.tubo{1}.indice=1;
valoresDefault.tubo{1}.modelo=1;
valoresDefault.tubo{1}.verticeOrigem=1;
valoresDefault.tubo{1}.verticeDestino=2;
valoresDefault.tubo{1}.x=[0 0];
valoresDefault.tubo{1}.y=[0 1000];
valoresDefault.tubo{1}.z=[0 0];
valoresDefault.tubo{1}.regime=4;
valoresDefault.tubo{1}.numeroElementos=[10];
valoresDefault.tubo{1}.rugosidade=1000;
valoresDefault.tubo{1}.diametro=0.1;
valoresDefault.tubo{1}.tipocondicaoContornoE1=1;
valoresDefault.tubo{1}.tipocondicaoContornoE2=1;
valoresDefault.tubo{1}.tipocondicaoContornoE3=2;
valoresDefault.tubo{1}.tipocondicaoContornoE4=2;
valoresDefault.tubo{1}.tipocondicaoContornoS1=1;
valoresDefault.tubo{1}.tipocondicaoContornoS2=2;
valoresDefault.tubo{1}.tipocondicaoContornoS3=1;
valoresDefault.tubo{1}.tipocondicaoContornoS4=1;
valoresDefault.tubo{1}.condicaoInicial1=[ ];
valoresDefault.tubo{1}.condicaoInicial2=[ ];
valoresDefault.tubo{1}.condicaoInicial3=[ ];
valoresDefault.tubo{1}.condicaoInicial4=[ ];
valoresDefault.tubo{1}.check=1;

% Valor padrao para a valvula
valoresDefault.valvula{1}.indice=2;
valoresDefault.valvula{1}.vertice=[];
valoresDefault.valvula{1}.CV=1000;
valoresDefault.valvula{1}.especificacaoT1=[]; % abertura manual
valoresDefault.valvula{1}.especificacaoT2=[]; % MODO
valoresDefault.valvula{1}.especificacaoT3=[]; % SP
valoresDefault.valvula{1}.especificacao1=0.5; % abertura manual
valoresDefault.valvula{1}.especificacao2=0;   % MODO
valoresDefault.valvula{1}.especificacao3=1;   % SP
valoresDefault.valvula{1}.tipoPV  =1;  %fração de área  , pressão, vazão L, vazão G, Nivel
valoresDefault.valvula{1}.nohPV   =[];
valoresDefault.valvula{1}.linhaPV =[];
valoresDefault.valvula{1}.acao    =[]; % Direta ou Indireta
valoresDefault.valvula{1}.ContP   =[];
valoresDefault.valvula{1}.ContI   =[];
valoresDefault.valvula{1}.ContD   =[];
valoresDefault.valvula{1}.ContBias=[];

% Valor padrao para a valvula de retenção
valoresDefault.valvulaRet{1}.indice=1;
valoresDefault.valvulaRet{1}.vertice=[];
valoresDefault.valvulaRet{1}.CV=1000;
valoresDefault.valvulaRet{1}.dPMin=0.0;

% Reservatorio
valoresDefault.reservatorio{1}.indice=1;
valoresDefault.reservatorio{1}.vertice=[];
valoresDefault.reservatorio{1}.IPL=10;
valoresDefault.reservatorio{1}.IPG=2;
valoresDefault.reservatorio{1}.P=100; %bar

% Valor padrao para o acoplador
valoresDefault.acoplador{1}.indice=1;
valoresDefault.acoplador{1}.verticeE=[];
valoresDefault.acoplador{1}.verticeS=[];

% Descricao de cada variavel de entrada

detalhesVariaveis.simulador.nome = 'Simulador';
detalhesVariaveis.vertice.nome   = 'Vértice';
detalhesVariaveis.separador.nome = 'Separador';
detalhesVariaveis.reservatorio.nome      = 'Reservatório';
detalhesVariaveis.tubo.nome      = 'Tubo';
detalhesVariaveis.fluido.nome    = 'Fluido';
detalhesVariaveis.valvula.nome   = 'Válvula';
detalhesVariaveis.valvulaRet.nome   = 'Válvula Retenção';
detalhesVariaveis.acoplador.nome = 'Acoplador';

% Variaveis de tubulações
detalhesVariaveis.tubo.indice.unidade='-';
detalhesVariaveis.tubo.indice.nome='Numero da tubulação';
detalhesVariaveis.tubo.indice.descricao=...
    'Numeração atribuida à tubulação.';
detalhesVariaveis.tubo.indice.tipo='fixo';
detalhesVariaveis.tubo.indice.tamanho='escalar';

detalhesVariaveis.tubo.modelo.unidade='-';
detalhesVariaveis.tubo.modelo.nome='Modelo';
detalhesVariaveis.tubo.modelo.descricao=...
    'Modelo de escoamento';
detalhesVariaveis.tubo.modelo.tipo='usuario';
detalhesVariaveis.tubo.modelo.opcoes=...
    {'Bifásico Two-Fluid','Bifásico drift flux','Monofásico Líquido','Monofásico Gás'};
detalhesVariaveis.tubo.modelo.tamanho='vetor';

detalhesVariaveis.tubo.regime.unidade='-';
detalhesVariaveis.tubo.regime.nome='Regime';
detalhesVariaveis.tubo.regime.descricao=...
    'Define se regime será determinado pelo modelo ou fixado';
detalhesVariaveis.tubo.regime.tipo='usuario';
detalhesVariaveis.tubo.regime.opcoes=...
    {'Anular','Bolhas','Intermitente','Est.Suave','Est.Ondas','Automático'};
detalhesVariaveis.tubo.regime.tamanho='escalar';

detalhesVariaveis.tubo.verticeOrigem.unidade='-';
detalhesVariaveis.tubo.verticeOrigem.nome='Vertice de origem da tubulação';
detalhesVariaveis.tubo.verticeOrigem.descricao=...
    'Numeração do vértice de onde se origina a tubulação.';
detalhesVariaveis.tubo.verticeOrigem.tipo='usuario';
detalhesVariaveis.tubo.verticeOrigem.tamanho='escalar';

detalhesVariaveis.tubo.verticeDestino.unidade='-';
detalhesVariaveis.tubo.verticeDestino.nome='Vertice de destino da tubulação';
detalhesVariaveis.tubo.verticeDestino.descricao=...
    'Numeração do vértice para onde se destina a tubulação.';
detalhesVariaveis.tubo.verticeDestino.tipo='usuario';
detalhesVariaveis.tubo.verticeDestino.tamanho='escalar';

detalhesVariaveis.tubo.numeroElementos.unidade='-';
detalhesVariaveis.tubo.numeroElementos.nome='Numero de Elementos Finitos';
detalhesVariaveis.tubo.numeroElementos.descricao=...
    'Numero de elementos utilizados no metodo de discretização.';
detalhesVariaveis.tubo.numeroElementos.tipo='usuario';
detalhesVariaveis.tubo.numeroElementos.tamanho='escalar';

detalhesVariaveis.tubo.rugosidade.unidade='10^-6 m';
detalhesVariaveis.tubo.rugosidade.nome='Rugosidade';
detalhesVariaveis.tubo.rugosidade.descricao=...
    'Valor da rugosidade na superficie interna do duto.';
detalhesVariaveis.tubo.rugosidade.tipo='usuario';
detalhesVariaveis.tubo.rugosidade.tamanho='escalar';

detalhesVariaveis.tubo.diametro.unidade='m';
detalhesVariaveis.tubo.diametro.nome='Diâmetro';
detalhesVariaveis.tubo.diametro.descricao=...
    'Valor do diametro interno da tubulação.';
detalhesVariaveis.tubo.diametro.tipo='usuario';
detalhesVariaveis.tubo.diametro.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoE1.nome='Conector Holdup Entrada';
detalhesVariaveis.tubo.tipocondicaoContornoE1.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoE1.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoE1.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoE1.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoE1.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoS1.nome='Conector Holdup Saida';
detalhesVariaveis.tubo.tipocondicaoContornoS1.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoS1.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoS1.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoS1.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoS1.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoE2.nome='Conector Pressão Entrada';
detalhesVariaveis.tubo.tipocondicaoContornoE2.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoE2.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoE2.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoE2.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoE2.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoS2.nome='Conector Pressão Saida';
detalhesVariaveis.tubo.tipocondicaoContornoS2.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoS2.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoS2.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoS2.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoS2.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoE3.nome='Conector Vazão Líq. Entrada';
detalhesVariaveis.tubo.tipocondicaoContornoE3.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoE3.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoE3.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoE3.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoE3.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoS3.nome='Conector Vazão Líq. Saida';
detalhesVariaveis.tubo.tipocondicaoContornoS3.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoS3.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoS3.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoS3.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoS3.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoE4.nome='Conector Vazão Gás Entrada';
detalhesVariaveis.tubo.tipocondicaoContornoE4.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoE4.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoE4.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoE4.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoE4.tamanho='escalar';

detalhesVariaveis.tubo.tipocondicaoContornoS4.nome='Conector Vazão Gás Saida';
detalhesVariaveis.tubo.tipocondicaoContornoS4.unidade='-';
detalhesVariaveis.tubo.tipocondicaoContornoS4.descricao=...
    'Gradiente nulo (von Neumann) ou valor especificado (Dirichlet)';
detalhesVariaveis.tubo.tipocondicaoContornoS4.tipo='usuario';
detalhesVariaveis.tubo.tipocondicaoContornoS4.opcoes={'Ausência','Dirichlet','von Neumann'};
detalhesVariaveis.tubo.tipocondicaoContornoS4.tamanho='escalar';

detalhesVariaveis.tubo.condicaoInicial1.nome='Vetor CI (hold-up)';
detalhesVariaveis.tubo.condicaoInicial1.unidade='-';
detalhesVariaveis.tubo.condicaoInicial1.descricao=...
    'Condicao inicial';
detalhesVariaveis.tubo.condicaoInicial1.tipo='usuario';
detalhesVariaveis.tubo.condicaoInicial1.tamanho='vetor';

detalhesVariaveis.tubo.condicaoInicial2.nome='Vetor CI (pressao)'; 
detalhesVariaveis.tubo.condicaoInicial2.unidade='10^7 Pa ';
detalhesVariaveis.tubo.condicaoInicial2.descricao=...
    'Condicao inicial';
detalhesVariaveis.tubo.condicaoInicial2.tipo='usuario';
detalhesVariaveis.tubo.condicaoInicial2.tamanho='vetor';

detalhesVariaveis.tubo.condicaoInicial3.nome='Vetor CI (vazao de líquido)';
detalhesVariaveis.tubo.condicaoInicial3.unidade='kg/s';
detalhesVariaveis.tubo.condicaoInicial3.descricao=...
    'Condicao inicial';
detalhesVariaveis.tubo.condicaoInicial3.tipo='usuario';
detalhesVariaveis.tubo.condicaoInicial3.tamanho='vetor';

detalhesVariaveis.tubo.condicaoInicial4.nome='Vetor CI (vazao de gás)';
detalhesVariaveis.tubo.condicaoInicial4.unidade='kg/s';
detalhesVariaveis.tubo.condicaoInicial4.descricao=...
    'Condicao inicial';
detalhesVariaveis.tubo.condicaoInicial4.tipo='usuario';
detalhesVariaveis.tubo.condicaoInicial4.tamanho='vetor';

detalhesVariaveis.tubo.x.unidade='m';
detalhesVariaveis.tubo.x.nome='Coordenada x';
detalhesVariaveis.tubo.x.descricao=...
    'Vetor de coordenada x';
detalhesVariaveis.tubo.x.tipo='usuario';
detalhesVariaveis.tubo.x.tamanho='vetor';

detalhesVariaveis.tubo.y.unidade='m';
detalhesVariaveis.tubo.y.nome='Coordenada y';
detalhesVariaveis.tubo.y.descricao=...
    'Vetor de coordenada y';
detalhesVariaveis.tubo.y.tipo='usuario';
detalhesVariaveis.tubo.y.tamanho='vetor';

detalhesVariaveis.tubo.z.unidade='m';
detalhesVariaveis.tubo.z.nome='Coordenada z';
detalhesVariaveis.tubo.z.descricao=...
    'Vetor de coordenada z';
detalhesVariaveis.tubo.z.tipo='usuario';
detalhesVariaveis.tubo.z.tamanho='vetor';

detalhesVariaveis.tubo.check.nome='Válvula de retenção na saída';
detalhesVariaveis.tubo.check.unidade='-';
detalhesVariaveis.tubo.check.descricao=...
    'Válvula de retenção na saída';
detalhesVariaveis.tubo.check.tipo='usuario';
detalhesVariaveis.tubo.check.opcoes={'Sim','Não'};
detalhesVariaveis.tubo.check.tamanho='escalar';

% Variaveis do fluido

detalhesVariaveis.fluido.densidadeL0.unidade='kg/m³';
detalhesVariaveis.fluido.densidadeL0.nome='Densidade (líquido)';
detalhesVariaveis.fluido.densidadeL0.descricao=...
    'Densidade média da fase líquida na pressão de referência.';
detalhesVariaveis.fluido.densidadeL0.tipo='usuario';
detalhesVariaveis.fluido.densidadeL0.tamanho='escalar';

detalhesVariaveis.fluido.densidadeG0.unidade='kg/m³';
detalhesVariaveis.fluido.densidadeG0.nome='Densidade (gás)';
detalhesVariaveis.fluido.densidadeG0.descricao=...
    'Densidade média da fase gasosa na pressão de referência.';
detalhesVariaveis.fluido.densidadeG0.tipo='usuario';
detalhesVariaveis.fluido.densidadeG0.tamanho='escalar';

detalhesVariaveis.fluido.viscosidadeL.unidade='Pa.s';
detalhesVariaveis.fluido.viscosidadeL.nome='Viscosidade Dinâmica (líquido)';
detalhesVariaveis.fluido.viscosidadeL.descricao=...
    'Viscosidade média da fase líquida.';
detalhesVariaveis.fluido.viscosidadeL.tipo='usuario';
detalhesVariaveis.fluido.viscosidadeL.tamanho='escalar';

detalhesVariaveis.fluido.viscosidadeG.unidade='Pa.s';
detalhesVariaveis.fluido.viscosidadeG.nome='Viscosidade Dinâmica (gás)';
detalhesVariaveis.fluido.viscosidadeG.descricao=...
    'Viscosidade média da fase gasosa.';
detalhesVariaveis.fluido.viscosidadeG.tipo='usuario';
detalhesVariaveis.fluido.viscosidadeG.tamanho='escalar';

detalhesVariaveis.fluido.pressao0.unidade='Pa';
detalhesVariaveis.fluido.pressao0.nome='Pressão de Referencia';
detalhesVariaveis.fluido.pressao0.descricao=...
    'Pressão de referencia para propriedades da fase gasosa.';
detalhesVariaveis.fluido.pressao0.tipo='usuario';
detalhesVariaveis.fluido.pressao0.tamanho='escalar';

detalhesVariaveis.fluido.gamaL.unidade='-';
detalhesVariaveis.fluido.gamaL.nome='Expoente politrópico';
detalhesVariaveis.fluido.gamaL.descricao=...
    'Expoente politrópico da fase líquida.';
detalhesVariaveis.fluido.gamaL.tipo='usuario';
detalhesVariaveis.fluido.gamaL.tamanho='escalar';

detalhesVariaveis.fluido.gamaG.unidade='-';
detalhesVariaveis.fluido.gamaG.nome='Expoente politrópico';
detalhesVariaveis.fluido.gamaG.descricao=...
    'Expoente politrópico da fase gasosa.';
detalhesVariaveis.fluido.gamaG.tipo='usuario';
detalhesVariaveis.fluido.gamaG.tamanho='escalar';

detalhesVariaveis.fluido.tensao.unidade='Pa.m';
detalhesVariaveis.fluido.tensao.nome='Tensao interfacial';
detalhesVariaveis.fluido.tensao.descricao=...
    'Tensao da interface gás-líquido.';
detalhesVariaveis.fluido.tensao.tipo='usuario';
detalhesVariaveis.fluido.tensao.tamanho='escalar';

% Variaveis dos vertices

detalhesVariaveis.vertice.indice.unidade='-';
detalhesVariaveis.vertice.indice.nome='Numeração do vértice';
detalhesVariaveis.vertice.indice.descricao=...
    'Número atribuído ao vértice';
detalhesVariaveis.vertice.indice.tipo='fixo';
detalhesVariaveis.vertice.indice.tamanho='escalar';

detalhesVariaveis.vertice.tipo.unidade='-';
detalhesVariaveis.vertice.tipo.nome='Tipo de vertice';
detalhesVariaveis.vertice.tipo.descricao=...
    'A - DINAMICO, B - ESTATICO,';
detalhesVariaveis.vertice.tipo.opcoes={'TIPO A','TIPO B'};
detalhesVariaveis.vertice.tipo.tipo='usuario';
detalhesVariaveis.vertice.tipo.tamanho='escalar';

detalhesVariaveis.vertice.volume.unidade='m³';
detalhesVariaveis.vertice.volume.nome='Volume {A}';
detalhesVariaveis.vertice.volume.descricao=...
    'Volume do vértice a ser considerado no acoplamento das tubulações';
detalhesVariaveis.vertice.volume.tipo='usuario';
detalhesVariaveis.vertice.volume.tamanho='escalar';

detalhesVariaveis.vertice.especificacaoT1.nome='Tempo CC1 {B}';
detalhesVariaveis.vertice.especificacaoT1.unidade='s';
detalhesVariaveis.vertice.especificacaoT1.descricao='';
detalhesVariaveis.vertice.especificacaoT1.tipo='usuario';
detalhesVariaveis.vertice.especificacaoT1.tamanho='vetor';

detalhesVariaveis.vertice.especificacaoT2.nome='Tempo CC2 {B}';
detalhesVariaveis.vertice.especificacaoT2.unidade='s';
detalhesVariaveis.vertice.especificacaoT2.descricao='';
detalhesVariaveis.vertice.especificacaoT2.tipo='usuario';
detalhesVariaveis.vertice.especificacaoT2.tamanho='vetor';

detalhesVariaveis.vertice.especificacaoT3.nome='Tempo CC3 {A,B}';
detalhesVariaveis.vertice.especificacaoT3.unidade='s';
detalhesVariaveis.vertice.especificacaoT3.descricao='';
detalhesVariaveis.vertice.especificacaoT3.tipo='usuario';
detalhesVariaveis.vertice.especificacaoT3.tamanho='vetor';

detalhesVariaveis.vertice.especificacaoT4.nome='Tempo CC4 {A,B}';
detalhesVariaveis.vertice.especificacaoT4.unidade='s';
detalhesVariaveis.vertice.especificacaoT4.descricao='';
detalhesVariaveis.vertice.especificacaoT4.tipo='usuario';
detalhesVariaveis.vertice.especificacaoT4.tamanho='vetor';

detalhesVariaveis.vertice.especificacao1.nome='CC1 (hold-up){B}';
detalhesVariaveis.vertice.especificacao1.unidade='-';
detalhesVariaveis.vertice.especificacao1.descricao='';
detalhesVariaveis.vertice.especificacao1.tipo='usuario';
detalhesVariaveis.vertice.especificacao1.tamanho='vetor';

detalhesVariaveis.vertice.especificacao2.nome='CC2 (pressão){B}';
detalhesVariaveis.vertice.especificacao2.unidade='-';
detalhesVariaveis.vertice.especificacao2.descricao='';
detalhesVariaveis.vertice.especificacao2.tipo='usuario';
detalhesVariaveis.vertice.especificacao2.tamanho='vetor';

detalhesVariaveis.vertice.especificacao3.nome='CC3 (vazão de líquido){A,B}';
detalhesVariaveis.vertice.especificacao3.unidade='kg/s';
detalhesVariaveis.vertice.especificacao3.descricao='';
detalhesVariaveis.vertice.especificacao3.tipo='usuario';
detalhesVariaveis.vertice.especificacao3.tamanho='vetor';

detalhesVariaveis.vertice.especificacao4.nome='CC4 (vazão de gás){A,B}';
detalhesVariaveis.vertice.especificacao4.unidade='kg/s';
detalhesVariaveis.vertice.especificacao4.descricao='';
detalhesVariaveis.vertice.especificacao4.tipo='usuario';
detalhesVariaveis.vertice.especificacao4.tamanho='vetor';

detalhesVariaveis.vertice.condicaoInicial1.nome='CI (hold-up){A}';
detalhesVariaveis.vertice.condicaoInicial1.unidade='-';
detalhesVariaveis.vertice.condicaoInicial1.descricao='';
detalhesVariaveis.vertice.condicaoInicial1.tipo='usuario';
detalhesVariaveis.vertice.condicaoInicial1.tamanho='escalar';

detalhesVariaveis.vertice.condicaoInicial2.nome='CI (pressão){A}';
detalhesVariaveis.vertice.condicaoInicial2.unidade='10^7 Pa';
detalhesVariaveis.vertice.condicaoInicial2.descricao='';
detalhesVariaveis.vertice.condicaoInicial2.tipo='usuario';
detalhesVariaveis.vertice.condicaoInicial2.tamanho='escalar';

% Variaveis do simulador
detalhesVariaveis.simulador.tempoInicial.nome='Tempo Inicial';
detalhesVariaveis.simulador.tempoInicial.unidade='s';
detalhesVariaveis.simulador.tempoInicial.descricao=...
    'Tempo de partida do simulador.';
detalhesVariaveis.simulador.tempoInicial.tipo='usuario';
detalhesVariaveis.simulador.tempoInicial.tamanho='escalar';

detalhesVariaveis.simulador.tempoFinal.nome='Tempo Final';
detalhesVariaveis.simulador.tempoFinal.unidade='s';
detalhesVariaveis.simulador.tempoFinal.descricao=...
    'Tempo de terminação do simulador.';
detalhesVariaveis.simulador.tempoFinal.tipo='usuario';
detalhesVariaveis.simulador.tempoFinal.tamanho='escalar';

detalhesVariaveis.simulador.incrementoTempo.nome='Incremento de Tempo';
detalhesVariaveis.simulador.incrementoTempo.unidade='s';
detalhesVariaveis.simulador.incrementoTempo.descricao=...
    'Incremento de tempo para armazenamento do resultado.';
detalhesVariaveis.simulador.incrementoTempo.tipo='usuario';
detalhesVariaveis.simulador.incrementoTempo.tamanho='escalar';

detalhesVariaveis.simulador.pressaoR.nome='Pressao de referencia';
detalhesVariaveis.simulador.pressaoR.unidade='Pa';
detalhesVariaveis.simulador.pressaoR.descricao=...
    'Pressao de referencia para escalonamento do estado de pressao';
detalhesVariaveis.simulador.pressaoR.tipo='usuario';
detalhesVariaveis.simulador.pressaoR.tamanho='escalar';

detalhesVariaveis.simulador.tipoOde.nome='Tipo de Ode Solver';
detalhesVariaveis.simulador.tipoOde.unidade='s';
detalhesVariaveis.simulador.tipoOde.descricao=...
    'Tipo de integrador para solução do sistema de EDOs.';
detalhesVariaveis.simulador.tipoOde.tipo='usuario';
detalhesVariaveis.simulador.tipoOde.tamanho='escalar';
detalhesVariaveis.simulador.tipoOde.opcoes={'ode45','ode23','ode113',...
    'ode15s','ode23s','ode23t','ode23tb','odej'};

detalhesVariaveis.simulador.metodo.nome='Tipo de Discretização';
detalhesVariaveis.simulador.metodo.unidade='s';
detalhesVariaveis.simulador.metodo.descricao=...
    'Tipo de metodo de discretização de EDP';
detalhesVariaveis.simulador.metodo.tipo='usuario';
detalhesVariaveis.simulador.metodo.tamanho='vetor';
detalhesVariaveis.simulador.metodo.opcoes={'Metodo de Diferenças Finitas',...
    'Metodo de Galerkin','Metodo de Petrov-Galerkin'};

detalhesVariaveis.simulador.maxStep.nome='Passo máximo do Ode solver';        
detalhesVariaveis.simulador.maxStep.unidade='s';
detalhesVariaveis.simulador.maxStep.descricao=...
    'Variação máxima do integrador.';
detalhesVariaveis.simulador.maxStep.tipo='usuario';
detalhesVariaveis.simulador.maxStep.tamanho='escalar';

detalhesVariaveis.simulador.absTol.nome='Tolerancia absoluta do Ode solver';        
detalhesVariaveis.simulador.absTol.unidade='s';
detalhesVariaveis.simulador.absTol.descricao=...
    'Parametro interno para determinar convergencia do metodo.';
detalhesVariaveis.simulador.absTol.tipo='usuario';
detalhesVariaveis.simulador.absTol.tamanho='escalar';

detalhesVariaveis.simulador.relTol.nome='Tolerancia relativa do Ode solver';        
detalhesVariaveis.simulador.relTol.unidade='s';
detalhesVariaveis.simulador.relTol.descricao=...
    'Parametro interno para determinar convergencia do metodo.';
detalhesVariaveis.simulador.relTol.tipo='usuario';
detalhesVariaveis.simulador.relTol.tamanho='escalar';

detalhesVariaveis.simulador.ordemQuadratura.nome='Ordem da quadratura';
detalhesVariaveis.simulador.ordemQuadratura.unidade='s';
detalhesVariaveis.simulador.ordemQuadratura.descricao='';
detalhesVariaveis.simulador.ordemQuadratura.tipo='usuario';
detalhesVariaveis.simulador.ordemQuadratura.tamanho='escalar';
detalhesVariaveis.simulador.ordemQuadratura.opcoes={'3','5','8','12'};

detalhesVariaveis.simulador.tolerancia.nome='Tolerância do SS';
detalhesVariaveis.simulador.tolerancia.unidade='s';
detalhesVariaveis.simulador.tolerancia.descricao='';
detalhesVariaveis.simulador.tolerancia.tipo='usuario';
detalhesVariaveis.simulador.tolerancia.tamanho='escalar';

detalhesVariaveis.simulador.maxIter.nome='N. maximo de iterações do SS';
detalhesVariaveis.simulador.maxIter.unidade='-';
detalhesVariaveis.simulador.maxIter.descricao='';
detalhesVariaveis.simulador.maxIter.tipo='usuario';
detalhesVariaveis.simulador.maxIter.tamanho='escalar';

detalhesVariaveis.simulador.delta.nome='RelTol da Jacobiana';
detalhesVariaveis.simulador.delta.unidade='-';
detalhesVariaveis.simulador.delta.descricao='';
detalhesVariaveis.simulador.delta.tipo='usuario';
detalhesVariaveis.simulador.delta.tamanho='escalar';

detalhesVariaveis.simulador.delta2.nome='AbsTol da Jacobiana';
detalhesVariaveis.simulador.delta2.unidade='-';
detalhesVariaveis.simulador.delta2.descricao='';
detalhesVariaveis.simulador.delta2.tipo='usuario';
detalhesVariaveis.simulador.delta2.tamanho='escalar';

% Variaveis das válvulas
detalhesVariaveis.valvula.indice.unidade='-';
detalhesVariaveis.valvula.indice.nome='Numeração da valvula';
detalhesVariaveis.valvula.indice.descricao=...
    'Número atribuído a valvula instalada na saída de linha';
detalhesVariaveis.valvula.indice.tipo='fixo';
detalhesVariaveis.valvula.indice.tamanho='escalar';

detalhesVariaveis.valvula.vertice.unidade='-';
detalhesVariaveis.valvula.vertice.nome='Vertice';
detalhesVariaveis.valvula.vertice.descricao=...
    'Indice do vertice';
detalhesVariaveis.valvula.vertice.tipo='usuario';
detalhesVariaveis.valvula.vertice.tamanho='escalar';

detalhesVariaveis.valvula.CV.unidade='-';
detalhesVariaveis.valvula.CV.nome='Coeficiente da válvula (CV)';
detalhesVariaveis.valvula.CV.descricao=...
    'Valor do CV da válvula';
detalhesVariaveis.valvula.CV.tipo='usuario';
detalhesVariaveis.valvula.CV.tamanho='escalar';

detalhesVariaveis.valvula.tipoPV.nome='Tipo de variavel controlada';
detalhesVariaveis.valvula.tipoPV.unidade='';
detalhesVariaveis.valvula.tipoPV.descricao='';
detalhesVariaveis.valvula.tipoPV.tipo='usuario';
detalhesVariaveis.valvula.tipoPV.tamanho='vetor';
detalhesVariaveis.valvula.tipoPV.opcoes={'Fração de Área','Pressão','Vazão de Líquido','Vazão de Gás','Nível'};

detalhesVariaveis.valvula.nohPV.nome='Noh correspondente a variavel controlada';
detalhesVariaveis.valvula.nohPV.unidade='';
detalhesVariaveis.valvula.nohPV.descricao='';
detalhesVariaveis.valvula.nohPV.tipo='usuario';
detalhesVariaveis.valvula.nohPV.tamanho='escalar';

detalhesVariaveis.valvula.linhaPV.unidade='-';
detalhesVariaveis.valvula.linhaPV.nome='Índice do elemento que contem variavel controlada';
detalhesVariaveis.valvula.linhaPV.descricao='';
detalhesVariaveis.valvula.linhaPV.tipo='usuario';
detalhesVariaveis.valvula.linhaPV.tamanho='escalar';

detalhesVariaveis.valvula.acao.unidade='-';
detalhesVariaveis.valvula.acao.nome='Acao do controlador';
detalhesVariaveis.valvula.acao.descricao=...
    '1 se Direta e -1 se indireta';
detalhesVariaveis.valvula.acao.tipo='usuario';
detalhesVariaveis.valvula.acao.tamanho='escalar';

detalhesVariaveis.valvula.ContP.unidade='-';
detalhesVariaveis.valvula.ContP.nome='Ganho Proporcional';
detalhesVariaveis.valvula.ContP.descricao=...
    'Ganho proporcional da malha de controle';
detalhesVariaveis.valvula.ContP.tipo='usuario';
detalhesVariaveis.valvula.ContP.tamanho='escalar';

detalhesVariaveis.valvula.ContI.unidade='-';
detalhesVariaveis.valvula.ContI.nome='Ação Integral';
detalhesVariaveis.valvula.ContI.descricao=...
    'Ação Integral na forma de reset-rate (1/s)';
detalhesVariaveis.valvula.ContI.tipo='usuario';
detalhesVariaveis.valvula.ContI.tamanho='escalar';

detalhesVariaveis.valvula.ContD.unidade='-';
detalhesVariaveis.valvula.ContD.nome='Ação Derivativa';
detalhesVariaveis.valvula.ContD.descricao=...
    'Ação Derivativa na forma de constante de tempo (s)';
detalhesVariaveis.valvula.ContD.tipo='usuario';
detalhesVariaveis.valvula.ContD.tamanho='escalar';

detalhesVariaveis.valvula.ContBias.unidade='-';
detalhesVariaveis.valvula.ContBias.nome='Bias do Controlador';
detalhesVariaveis.valvula.ContBias.descricao=...
    'Bias em termos de abertura (0 - 1)';
detalhesVariaveis.valvula.ContBias.tipo='usuario';
detalhesVariaveis.valvula.ContBias.tamanho='escalar';

detalhesVariaveis.valvula.especificacaoT1.unidade='-';
detalhesVariaveis.valvula.especificacaoT1.nome='Tempo Especificacao 1';
detalhesVariaveis.valvula.especificacaoT1.descricao=...
    'Abertura no modo manual';
detalhesVariaveis.valvula.especificacaoT1.tipo='usuario';
detalhesVariaveis.valvula.especificacaoT1.tamanho='vetor';

detalhesVariaveis.valvula.especificacaoT2.unidade='-';
detalhesVariaveis.valvula.especificacaoT2.nome='Tempo Especificacao 2';
detalhesVariaveis.valvula.especificacaoT2.descricao=...
    'Modo da valvula de controle';
detalhesVariaveis.valvula.especificacaoT2.tipo='usuario';
detalhesVariaveis.valvula.especificacaoT2.tamanho='vetor';

detalhesVariaveis.valvula.especificacaoT3.unidade='-';
detalhesVariaveis.valvula.especificacaoT3.nome='Tempo Especificacao 3';
detalhesVariaveis.valvula.especificacaoT3.descricao=...
    'Set-point de controle';
detalhesVariaveis.valvula.especificacaoT3.tipo='usuario';
detalhesVariaveis.valvula.especificacaoT3.tamanho='vetor';

detalhesVariaveis.valvula.especificacao1.unidade='-';
detalhesVariaveis.valvula.especificacao1.nome='Especificacao Abertura ';
detalhesVariaveis.valvula.especificacao1.descricao=...
    'Abertura no modo manual';
detalhesVariaveis.valvula.especificacao1.tipo='usuario';
detalhesVariaveis.valvula.especificacao1.tamanho='vetor';

detalhesVariaveis.valvula.especificacao2.unidade='-';
detalhesVariaveis.valvula.especificacao2.nome='Especificacao Modo';
detalhesVariaveis.valvula.especificacao2.descricao=...
    'Modo da malha de controle';
detalhesVariaveis.valvula.especificacao2.tipo='usuario';
detalhesVariaveis.valvula.especificacao2.tamanho='vetor';

detalhesVariaveis.valvula.especificacao3.unidade='-';
detalhesVariaveis.valvula.especificacao3.nome='Especificacao SP';
detalhesVariaveis.valvula.especificacao3.descricao=...
    'Set-point de controle';
detalhesVariaveis.valvula.especificacao3.tipo='usuario';
detalhesVariaveis.valvula.especificacao3.tamanho='vetor';


% Variaveis das válvulas
detalhesVariaveis.valvulaRet.indice.unidade='-';
detalhesVariaveis.valvulaRet.indice.nome='Numeração da valvula de Ret.';
detalhesVariaveis.valvulaRet.indice.descricao=...
    'Número atribuído a valvula de retenção instalada na saída de linha';
detalhesVariaveis.valvulaRet.indice.tipo='fixo';
detalhesVariaveis.valvulaRet.indice.tamanho='escalar';

detalhesVariaveis.valvulaRet.vertice.unidade='-';
detalhesVariaveis.valvulaRet.vertice.nome='Vertice';
detalhesVariaveis.valvulaRet.vertice.descricao=...
    'Indice do vertice';
detalhesVariaveis.valvulaRet.vertice.tipo='usuario';
detalhesVariaveis.valvulaRet.vertice.tamanho='escalar';

detalhesVariaveis.valvulaRet.CV.unidade='-';
detalhesVariaveis.valvulaRet.CV.nome='Coeficiente da válvula (CV)';
detalhesVariaveis.valvulaRet.CV.descricao=...
    'Valor do CV da válvula';
detalhesVariaveis.valvulaRet.CV.tipo='usuario';
detalhesVariaveis.valvulaRet.CV.tamanho='escalar';

detalhesVariaveis.valvulaRet.dPMin.unidade='-';
detalhesVariaveis.valvulaRet.dPMin.nome='Mínimo Diferencial de Pressão';
detalhesVariaveis.valvulaRet.dPMin.descricao=...
    'Valor do diferencial mínimo de pressão (bar)';
detalhesVariaveis.valvulaRet.dPMin.tipo='usuario';
detalhesVariaveis.valvulaRet.dPMin.tamanho='escalar';

% Variaveis dos acopladores
detalhesVariaveis.acoplador.indice.unidade='-';
detalhesVariaveis.acoplador.indice.nome='Numeração do acoplador';
detalhesVariaveis.acoplador.indice.descricao=...
    'Numero atribuido ao acoplador';
detalhesVariaveis.acoplador.indice.tipo='fixo';
detalhesVariaveis.acoplador.indice.tamanho='escalar';

detalhesVariaveis.acoplador.verticeE.unidade='-';
detalhesVariaveis.acoplador.verticeE.nome='Numeração do vertice 1';
detalhesVariaveis.acoplador.verticeE.descricao=...
    'Numero atribuido ao vertice a montante';
detalhesVariaveis.acoplador.verticeE.tipo='usuario';
detalhesVariaveis.acoplador.verticeE.tamanho='escalar';

detalhesVariaveis.acoplador.verticeS.unidade='-';
detalhesVariaveis.acoplador.verticeS.nome='Numeração do vertice 2';
detalhesVariaveis.acoplador.verticeS.descricao=...
    'Numero atribuido ao vertice a jusante';
detalhesVariaveis.acoplador.verticeS.tipo='usuario';
detalhesVariaveis.acoplador.verticeS.tamanho='escalar';


% Variaveis dos reservatorios
detalhesVariaveis.reservatorio.indice.unidade='-';
detalhesVariaveis.reservatorio.indice.nome='Numeração do reservatório';
detalhesVariaveis.reservatorio.indice.descricao=...
    'Numero atribuido ao reservatorio';
detalhesVariaveis.reservatorio.indice.tipo='fixo';
detalhesVariaveis.reservatorio.indice.tamanho='escalar';

detalhesVariaveis.reservatorio.vertice.unidade='-';
detalhesVariaveis.reservatorio.vertice.nome='Vertice';
detalhesVariaveis.reservatorio.vertice.descricao=...
    'Indice do vertice';
detalhesVariaveis.reservatorio.vertice.tipo='usuario';
detalhesVariaveis.reservatorio.vertice.tamanho='escalar';

detalhesVariaveis.reservatorio.IPL.unidade='-';
detalhesVariaveis.reservatorio.IPL.nome='IP Líquido';
detalhesVariaveis.reservatorio.IPL.descricao=...
    'Índice de Produção de Líquido (kg/s/bar)';
detalhesVariaveis.reservatorio.IPL.tipo='usuario';
detalhesVariaveis.reservatorio.IPL.tamanho='escalar';

detalhesVariaveis.reservatorio.IPG.unidade='-';
detalhesVariaveis.reservatorio.IPG.nome='IP Gás';
detalhesVariaveis.reservatorio.IPG.descricao=...
    'Índice de Produção de Gás (kg/s/bar)';
detalhesVariaveis.reservatorio.IPG.tipo='usuario';
detalhesVariaveis.reservatorio.IPG.tamanho='escalar';

detalhesVariaveis.reservatorio.P.unidade='-';
detalhesVariaveis.reservatorio.P.nome='Pressão Reservatório';
detalhesVariaveis.reservatorio.P.descricao=...
    'Pressão do reservatório (bar)';
detalhesVariaveis.reservatorio.P.tipo='usuario';
detalhesVariaveis.reservatorio.P.tamanho='escalar';