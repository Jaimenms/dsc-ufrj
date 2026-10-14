function [nomes] = arvore_saidas(saidas,nomes)
% Função que compõe os nomes de todas as saídas do modelo

% Definição dos nomes
nomes.vertice  = 'Vértice';
nomes.tubo     = 'Tubulação';
nomes.valvula  = 'Valvula';
nomes.valvulaRet  = 'Valvula de Retenção';
nomes.reservatorio  = 'Reservatorio';

nomes.twoDim         = 'Curvas';
nomes.threeDim       = 'Superfícies';
nomes.compare        = 'Propriedades';
nomes.inclinacao     = 'Inclinação (o)';
nomes.diametro       = 'Diâmetro (m)';
nomes.posX           = 'Posição X (m)';
nomes.posY           = 'Posição Y (m)';
nomes.posZ           = 'Posição Z (m)';
nomes.vazaoMassaG    = 'Vazão de Gás (kg/s)';
nomes.vazaoMassaL    = 'Vazão de Líquido (kg/s)';
nomes.vazaoMassa     = 'Vazão Total (kg/s)';
nomes.vazaoVolG      = 'Vazão de Gás (m³/s)';
nomes.vazaoVolL      = 'Vazão de Líquido (m³/s)';
nomes.vazaoVol       = 'Vazão Total (m³/s)';
nomes.velocidadeSG   = 'Vel. Superficial de Gas (m/s)';
nomes.velocidadeSL   = 'Vel. Superficial de Líquido (m/s)';
nomes.velocidadeG    = 'Velocidade de Gas (m/s)';
nomes.velocidadeL    = 'Velocidade de Líquido (m/s)';
nomes.velocidadeI    = 'Velocidade relativa L-G (m/s)';
nomes.velSomG        = 'Velocidade do Som no Gas (m/s)';
nomes.velSomL        = 'Velocidade do Som no Líquido (m/s)';
nomes.velSomM        = 'Velocidade no Som na Mistura (m/s)';
nomes.tensaoL        = 'Tensão Cisalhamento Liquido (Pa)';
nomes.tensaoG        = 'Tensão Cisalhamento Gás (Pa)';
nomes.tensaoIL       = 'Tensão Cisalhamento Interface (L) (Pa)';
nomes.tensaoIG       = 'Tensão Cisalhamento Interface (G) (Pa)';
nomes.cargaL         = 'Perda de carga no Liquido (10^5 Pa/100m)';
nomes.cargaG         = 'Perda de Carga no Gás (10^5 Pa/100m)';
nomes.densidadeL     = 'Densidade Liquido (kg/m³)';
nomes.densidadeG     = 'Densidade Gás (kg/m³)';
nomes.densidadeM     = 'Densidade da Mistura (kg/m³)';
nomes.viscosidadeL   = 'Viscosidade Liquido (Pa.s)';
nomes.viscosidadeG   = 'Viscosidade Gás (Pa.s)';

nomes.pressao        = 'Pressao (10^5 Pa)';
nomes.dP             = 'Diferencial de Pressao (Bar)';
nomes.dpresL         = 'Dif. Pressao Líquido-Interface(Pa)';
nomes.dpresG         = 'Dif. Pressao Gas-Interface(Pa)';

nomes.alfaG          = 'Fração de Área de Gas (m²/m²)';
nomes.alfaL          = 'Fração de Área de Liquido (m²/m²)';
nomes.holdupG        = 'Hold-up de Gas (m²/m²)';
nomes.holdupL        = 'Hold-up de Liquido (m²/m²)';
nomes.iter           = 'Numero de Iterações';
nomes.fob            = 'Função Objetivo';
nomes.Re_L           = 'No.de Reynods do Líquido';
nomes.Re_G           = 'No.de Reynods do Gás';
nomes.Re_I           = 'No.de Reynods da Interface';
nomes.RazVol         = 'Razão de vazões volumétricas';
nomes.reg1           = 'Regime de Escoamento - tipo 1';
nomes.reg1C          = 'Regime Calculado - tipo 1';
nomes.reg2           = 'Regime de Escoamento - tipo 2';
nomes.fCFL           = 'FL Critico';
nomes.fCNL           = 'NL Critico';
nomes.fCK            = 'K Critico';
nomes.fCT            = 'T Critico';
nomes.fCFGcos        = 'Kelvin-Helmholtz Critico';
nomes.fFL            = 'Fator FL';
nomes.fNL            = 'Fator NL';
nomes.fK             = 'FatorK';
nomes.fT             = 'Fator T';
nomes.fFGcos         = 'Kelvin-Helmholtz';
nomes.hD             = 'h/D';
nomes.NivelPerc      = 'Nível Percentual (%)';
nomes.NivelAbs       = 'Nível (m)';
nomes.VolumeL        = 'Volume de Líquido (m³)';
nomes.VolumeG        = 'Volume de Gás (m³)';

nomes.SP        = 'Set-point (-)';
nomes.PV        = 'Variavel de Process (-)';
nomes.Erro      = 'Erro do controlador';
nomes.Modo      = 'Modo do Controlador';
nomes.Ab        = 'Abertura da Válvula';


nomes.db        = 'Diametro das bolhas (mm)';
nomes.Rs        = 'Hold-up de líquido no slug (-)';
nomes.Rf        = 'Hold-up de líquido no filme (-)';
nomes.ls        = 'Comprimento do slug (m)';
nomes.lf        = 'Comprimento do filme (m)';
nomes.vt        = 'Velocidade de deslocamento do slug (m;s)';


nomes.alfaG_est = 'Fração de gás para reg. estratificado';
nomes.alfaG_anu = 'Fração de gás para reg. anular';
nomes.alfaG_bol = 'Fração de gás para reg. bolhas';

nomes.Eig_1Real = 'Autovalor 1 - parte real';
nomes.Eig_2Real = 'Autovalor 2 - parte real';
nomes.Eig_3Real = 'Autovalor 3 - parte real';
nomes.Eig_4Real = 'Autovalor 4 - parte real';

nomes.Eig_1Imag = 'Autovalor 1 - parte imaginária';
nomes.Eig_2Imag = 'Autovalor 2 - parte imaginária';
nomes.Eig_3Imag = 'Autovalor 3 - parte imaginária';
nomes.Eig_4Imag = 'Autovalor 4 - parte imaginária';

nomes.ZL = 'Vazão mássica de líquido (kg/s)';
nomes.ZG = 'Vazão mássica de gás (kg/s)';
nomes.qL = 'Vazão mássica de líquido (kg/s)';
nomes.qG = 'Vazão mássica de gás (kg/s)';
nomes.ZP = 'Pressão a montante (bar)';


nomes.flTra = 'Flag de indicação de padrão transição';
nomes.flEst = 'Flag de indicação de padrão estratificado';
nomes.flInt = 'Flag de indicação de padrão intermitente';
nomes.flDis = 'Flag de indicação de padrão disperso';
