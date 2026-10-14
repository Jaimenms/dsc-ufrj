function [F,saidas,FSS,A,B,ixModel]=modelo_duto_L(t,x,y,parametros,j)
% Função que gera o vetor de coeficientes da matriz da EDP discretizada
% para o modelo monofásico de líquido.

% Preparando saídas
saidas=[];
FSS=[];
F=[];
ixModel=[];
if nargout>3
    % Matrizes de coeficientes
    A = [0 1;...
         2 3];
    B = [0;4];
    ixModel = [0;1;1;0];
else

    
%% Executando cálculos iniciais

    % ny -> 
    % ix -> vetor de ponteiros (1xNk)
    ne = 4; %numero de variaveis de estado (1x1)
    [nx,ny] = size(y); % ny (colunas do vetor de estados)
    ni = nx/ne; % numero de nohs
    nt = length(t); % numero de instantes de tempo
    ix = 1:ne:nx;

    t = ones(ny/nt,1)*t;
    t = ones(ni,1)*t(:)';
    
    g = 9.80665;

    parametrosDuto   = parametros{j};
    parametrosFluido = parametros{j}.Fluido;

    
%% Extraindo variaveis de entrada
    angulo_teta = interpola_angulo(parametrosDuto.x,parametrosDuto.angulo_teta,x);
    rugo        = parametrosDuto.rugo.*ones(ni,ny);
    D           = parametrosDuto.D.*ones(ni,ny);
    PR          = parametrosDuto.PR;

%% Capturando variáveis de estado

    % Desempacotando variaveis de estado:
    AUX  = y(ix,:);
    PT   = y(ix+1,:);
    q    = y(ix+2,:);
    AUX  = y(ix+3,:);

    
%% Modelo
    % P     -> pressão absoluta em Pa
    % v   -> velocidade media
    P = PR.*PT;

    % Calculando propriedades dos fluidos:
    [prop,aux,aux] = modelo_fluido(P,0,parametrosFluido);

    Area  = pi.*D.^2./4;
    v  = q./(prop.dens.*Area);
    Per = pi.*D;

    [f,Re]   =fator  (v,prop.dens,prop.visc,D,rugo);
    tensao  = f.*prop.dens.*(v).*abs(v)./2;
    Gama = Per.*tensao;
    
    FSS=(Gama+9.80665.*Area.*prop.dens.*sin(angulo_teta))./Area;
        
    % Deteminando o vetor de saida do modelo
       
    F =  [1./Area./prop.C./PR;... % F1
          Area.*(1-prop.C.*v.*abs(v)).*PR;... % F2
          2.*abs(v);... % F3
          Gama+9.80665.*Area.*prop.dens.*sin(angulo_teta)];... % F4
       

%% Calculando saídas
    if nargout>1

        saidas.inclinacao   = angulo_teta(:,1)*ones(1,size(y,2));
        saidas.diametro     = D.*ones(nx./ne,size(y,2));
        saidas.alfaL        = ones(size(v));
        saidas.alfaG        = zeros(size(v));
        saidas.pressao      = PT.*PR./1e5;
        saidas.vazaoMassaL  = q;
        saidas.vazaoVolL    = q./prop.dens;
        saidas.velocidadeL  = v;
        saidas.velSomL      = (1./prop.C).^0.5;
        saidas.cargaL       = FSS.*100/1e5;
        saidas.densidadeL   = prop.dens;
        saidas.viscosidadeL = prop.visc.*ones(nx./ne,size(y,2));
        
    end

end

function yi = interpola_angulo(x,y,xi)
[nxi1,nxi2] = size(xi);
yi = zeros(nxi1,nxi2);
nx = length(x);

for i=1:nxi1
    for j=1:nxi2
        ix2 = find(x==xi(i,j));
        if ~isempty(ix2)
            if ix2==1,      yi(i,j) = y(1);
            elseif ix2==nx, yi(i,j) = y(end);
            else            yi(i,j) = 0.5.*(y(ix2)+y(ix2-1));
            end
        else
            ix2 = find(x<=xi(i,j));
            yi(i,j) = y(ix2(end));
        end
    end
end
