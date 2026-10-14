function [hD_est,hD_anu,alfaG_est,alfaG_anu,alfaG_bol]=determinaAlfaEstavel(entradas)
% Função que determina frações de área para modelos de Taitel, Pots e Zuber

% Desempacotando parametros
v_LS  = entradas.v_LS;
v_GS  = entradas.v_GS;
D     = entradas.D;
angulo_teta = entradas.angulo_teta;
densL = entradas.densL;
densG = entradas.densG;
viscL = entradas.viscL;
viscG = entradas.viscG;
tensao = entradas.tensao;

[nx,ny] = size(v_LS);
hD_est = zeros(nx,ny);
hD_anu = hD_est;

aux = (10.^(-3:0.01:0))./2;
aux2 = [aux (1-fliplr(aux))];
hd_teste = aux2;
hd_teste2 = 0.01:0.01:0.99;


for i=1:nx
    for j=1:ny
        [vl_teste] = modelo_Taitel(hd_teste,v_GS(i,j),densL(i,j),densG(i,j),viscL(i,j),viscG(i,j),angulo_teta(i,j),D(i,j),2,2);
        [vl_teste2] = modelo_Pots(hd_teste2,v_GS(i,j),densL(i,j),densG(i,j),viscL(i,j),viscG(i,j),angulo_teta(i,j),D(i,j),2,2);
        
        hd_teste_mod = hd_teste;
        ix=find(isnan(vl_teste));
        if ~isempty(ix)
            hd_teste_mod=hd_teste_mod(ix(end):end);
            vl_teste=vl_teste(ix(end):end);
        end
        hd_teste_mod(isnan(vl_teste))=[];
        vl_teste(isnan(vl_teste))=[];
        
        [aux,ix]=min(abs(real(vl_teste)-v_LS(i,j)));
        if isempty(ix)
            hD_est(i,j) = 1e-6;
        else
            hD_est(i,j) = hd_teste_mod(ix(1));
        end

        [aux,ix2]=min(abs(vl_teste2-v_LS(i,j)));
        if isempty(ix2)
            hD_anu(i,j) = 1e-6;
        else
            hD_anu(i,j) = hd_teste2(ix2(1));
        end
    end
end

% Determinando alfa para estratificado
beta_est = 2.*acos(1-2.*hD_est);
alfaG_est = 1-((beta_est-sin(beta_est))./(2*pi));

% Determinando alfa para anular
alfaG_anu = (1-2.*hD_anu).^2;

% Determinando alfa para bolhas
ub = 1.53.*(tensao.*9.81.*abs(densL-densG)./(densL.^2)).^0.25.*sin(angulo_teta);
C0 = 1.25;
v_M = v_LS+v_GS;
v_G = C0.*v_M+ub;
alfaG_bol = v_GS./v_G;
alfaG_bol = max(alfaG_bol ,1e-6);
alfaG_bol = min(alfaG_bol ,1-1e-6);