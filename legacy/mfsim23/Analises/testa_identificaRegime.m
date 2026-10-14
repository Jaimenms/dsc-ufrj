% Rotina que gera curvas críticas dos mapas de escoamento.

caso = 1;

switch caso
    case 1
        angulo_teta = 0;
        D = 0.58;
        rugo = 0;
        hL = 0.45*D;
        q_L = 96.7;
        q_L = 96.7.*(0.001:0.25:1);
        q_G = 38.1;
        q_G = 38.1.*(0.001:0.25:1);
        viscL = 0.35e-3;
        viscG = 0.015e-3;
        densL = 610;
        densG = 120;
    case 2
        angulo_teta = 50./180.*pi;
        D = 0.1;
        rugo = 0;
        hL = 0.55*D;
        densL = 560;
        densG = 100;
        q_L = 0.3.*pi.*D.^2/4.*densL;
        q_G = 10.*pi.*D.^2/4.*densG;
        viscL = 0.25e-3;
        viscG = 0.015e-3;
end
        
        
beta=2*acos(1-2*hL);
for i=1:size(beta,1)
    for j=1:size(beta,2)
        alfa(i,j)=fzero(@(x) beta(i,j)...
            -sin(beta(i,j))-2*pi*(1-x),1-hL(i,j));
    end
end

Area = (pi/4).*D.^2;
AreaG = Area*alfa;
AreaL = Area-AreaG;

[fl, reg1, reg2, X, Y, FL, FG, FGcosteta, NL, K, T] = ...
    identificaRegime(angulo_teta,D,rugo,hL,alfa,q_L,q_G,viscL,viscG,densL,densG,AreaL,AreaG);


% Definindo coeficientes dos polinômios
p.FL =  [-0.6563  1.4823  0.1011 -0.0047];
p.K  =  [ 0.0320 -0.0717  0.0398  2.6442e-004];
p.NL =  [-1.0578  0.7626  0.3833 -0.0026];
p.T  =  [-3.1580  3.1666 -0.0385  0.2025];
p.FGcosteta  = [-55.8352 222.0618 -364.8699 319.0999 -159.6804 46.5713 -8.4552 1.0795];

x_est = 0:0.01:1;
y_est.FL = polyval(p.FL,x_est);
y_est.K = polyval(p.K,x_est);
y_est.NL = polyval(p.NL,x_est);
y_est.FGcosteta = polyval(p.FGcosteta,x_est);
y_est.T = polyval(p.T,x_est);

semilogy(x_est,y_est.FL,'b-')
hold on
semilogy(ones(size(FL)).*hL/D,FL,'ob')

semilogy(x_est,y_est.K,'r-')
semilogy(ones(size(K)).*hL/D,K,'or')

semilogy(x_est,y_est.NL,'k-')
semilogy(ones(size(NL)).*hL/D,NL,'ok')

semilogy(x_est,y_est.FGcosteta,'m-')
semilogy(ones(size(FGcosteta)).*hL/D,FGcosteta,'om')

semilogy(x_est,y_est.T,'g-')
semilogy(ones(size(T)).*hL/D,T,'og')

text(0.3,8e-3,'K:Estratificado wavy','color',[1 0 0])
text(0.3,5.5e-3,'K:Estratificado smooth','color',[1 0 0])

text(0.3,0.5,'T:Bolhas','color',[0 1 0])
text(0.3,0.35,'T:Nao-bolhas','color',[0 1 0])

text(0.7,0.3,'NL:Anular')
text(0.7,0.2,'NL:Estratificado')

text(0.7,0.03,'FGt:Nao Estratificado','color',[1 0 1])
text(0.7,0.02,'FGt:Estratificado','color',[1 0 1])

text(0.8,0.9,'FL:Estratificado wavy','color',[0 0 1])
text(0.8,0.65,'FL:Estratificado smooth','color',[0 0 1])


xlim([0 1])
ylim([0.001 1.1])