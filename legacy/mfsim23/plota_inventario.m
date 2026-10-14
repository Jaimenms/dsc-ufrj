function plota_inventario(ind_tubo)
global saidas eixos

valores = saidas.tubo{ind_tubo};
eixoL = eixos.tubo.X.valor{ind_tubo};
eixoT = eixos.tubo.Y.valor{ind_tubo};

% Determinando o inventário de líquido no tubo
Area = valores.diametro.^2.*pi./4;

inventarioL = valores.alfaL.*Area.*valores.densidadeL;
inventarioL = (eixoL(2:end)-eixoL(1:(end-1)))'*((inventarioL(2:end,:)+inventarioL(1:(end-1),:))./2);
dqL = valores.vazaoMassaL(1,:)-valores.vazaoMassaL(end,:);
aux_int = [0 (eixoT(2:end)-eixoT(1:(end-1)))];
int_dqL = cumsum(aux_int(:).*dqL(:));

inventarioG = valores.alfaG.*Area.*valores.densidadeG;
inventarioG = (eixoL(2:end)-eixoL(1:(end-1)))'*((inventarioG(2:end,:)+inventarioG(1:(end-1),:))./2);
dqG = valores.vazaoMassaG(1,:)-valores.vazaoMassaG(end,:);
int_dqG = cumsum(aux_int(:).*dqG(:));

figure,
subplot(2,1,1)
a=plot(eixoT,inventarioL,eixoT,inventarioL(1)+int_dqL);
set(a,'linewidth',2)
xlabel(eixos.tubo.Y.nome,'fontsize',12)
ylabel('Massa de líquido (kg)','fontsize',12)
legend('Líquido','Inicial + Integral (Qin-Qout)')
subplot(2,1,2)
plot(eixoT,dqL,'linewidth',2)
xlabel(eixos.tubo.Y.nome,'fontsize',12)
ylabel('Vazão de líquido (kg/s)','fontsize',12)

figure,
subplot(2,1,1)
a=plot(eixoT,inventarioG,eixoT,inventarioG(1)+int_dqG);
xlabel(eixos.tubo.Y.nome,'fontsize',12)
ylabel('Massa de gás (kg)','fontsize',12)
set(a,'linewidth',2)
legend('Gás','Inicial + Integral (Qin-Qout)')
subplot(2,1,2)
plot(eixoT,dqG,'linewidth',2)
xlabel(eixos.tubo.Y.nome,'fontsize',12)
ylabel('Vazão de gás (kg/s)','fontsize',12)