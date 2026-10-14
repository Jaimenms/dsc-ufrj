% Rotina que analisa razões.

alfaG = [0.001:0.001:0.999];
alfaL = 1-alfaG;

gamaL1 = 1000;
gamaG1 = 2.0;
dG1 = alfaG./gamaG1;
dL1 = alfaL./gamaL1;
rG1 = dG1./(dL1+dG1);
rL1 = dL1./(dL1+dG1);

gamaL2 = 1000;
gamaG2 = 1.2;
dG2 = alfaG./gamaG2;
dL2 = alfaL./gamaL2;
rG2 = dG2./(dL2+dG2);
rL2 = dL2./(dL2+dG2);

gamaL3 = 100;
gamaG3 = 1.2;
dG3 = alfaG./gamaG3;
dL3 = alfaL./gamaL3;
rG3 = dG3./(dL3+dG3);
rL3 = dL3./(dL3+dG3);

figure
subplot(2,1,1)
semilogy(alfaG,rL1,'b-','linewidth',2)
hold on
semilogy(alfaG,rL2,'k-','linewidth',2)
semilogy(alfaG,rL3,'r-','linewidth',2)
ht(1)=title('Razões de Gás e Líquido');
he(1)=ylabel('r_L');
he(2)=xlabel('Alfa_G');

subplot(2,1,2)
semilogy(alfaG,rG1,'b-','linewidth',2)
hold on
semilogy(alfaG,rG2,'k-','linewidth',2)
semilogy(alfaG,rG3,'r-','linewidth',2)
legend('Coef. Politrópico: Liq. 1000 / Gás 2.0','Coef. Politrópico: Liq. 1000 / Gás 1.2','Coef. Politrópico: Liq. 100 / Gás 1.2') 
he(3)=ylabel('r_G');
he(4)=xlabel('Alfa_G');

set(ht, 'fontsize', 14,'FontWeight','bold')
set(he, 'fontsize', 12)
