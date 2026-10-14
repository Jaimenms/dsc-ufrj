function apresenta_galerkin
% Rotina par teste e apresentação do método de galerkin

% Gerando gráfico da funçao base pulso triangular (x)
hk = 1;
j = 2;
x = ((j-2).*hk):0.1:((j+2).*hk);
fi = base_triangular_para_independente_x(x,j,hk);
x2 = [(j-1).*hk (j).*hk (j+1).*hk];
fi2 = base_triangular_para_independente_x(x2,j,hk);
figure,plot(x,fi,'--','linewidth',3)
hold on
plot(x2,fi2,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('\phij (x)')
ylim([-0.1 1.1])
text((j-1+0.2).*hk,0.1,'Elemento j')
text((j+0.2).*hk,0.1,'Elemento j+1')
set(gca,'XTick',x2)
set(gca,'XTickLabel',{'Nó j-1','Nó j','Nó j+1'})
set(gca,'YTick',[0 1])
grid on
title('Função Base Triangular','FontSize',14)

pause 

% Gerando gráfico da funçao base quadratica por partes (x)
hk = 1;
j = 1;
x = ((j-3).*hk):0.1:((j+3).*hk);
fi = base_quadratica(x,j,hk);
x2 = [(j-3).*hk (j-2).*hk (j-1).*hk (j).*hk (j+1).*hk (j+2).*hk (j+3).*hk];
fi2 = base_quadratica(x2,j,hk);
figure,plot(x,fi,'--','linewidth',3)
hold on
plot(x2,fi2,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('\phij (x)')
ylim([-0.1 1.1])
text((j-1+0.2).*hk,0.1,'Elem. j')
text((j+0.2).*hk,0.1,'Elem. j+1')
set(gca,'XTick',x2)
set(gca,'XTickLabel',{'Nó j-3','Nó j-2','Nó j-1','Nó j','Nó j+1','Nó j+2'})
set(gca,'YTick',[0 1])
grid on
title('Função Base Quadrática Por Partes','FontSize',14)

pause 



% Gerando gráfico da funçao base pulso triangular (teta)
hk = 1;
j = 2;
x = ((j-2).*hk):0.1:((j+2).*hk);
fi = base_triangular_para_independente_x(x,j,hk);
x2 = [(j-1).*hk (j).*hk (j+1).*hk];
fi2 = base_triangular_para_independente_x(x2,j,hk);
figure,plot(x,fi,'--','linewidth',3)
hold on
plot(x2,fi2,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('\theta')
ylabel('\phi(\theta)')
ylim([-0.1 1.1])
set(gca,'XTick',x2)
set(gca,'XTickLabel',{'-1','0','1'})
set(gca,'YTick',[0 1])
grid on
title('Função Base Triangular','FontSize',14)

% Gerando gráfico da discretização de um trecho de linha
hk = 1;
x=[];
fi=[];
x2=[];
fi2=[];
for j=0:10
    x(j+1,:) = ((j-1).*hk):0.1:((j+1).*hk);
    fi(j+1,:) = base_triangular_para_independente_x(x(j+1,:),j,hk);
    x2(j+1,:) = [(j-1).*hk (j).*hk (j+1).*hk];
    fi2(j+1,:) = base_triangular_para_independente_x(x2(j+1,:),j,hk);
end
x([3;4;5;7;8;9],:)=[];
x2([3;4;5;7;8;9],:)=[];
fi([3;4;5;7;8;9],:)=[];
fi2([3;4;5;7;8;9],:)=[];
figure,plot(x',fi','b--','linewidth',3)
hold on
plot(x2',fi2','or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('Nó')
ylabel('\phi(x)')
ylim([-0.1 1.1])
xlim([0 10])
set(gca,'XTick',[0:10])
set(gca,'XTickLabel',{'0','1','2','...','j-2','j-1','j','...','Nk-2','Nk-1','Nk'})
set(gca,'YTick',[0 1])
grid on
title('Discretização de um trecho de linha','FontSize',14)
text(0.5.*hk+(1-1)*hk,0,'1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(2-1)*hk,0,'2','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(5-1)*hk,0,'j-1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(6-1)*hk,0,'j','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(9-1)*hk,0,'Nk-1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(10-1)*hk,0,'Nk','VerticalAlignment','bottom','HorizontalAlignment','center')
pause
% Gerando gráfico da discretização de um trecho de linha
hk = 1;
x=[];
fi=[];
x2=[];
fi2=[];
for j=0:10
    x(j+1,:) = ((j-1).*hk):0.1:((j+1).*hk);
    fi(j+1,:) = base_triangular_para_independente_x(x(j+1,:),j,hk);
    x2(j+1,:) = [(j-1).*hk (j).*hk (j+1).*hk];
    fi2(j+1,:) = base_triangular_para_independente_x(x2(j+1,:),j,hk);
end
x([3;4;5;7;8;9],:)=[];
x2([3;4;5;7;8;9],:)=[];
fi([3;4;5;7;8;9],:)=[];
fi2([3;4;5;7;8;9],:)=[];
figure,plot(x',fi','b--','linewidth',3)
hold on
plot(x2',fi2','or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('\theta')
ylabel('\phi(x)')
ylim([-0.1 1.1])
xlim([0 10])
set(gca,'XTick',[0:10])
set(gca,'XTickLabel',{'0','hk','2hk','...','(j-2)hk','(j-1)hk','jhk','...','(Nk-2)hk','(Nk-1)hk','Lk'})
set(gca,'YTick',[0 1])
grid on
title('Discretização de um trecho de linha','FontSize',14)
text(0.5.*hk+(1-1)*hk,0,'1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(2-1)*hk,0,'2','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(5-1)*hk,0,'j-1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(6-1)*hk,0,'j','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(9-1)*hk,0,'Nk-1','VerticalAlignment','bottom','HorizontalAlignment','center')
text(0.5.*hk+(10-1)*hk,0,'Nk','VerticalAlignment','bottom','HorizontalAlignment','center')


% Gerando aproximação do vetor de estados por truncamento da série de
% fourier com bases pulso riangulares

% Seja um fetor Y de estados nodais linear em relação a x.
% Y=a*x+b;
a=1;
b=0.1;
hk = 1;
Nk = 4;
x = 0:hk:(Nk.*hk);
x2= 0:0.1:(Nk.*hk);

Y = a.*x+b;
Y2 = a.*x2+b;

Yast=0;
Yast2=0;
for j=0:Nk
    fi = base_triangular_para_independente_x(x,j,hk);
    fi2 = base_triangular_para_independente_x(x2,j,hk);
    Yast=Yast+Y.*fi;
    Yast2=Yast2+Y2.*fi2;
end

figure
subplot(2,1,1)
plot(x2,Yast2,'b--','linewidth',3)
hold on
plot(x,Y,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
plot(x2,Y2,'k-')
xlabel('x')
ylabel('Y(x)')
subplot(2,1,2)
plot(x2,(Yast2-Y2)./Y2.*100,'b--','linewidth',3)
hold on
plot(x,(Yast-Y)./Y.*100,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('Erro Y(x)')

% Seja um fetor Y de estados nodais linear em relação a x.
% Y=a*x^2+b;
a=1;
b=0.1;
hk = 1;
Nk = 4;
x = 0:hk:(Nk.*hk);
x2= 0:0.1:(Nk.*hk);

Y = a.*x.^2+b;
Y2 = a.*x2.^2+b;

Yast=0;
Yast2=0;
for j=0:Nk
    fi = base_triangular_para_independente_x(x,j,hk);
    fi2 = base_triangular_para_independente_x(x2,j,hk);
    Yast=Yast+Y.*fi;
    Yast2=Yast2+Y2.*fi2;
end

figure
subplot(2,1,1)
plot(x2,Yast2,'b--','linewidth',3)
hold on
plot(x,Y,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
plot(x2,Y2,'k-')
xlabel('x')
ylabel('Y(x)')
subplot(2,1,2)
plot(x2,(Yast2-Y2)./Y2.*100,'b--','linewidth',3)
hold on
plot(x,(Yast-Y)./Y.*100,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('Erro Y(x)')

% Seja um fetor Y de estados nodais linear em relação a x.
% Y=a*sin(c*x)+b;
a=1;
b=1;
c=10;
hk = 1;
Nk = 4;
x = 0:hk:(Nk.*hk);
x2= 0:0.1:(Nk.*hk);

Y = a.*sin(c.*x)+b;
Y2 = a.*sin(c.*x2)+b;

Yast=0;
Yast2=0;
for j=0:Nk
    fi = base_triangular_para_independente_x(x,j,hk);
    fi2 = base_triangular_para_independente_x(x2,j,hk);
    Yast=Yast+Y.*fi;
    Yast2=Yast2+Y2.*fi2;
end

figure
subplot(2,1,1)
plot(x2,Yast2,'b--','linewidth',3)
hold on
plot(x,Y,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
plot(x2,Y2,'k-')
xlabel('x')
ylabel('Y(x)')
subplot(2,1,2)
plot(x2,(Yast2-Y2)./Y2.*100,'b--','linewidth',3)
hold on
plot(x,(Yast-Y)./Y.*100,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('Erro Y(x)')


% Gerando gráfico da funçao base pulso triangular (x) ao quadrado
hk = 1;
j = 2;
x = ((j-2).*hk):0.1:((j+2).*hk);
fi = base_triangular_para_independente_x(x,j,hk);
x2 = [(j-1).*hk (j).*hk (j+1).*hk];
fi2 = base_triangular_para_independente_x(x2,j,hk);
figure,plot(x,fi.^2,'--','linewidth',3)
hold on
plot(x2,fi2.^2,'or',...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',6)
xlabel('x')
ylabel('\phi^2(x)')
ylim([-0.1 1.1])
set(gca,'XTick',x2)
set(gca,'XTickLabel',{'(j-1)*hk','j*hk','(j+1)*hk'})
set(gca,'YTick',[0 1])
grid on
title('Função Base Triangular ao Quadrado','FontSize',14)


function teta = x2fi(x,j,hk)
teta = (x-j.*hk)./hk;

function fi = base_triangular_para_independente_x(x,j,hk)

nx = length(x);
for i=1:nx
    if x(i)<(j-1).*hk
        fi(i)=0;
    elseif x(i)>(j+1).*hk
        fi(i)=0;
    elseif (x(i)<=j.*hk)&(x(i)>=(j-1).*hk)
        fi(i)=(x(i)-(j-1).*hk)./hk;
    else
        fi(i)=-(x(i)-(j+1).*hk)./hk;
    end
end

function fi = base_triangular_para_independente_teta(teta)

nteta = length(teta);
for i=1:nteta
    if teta(i)<-1
        fi(i)=0;
    elseif teta(i)>1
        fi(i)=0;
    elseif (teta(i)<=0)&(teta(i)>=-1)
        fi(i)=1+teta(i);
    else
        fi(i)=1-teta(i);
    end
end

function fi = base_quadratica(x2,j,hk)
alfa=0.1;
nx = length(x2);

x = x2-j*hk;

for i=1:nx
    if x(i)<-2*hk
        fi(i)=0;
    elseif x(i)<=-hk
        fi(i)=8*alfa-(12*alfa/hk).*(-x(i))+(4*alfa/hk^2).*(x(i).^2);
    elseif x(i)<=0
        fi(i)=1+(4*alfa/hk-1).*(-x(i))-(4*alfa/hk^2).*(x(i).^2);
    elseif x(i)<=hk
        fi(i)=1+(4*alfa/hk-1).*(+x(i))-(4*alfa/hk^2).*(x(i).^2);
    elseif x(i)<=2.*hk
        fi(i)=8*alfa-(12*alfa/hk).*(+x(i))+(4*alfa/hk^2).*(x(i).^2);
    else
        fi(i)=0;
    end
end

