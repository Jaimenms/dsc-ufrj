function plota_resultados(tipo)
% Função que plota os resultados da simulação.

global status maxstatus dados eixos saidas

status=[];
maxstatus=[];

switch tipo
    case 1 % Plota rede de escoamento
        
        m=length(dados.tubo);
        a=figure;
        b=figure;
        c=figure;
        d=figure;
        
        for i=1:m
            x = dados.tubo{i}.x;
            y = dados.tubo{i}.y;
            z = dados.tubo{i}.z;
            nx = length(x);
            
            figure(a)
            plot3(x,y,z,'linewidth',3)
            hold on
            plot3([x(1) x(end)],[y(1) y(end)],[z(1) z(end)],'ob','markersize',10,'MarkerFaceColor',[0 0 0])
            title('3D','FontSize',12)
            xlabel('x')
            ylabel('y')
            zlabel('z')
            axis equal
            grid on
            
            figure(b)
            plot(x,y,'linewidth',3)
            hold on
            plot([x(1) x(end)],[y(1) y(end)],'ob','markersize',10,'MarkerFaceColor',[0 0 0])
            title('Vista X-Y','FontSize',12)
            xlabel('x')
            ylabel('y')
            axis equal
            grid on
            
            figure(c)
            plot(x,z,'linewidth',3)
            hold on
            plot([x(1) x(end)],[z(1) z(end)],'ob','markersize',10,'MarkerFaceColor',[0 0 0])
            title('Vista X-Z','FontSize',12)
            xlabel('x')
            ylabel('z')
            axis equal
            grid on
            
            figure(d)
            plot(y,z,'linewidth',3)
            hold on
            hold on
            plot([y(1) y(end)],[z(1) z(end)],'ob','markersize',10,'MarkerFaceColor',[0 0 0])
            title('Vista Y-Z','FontSize',12)
            xlabel('y')
            ylabel('z')
            axis tight
            grid on
            
        end
        
       
        
        
        
        
        
    case 2 % Plota rede de escoamento com nós

        figure
        m=length(dados.tubo);
        
        for i=1:m
            x = dados.tubo{i}.x;
            y = dados.tubo{i}.y;
            z = dados.tubo{i}.z;
            nx = length(x);
            deltaX = x(2:nx)-x(1:(nx-1));
            deltaY = y(2:nx)-y(1:(nx-1));
            deltaZ = z(2:nx)-z(1:(nx-1));
            vNk=dados.tubo{i}.numeroElementos;
            hk=[];
            Ank=[];
            Nohx=x(:,1);
            Nohy=y(:,1);
            Nohz=z(:,1);
            for j=1:(nx-1)
                dL=sqrt(deltaX(j).^2+deltaY(j).^2+deltaZ(j).^2);
                dAn=asin(deltaZ(j)./dL);
                dh = dL./vNk(j);

                hk = [hk dh.*ones(1,vNk(j))];
                Ank = [Ank dAn.*ones(1,vNk(j))];

                Nohx=[Nohx x(j)+deltaX(j).*[(1/vNk(j)):(1/vNk(j)):1]];
                Nohy=[Nohy y(j)+deltaY(j).*[(1/vNk(j)):(1/vNk(j)):1]];
                Nohz=[Nohz z(j)+deltaZ(j).*[(1/vNk(j)):(1/vNk(j)):1]];
            end

            subplot(2,2,1)
            plot3(Nohx,Nohy,Nohz,'ob-','linewidth',3,'markersize',5,'MarkerFaceColor',[0 0 1])
            hold on
            title('3D','FontSize',12)
            xlabel('x')
            ylabel('y')
            zlabel('z')
            axis tight
            grid on
            
            subplot(2,2,2)
            plot(Nohx,Nohy,'ob-','linewidth',3,'markersize',5,'MarkerFaceColor',[0 0 1])
            hold on
            title('Vista X-Y','FontSize',12)
            xlabel('x')
            ylabel('y')
            axis tight
            grid on
            
            subplot(2,2,3)
            plot(Nohx,Nohz,'ob-','linewidth',3,'markersize',5,'MarkerFaceColor',[0 0 1])
            hold on
            title('Vista X-Z','FontSize',12)
            xlabel('x')
            ylabel('z')
            axis tight
            grid on
            
            subplot(2,2,4)
            plot(Nohy,Nohz,'ob-','linewidth',3,'markersize',5,'MarkerFaceColor',[0 0 1])
            hold on
            title('Vista Y-Z','FontSize',12)
            xlabel('y')
            ylabel('z')
            axis tight
            grid on
            
        end
        
       
        case 3
                
        m=length(dados.tubo);
        for i=1:m
            gera_radiografia(i,dados,eixos,saidas,1)
        end
        
        
        case 4
                
        m=length(dados.tubo);
        for i=1:m
            gera_radiografia(i,dados,eixos,saidas,1)
            gera_acao_grafico(i)
        end

    case 5
        
        gera_topografico(dados,eixos,1,saidas)
        gera_acao_grafico2
        
    case 6
        
        gera_topografico(dados,eixos,1,saidas)

    case 7
        
        gera_radiografia_all(dados,eixos,saidas,1)

    case 8
        
        gera_radiografia_all(dados,eixos,saidas,1)
        gera_acao_grafico_all
        
end



function gera_topografico(dados,eixos,k,saidas)
global status maxstatus

tipo = 2;

h=findobj('Userdata',['Topografico']);
if isempty(h)
    figure
    set(gcf,'Userdata',['Topografico'])
else
    delete(findall(h,'type','axes'))
end


t = eixos.tubo.Y.valor{1};

if k > length(t)
    k=length(t);
end
maxstatus{1}=length(t);

if k<=0, k=1; end
status{1} = k;
t = t(k);

title(['Tempo: ' num2str(t) ' s'],'fontsize',14)

minP=inf;
maxP=0;
minalfaG=inf;
maxalfaG=0;
minqG=inf;
maxqG=0;
minqL=inf;
maxqL=0;
for i=1:length(dados.tubo)
    P = saidas.tubo{i}.pressao;
    alfaG = saidas.tubo{i}.alfaG;
    qL = saidas.tubo{i}.vazaoMassaL;
    qG = saidas.tubo{i}.vazaoMassaG;
    
    minP = min(minP,min(min(P)));
    maxP = max(maxP,max(max(P)));
    
    minalfaG = min(minalfaG,min(min(alfaG)));
    maxalfaG = max(maxalfaG,max(max(alfaG)));

    minqL = min(minqL,min(min(qL)));
    maxqL = max(maxqL,max(max(qL)));

    minqG = min(minqG,min(min(qG)));
    maxqG = max(maxqG,max(max(qG)));
end

for i=1:length(dados.tubo)
    
    P = saidas.tubo{i}.pressao(:,k);
    alfaG = saidas.tubo{i}.alfaG(:,k);
    qL = saidas.tubo{i}.vazaoMassaL(:,k);
    qG = saidas.tubo{i}.vazaoMassaG(:,k);

    fP = (P-minP)./(maxP-minP);
    iP = round(fP.*64);
    iP = max(1,iP);

    falfaG = (alfaG-minalfaG)./(maxalfaG-minalfaG);
    ialfaG = round(falfaG.*64);
    ialfaG = max(1,ialfaG);

    fqL = (qL-minqL)./(maxqL-minqL);
    iqL = round(fqL.*64);
    iqL = max(1,iqL);

    fqG = (qG-minqG)./(maxqG-minqG);
    iqG = round(fqG.*64);
    iqG = max(1,iqG);

    a=colormap('jet');
    
    x = dados.tubo{i}.x;
    y = dados.tubo{i}.y;
    z = dados.tubo{i}.z;
    nx = length(x);
    deltaX = x(2:nx)-x(1:(nx-1));
    deltaY = y(2:nx)-y(1:(nx-1));
    deltaZ = z(2:nx)-z(1:(nx-1));
    vNk=dados.tubo{i}.numeroElementos;
    hk=[];
    Ank=[];
    Nohx=x(:,1);
    Nohy=y(:,1);
    Nohz=z(:,1);
    for j=1:(nx-1)
        Nohx=[Nohx x(j)+deltaX(j).*[(1/vNk(j)):(1/vNk(j)):1]];
        Nohy=[Nohy y(j)+deltaY(j).*[(1/vNk(j)):(1/vNk(j)):1]];
        Nohz=[Nohz z(j)+deltaZ(j).*[(1/vNk(j)):(1/vNk(j)):1]];
    end

    for j=2:size(Nohx,2)
        diam = dados.tubo{i}.diametro;
        CylfP = round(mean(iP(j),iP(j-1)));
        CylfalfaG = round(mean(ialfaG(j),ialfaG(j-1)));
        CylfqL = round(mean(iqL(j),iqL(j-1)));
        CylfqG = round(mean(iqG(j),iqG(j-1)));
        subplot(2,2,1)
        plot3([Nohx(j-1) Nohx(j)],[Nohy(j-1) Nohy(j)],[Nohz(j-1) Nohz(j)],'linewidth',6,'Color',a(CylfP,:));
        
        hold on
        axis tight
        xlabel('x (m)')
        ylabel('y (m)')
        zlabel('z (m)')
        title(['Pressão / Tempo: ' num2str(k) ' s'])

        subplot(2,2,2)
        plot3([Nohx(j-1) Nohx(j)],[Nohy(j-1) Nohy(j)],[Nohz(j-1) Nohz(j)],'linewidth',6,'Color',a(CylfalfaG,:));
        hold on
        xlabel('x (m)')
        ylabel('y (m)')
        zlabel('z (m)')
        title('Fr. Area Gas')
        axis tight
        
        subplot(2,2,3)
        plot3([Nohx(j-1) Nohx(j)],[Nohy(j-1) Nohy(j)],[Nohz(j-1) Nohz(j)],'linewidth',6,'Color',a(CylfqL,:));
        hold on
        axis tight
        xlabel('x (m)')
        ylabel('y (m)')
        zlabel('z (m)')
        title('Vazão Líquido')
        
        subplot(2,2,4)
        plot3([Nohx(j-1) Nohx(j)],[Nohy(j-1) Nohy(j)],[Nohz(j-1) Nohz(j)],'linewidth',6,'Color',a(CylfqG,:));
        hold on
        xlabel('x (m)')
        ylabel('y (m)')
        zlabel('z (m)')
        title('Vazão Gás')
        axis tight
        
    end
end
drawnow



function gera_radiografia(i,dados,eixos,saidas,k)
global status maxstatus

h=findobj('Userdata',['Radiografia' num2str(i)]);
if isempty(h)
    figure
    set(gcf,'Userdata',['Radiografia' num2str(i)])
else
    delete(findall(h,'type','axes'))
end


t = eixos.tubo.Y.valor{i};

if k > length(t)
    k=length(t);
end
maxstatus{i}=length(t);

if k<=0, k=1; end
status{i} = k;
t = t(k);

title(['Tubulação ' num2str(i) ' / Tempo: ' num2str(t) ' s'],'fontsize',14)
x = eixos.tubo.X.valor{i};
y = saidas.tubo{i}.hD;
c1 = saidas.tubo{i}.vazaoMassaL;
c2 = saidas.tubo{i}.vazaoMassaG;

y = y(:,k);
c1 = c1(:,k);
c2 = c2(:,k);

minc1 = min(min(c1));
maxc1 = max(max(c2));
minc2 = min(min(c2));
maxc2 = max(max(c2));

fc1 = (c1-minc1)./(maxc1-minc1);
fc2 = (c2-minc2)./(maxc2-minc2);

C1 = [fc1 zeros(size(fc1)) 1-fc1];
C2 = [fc2 zeros(size(fc2)) 1-fc2];


C1 = min(C1,1);
C1 = max(C1,0);
C2 = min(C2,1);
C2 = max(C2,0);

PX=[];
PY=[];
tcolor=[];
for j=2:length(x)
    px = [x(j-1); x(j-1); x(j); x(j)];
    py = [0; y(j-1); y(j); 0];
    PX = [PX px];
    PY = [PY py];
    tcolor(1,j-1,1:3) = C1(j-1,:);
    tcolor(2,j-1,1:3) = C1(j-1,:);
    tcolor(3,j-1,1:3) = C1(j,:);
    tcolor(4,j-1,1:3) = C1(j,:);

end
patch(PX,PY,zeros(size(PY)),tcolor)

PX=[];
PY=[];
tcolor=[];
for j=2:length(x)
    px = [x(j-1); x(j-1); x(j); x(j)];
    py = [y(j-1); 1 ; 1 ;y(j)];
    PX = [PX px];
    PY = [PY py];
    tcolor(1,j-1,1:3) = C2(j-1,:);
    tcolor(2,j-1,1:3) = C2(j-1,:);
    tcolor(3,j-1,1:3) = C2(j,:);
    tcolor(4,j-1,1:3) = C2(j,:);

end
patch(PX,PY,zeros(size(PY)),tcolor)

xlabel(eixos.tubo.X.nome)
ylabel('Seção do Duto')
xlim([min(x) max(x)])
ylim([0 1])
set(gca,'YGrid','on')


PX=[];
PY=[];
tcolor=[];
dx=(max(x)-min(x))./length(x)./5;
for j=1:length(x)
    if c1>0
        dy = 0.02;
        px = [x(j)-dx; x(j)-dx; x(j)+dx];
        py = [y(j)./2-dy; y(j)./2+dy ; y(j)./2];
    else
        dy = 0.02;
        px = [x(j)+dx; x(j)+dx; x(j)-dx];
        py = [y(j)./2-dy; y(j)./2+dy ; y(j)./2];
    end
    tcolor(1,j,1:3) = [1 1 0];
    tcolor(2,j,1:3) = [1 1 0];
    tcolor(3,j,1:3) = [1 1 0];
    PX = [PX px];
    PY = [PY py];
end

patch(PX,PY,zeros(size(PY)),tcolor)

PX=[];
PY=[];
tcolor=[];
dx=(max(x)-min(x))./length(x)./5;
for j=1:length(x)
    if c2>0
        dy = 0.02;
        px = [x(j)-dx; x(j)-dx; x(j)+dx];
        py = y(j)+(1-y(j))/2+[-dy; +dy ; 0];
    else
        dy = 0.02;
        px = [x(j)+dx; x(j)+dx; x(j)-dx];
        py = y(j)+(1-y(j))/2+[-dy; +dy ; 0];
    end
    tcolor(1,j,1:3) = [1 1 0];
    tcolor(2,j,1:3) = [1 1 0];
    tcolor(3,j,1:3) = [1 1 0];
    PX = [PX px];
    PY = [PY py];
end

patch(PX,PY,zeros(size(PY)),tcolor)

function gera_radiografia_all(dados,eixos,saidas,k)
global status maxstatus

h=findobj('Userdata','Radiografia');
if isempty(h)
    figure
    set(gcf,'Userdata','Radiografia')
else
    delete(findall(h,'type','axes'))
end

m = length(dados.tubo);

minc1=+inf;
minc2=+inf;
maxc1=-inf;
maxc2=-inf;
for i=1:m
    c1 = saidas.tubo{i}.vazaoMassaL;
    c2 = saidas.tubo{i}.vazaoMassaG;

    minc1 = min(minc1,min(min(c1)));
    maxc1 = max(maxc1,max(max(c2)));
    minc2 = min(minc2,min(min(c2)));
    maxc2 = max(maxc2,max(max(c2)));
end


for i=1:m
    subplot(1,m,i)
    
    t = eixos.tubo.Y.valor{i};

    if k > length(t)
        k=length(t);
    end
    maxstatus{i}=length(t);

    if k<=0, k=1; end
    status{i} = k;
    t = t(k);

    if i==1
        title(['Tubulação ' num2str(i) ' / Tempo: ' num2str(t) ' s'],'fontsize',14)
    else
        title(['Tubulação ' num2str(i)],'fontsize',14)
    end
    x = eixos.tubo.X.valor{i};
    y = saidas.tubo{i}.hD;

    c1 = saidas.tubo{i}.vazaoMassaL;
    c2 = saidas.tubo{i}.vazaoMassaG;

    y = y(:,k);
    c1 = c1(:,k);
    c2 = c2(:,k);

    fc1 = (c1-minc1)./(maxc1-minc1);
    fc2 = (c2-minc2)./(maxc2-minc2);

    C1 = [fc1 zeros(size(fc1)) 1-fc1];
    C2 = [fc2 zeros(size(fc2)) 1-fc2];

    
    C1 = min(C1,1);
    C1 = max(C1,0);
    C2 = min(C2,1);
    C2 = max(C2,0);
    
    PX=[];
    PY=[];
    tcolor=[];
    for j=2:length(x)
        px = [x(j-1); x(j-1); x(j); x(j)];
        py = [0; y(j-1); y(j); 0];
        PX = [PX px];
        PY = [PY py];
        tcolor(1,j-1,1:3) = C1(j-1,:);
        tcolor(2,j-1,1:3) = C1(j-1,:);
        tcolor(3,j-1,1:3) = C1(j,:);
        tcolor(4,j-1,1:3) = C1(j,:);

    end
    patch(PX,PY,zeros(size(PY)),tcolor)

    PX=[];
    PY=[];
    tcolor=[];
    for j=2:length(x)
        px = [x(j-1); x(j-1); x(j); x(j)];
        py = [y(j-1); 1 ; 1 ;y(j)];
        PX = [PX px];
        PY = [PY py];
        tcolor(1,j-1,1:3) = C2(j-1,:);
        tcolor(2,j-1,1:3) = C2(j-1,:);
        tcolor(3,j-1,1:3) = C2(j,:);
        tcolor(4,j-1,1:3) = C2(j,:);

    end
    patch(PX,PY,zeros(size(PY)),tcolor)

    xlim([min(x) max(x)])
    ylim([0 1])
    set(gca,'YGrid','on')

    PX=[];
    PY=[];
    tcolor=[];
    dx=(max(x)-min(x))./length(x)./5;
    for j=1:length(x)
        if c1>0
            dy = 0.02;
            px = [x(j)-dx; x(j)-dx; x(j)+dx];
            py = [y(j)./2-dy; y(j)./2+dy ; y(j)./2];
        else
            dy = 0.02;
            px = [x(j)+dx; x(j)+dx; x(j)-dx];
            py = [y(j)./2-dy; y(j)./2+dy ; y(j)./2];
        end
        tcolor(1,j,1:3) = [1 1 0];
        tcolor(2,j,1:3) = [1 1 0];
        tcolor(3,j,1:3) = [1 1 0];
        PX = [PX px];
        PY = [PY py];
    end

    patch(PX,PY,zeros(size(PY)),tcolor)

    PX=[];
    PY=[];
    tcolor=[];
    dx=(max(x)-min(x))./length(x)./5;
    for j=1:length(x)
        if c2>0
            dy = 0.02;
            px = [x(j)-dx; x(j)-dx; x(j)+dx];
            py = y(j)+(1-y(j))/2+[-dy; +dy ; 0];
        else
            dy = 0.02;
            px = [x(j)+dx; x(j)+dx; x(j)-dx];
            py = y(j)+(1-y(j))/2+[-dy; +dy ; 0];
        end
        tcolor(1,j,1:3) = [1 1 0];
        tcolor(2,j,1:3) = [1 1 0];
        tcolor(3,j,1:3) = [1 1 0];
        PX = [PX px];
        PY = [PY py];
    end

    patch(PX,PY,zeros(size(PY)),tcolor)

    set(gca,'XTickLabel','')
    set(gca,'YTickLabel','')
    
end


function gera_acao_grafico(j)
uicontrol(gcf,'style','pushbutton','string','<<<',...
    'position',[20 5 30 20],'callback',{@callbackPlot,1,j});
uicontrol(gcf,'style','pushbutton','string',' <<',...
    'position',[50 5 30 20],'callback',{@callbackPlot,5,j});
uicontrol(gcf,'style','pushbutton','string','  <',...
    'position',[80 5 30 20],'callback',{@callbackPlot,2,j});


uicontrol(gcf,'style','pushbutton','string','>  ',...
    'position',[110 5 30 20],'callback',{@callbackPlot,3,j});
uicontrol(gcf,'style','pushbutton','string','>> ',...
    'position',[140 5 30 20],'callback',{@callbackPlot,6,j});
uicontrol(gcf,'style','pushbutton','string','>>>',...
    'position',[170 5 30 20],'callback',{@callbackPlot,4,j});

uicontrol(gcf,'style','pushbutton','string','RR',...
    'position',[400 5 40 20],'callback',{@callbackPlot,10,j});
uicontrol(gcf,'style','pushbutton','string','PLAY',...
    'position',[440 5 40 20],'callback',{@callbackPlot,7,j});
uicontrol(gcf,'style','pushbutton','string','FF',...
    'position',[480 5 40 20],'callback',{@callbackPlot,8,j});
uicontrol(gcf,'style','pushbutton','string','PAUSE',...
    'position',[520 5 40 20],'callback',{@callbackPlot,12,j});
uicontrol(gcf,'style','pushbutton','string','fit',...
    'position',[20 25 30 40],'callback',{@callbackPlot,11,j});

function gera_acao_grafico_all
uicontrol(gcf,'style','pushbutton','string','<<<',...
    'position',[20 5 30 20],'callback',{@callbackPlot_all,1});
uicontrol(gcf,'style','pushbutton','string',' <<',...
    'position',[50 5 30 20],'callback',{@callbackPlot_all,5});
uicontrol(gcf,'style','pushbutton','string','  <',...
    'position',[80 5 30 20],'callback',{@callbackPlot_all,2});


uicontrol(gcf,'style','pushbutton','string','>  ',...
    'position',[110 5 30 20],'callback',{@callbackPlot_all,3});
uicontrol(gcf,'style','pushbutton','string','>> ',...
    'position',[140 5 30 20],'callback',{@callbackPlot_all,6});
uicontrol(gcf,'style','pushbutton','string','>>>',...
    'position',[170 5 30 20],'callback',{@callbackPlot_all,4});

uicontrol(gcf,'style','pushbutton','string','RR',...
    'position',[400 5 40 20],'callback',{@callbackPlot_all,10});
uicontrol(gcf,'style','pushbutton','string','PLAY',...
    'position',[440 5 40 20],'callback',{@callbackPlot_all,7});
uicontrol(gcf,'style','pushbutton','string','FF',...
    'position',[480 5 40 20],'callback',{@callbackPlot_all,8});
uicontrol(gcf,'style','pushbutton','string','PAUSE',...
    'position',[520 5 40 20],'callback',{@callbackPlot_all,12});
uicontrol(gcf,'style','pushbutton','string','fit',...
    'position',[20 25 30 40],'callback',{@callbackPlot_all,11});




function callbackPlot(src,eventdata,flag,j)
global parar roda status maxstatus dados eixos saidas


if flag==10|flag==7|flag==8,
    parar=0;
end

roda=1;
while roda

    if parar, break,end
    
    switch flag
        case 1
            status{j} = 1;
            roda = 0;
        case 2
            status{j} = status{j}-1;
            roda = 0;
        case 3
            status{j} = status{j}+1;
            roda = 0;
        case 4
            status{j} = inf;
            roda = 0;
        case 5
            status{j} = max(1,status{j} - round(maxstatus{j}./10));
            roda = 0;
        case 6
            status{j} = min(maxstatus{j},status{j} + round(maxstatus{j}./10));
            roda = 0;
        case 7
            status{j} = status{j}+1;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                roda =0;
                status{j} = maxstatus{j};
            end
        case 8
            status{j} = status{j}+5;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                status{j} = maxstatus{j};
                roda =0;
            end
        case 10
            status{j} = status{j}-5;
            if status{j}>1
                roda = 1;
            else
                status{j} = 1;
                roda =0;
            end
    end
    
    gera_radiografia(j,dados,eixos,saidas,status{j})   
    
    drawnow
    
end

parar=0;

function callbackPlot_all(src,eventdata,flag)
global parar roda status maxstatus dados eixos saidas

j=1;

if flag==10|flag==7|flag==8,
    parar=0;
end

roda=1;
while roda

    if parar, break,end
    
    switch flag
        case 1
            status{j} = 1;
            roda = 0;
        case 2
            status{j} = status{j}-1;
            roda = 0;
        case 3
            status{j} = status{j}+1;
            roda = 0;
        case 4
            status{j} = inf;
            roda = 0;
        case 5
            status{j} = max(1,status{j} - round(maxstatus{j}./10));
            roda = 0;
        case 6
            status{j} = min(maxstatus{j},status{j} + round(maxstatus{j}./10));
            roda = 0;
        case 7
            status{j} = status{j}+1;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                roda =0;
                status{j} = maxstatus{j};
            end
        case 8
            status{j} = status{j}+5;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                status{j} = maxstatus{j};
                roda =0;
            end
        case 10
            status{j} = status{j}-5;
            if status{j}>1
                roda = 1;
            else
                status{j} = 1;
                roda =0;
            end
    end
    
    gera_radiografia_all(dados,eixos,saidas,status{j})   
    
    drawnow
    
end

parar=0;


function gera_acao_grafico2
    j=1;
uicontrol(gcf,'style','pushbutton','string','<<<',...
    'position',[20 5 30 20],'callback',{@callbackPlot2,1,j});
uicontrol(gcf,'style','pushbutton','string',' <<',...
    'position',[50 5 30 20],'callback',{@callbackPlot2,5,j});
uicontrol(gcf,'style','pushbutton','string','  <',...
    'position',[80 5 30 20],'callback',{@callbackPlot2,2,j});


uicontrol(gcf,'style','pushbutton','string','>  ',...
    'position',[110 5 30 20],'callback',{@callbackPlot2,3,j});
uicontrol(gcf,'style','pushbutton','string','>> ',...
    'position',[140 5 30 20],'callback',{@callbackPlot2,6,j});
uicontrol(gcf,'style','pushbutton','string','>>>',...
    'position',[170 5 30 20],'callback',{@callbackPlot2,4,j});

uicontrol(gcf,'style','pushbutton','string','RR',...
    'position',[400 5 40 20],'callback',{@callbackPlot2,10,j});
uicontrol(gcf,'style','pushbutton','string','PLAY',...
    'position',[440 5 40 20],'callback',{@callbackPlot2,7,j});
uicontrol(gcf,'style','pushbutton','string','FF',...
    'position',[480 5 40 20],'callback',{@callbackPlot2,8,j});
uicontrol(gcf,'style','pushbutton','string','PAUSE',...
    'position',[520 5 40 20],'callback',{@callbackPlot2,12,j});
uicontrol(gcf,'style','pushbutton','string','fit',...
    'position',[20 25 30 40],'callback',{@callbackPlot2,11,j});



function callbackPlot2(src,eventdata,flag,j)
global parar roda status maxstatus dados eixos saidas

j=1;

if flag==10|flag==7|flag==8,
    parar=0;
end

roda=1;
while roda

    if parar, break,end
    
    switch flag
        case 1
            status{j} = 1;
            roda = 0;
        case 2
            status{j} = status{j}-1;
            roda = 0;
        case 3
            status{j} = status{j}+1;
            roda = 0;
        case 4
            status{j} = inf;
            roda = 0;
        case 5
            status{j} = max(1,status{j} - round(maxstatus{j}./10));
            roda = 0;
        case 6
            status{j} = min(maxstatus{j},status{j} + round(maxstatus{j}./10));
            roda = 0;
        case 7
            status{j} = status{j}+1;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                roda =0;
                status{j} = maxstatus{j};
            end
        case 8
            status{j} = status{j}+5;
            if status{j}<maxstatus{j} 
                roda = 1;
            else
                status = maxstatus{j};
                roda =0;
            end
        case 10
            status{j} = status{j}-5;
            if status>1
                roda = 1;
            else
                status{j} = 1;
                roda =0;
            end
    end
    
    gera_topografico(dados,eixos,status{1},saidas)
end

parar=0;