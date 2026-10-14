function analisador(dados,estados,flag)
% Função que permite analisar os autovalores da matriz jacobiana das
% tubulações discretizadas.

%% Tratamentos Iniciais
clc, clear functions, warning('off')
global parametros
tic
tempo_cpu = cputime;

[parametros]=calculos_iniciais(dados);

T = estados.T;
X = estados.X;

if length(T)>1
    t = input('Escreva o instante de tempo para análise (s):');
    Xt = [interp1(T,X',t)]';
else
    t = 0;
    Xt = X;
end

switch flag
    
    case 1
        
        [J]=modelo_rede_jacobiana(t,Xt,parametros);
        
        figure
        h =bar3(J);
        ht(1) = title('Análise da Jacobiana');
        he(1) = xlabel('d/dx');
        he(2) = ylabel('F');
        he(3) = zlabel('dF/dx');

        figure
        h =bar(J,'stacked');
        ht(2) = title('Análise da Jacobiana');
        he(4) = xlabel('F');
        he(5) = ylabel('sum(dF/dx)');
       
        set(ht, 'fontsize', 14,'FontWeight','bold')
        set(he, 'fontsize', 12)

    case 2
        
        [J]=modelo_rede_jacobiana(t,Xt,parametros);
        
        E = eig(J,'nobalance');
        E = sort(E);
        Epos = E;
        Eneg = E;
        Epos(E<0)=nan;
        Eneg(E>=0)=nan;
        npos = sum(E>=0);
        nneg = sum(E<0);
        
        Ermax = max(abs(real(E)));
        Eimax = max(abs(imag(E)));
        
        figure
        plot([0 0],[-Eimax Eimax],'-k')
        hold on
        plot([-Ermax Ermax],[0 0],'-k')
        h(1)=plot(real(Epos),imag(Epos),'ob','MarkerSize',8,'MarkerEdgeColor',[0 0 1],'MarkerFaceColor',[0 0 1]);
        set(gca,'box','off')
        h(2)=plot(real(Eneg),imag(Eneg),'or','MarkerSize',8,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0]);
        axis tight
        
        legend(h,['Valores Positivos (' num2str(npos) ')'], ['Valores Negativos (' num2str(nneg) ')'])
        
        ht(1) = title('Autovalores da Matriz Jacobiana');
        he(1) = xlabel('Real');
        he(2) = ylabel('Imag');

        set(ht, 'fontsize', 14,'FontWeight','bold')
        set(he, 'fontsize', 12)

        
end
