function analisaFatoresAtrito
% Função que plota os fatores de atrito.

Re = 10.^linspace(3,8,1000);
Relam = 10.^linspace(3,log10(3e3),1000);
Returb = 10.^linspace(log10(7e3),8,1000);

figure,
loglog(Re,fator(Re,1e-5,1),...
    Re,fator(Re,1e-4,1),...
    Re,fator(Re,1e-3,1),...
    Re,fator(Re,1e-2,1),...
    Relam,fator(Relam,0,6),'k--',...
    Returb,fator(Returb,0,7),'k-',...
    'linewidth',2)
axis tight
%grid on
xlabel('Re','fontsize',12)
ylabel('f','fontsize',12)
title('Fatores de Atrito de Fanning','fontsize',14)
legend('Churchill - \epsilon/D=0.00001',...
    'Churchill - \epsilon/D=0.0001',...
    'Churchill - \epsilon/D=0.001',...
    'Churchill - \epsilon/D=0.01',...
    'Taitel para regime laminar',...
    'Taitel para regime turbulento')
set(gca,'Fontsize',12)

figure,
loglog(Re,fator(Re,1e-3,1),'-.',...
    Relam,fator(Relam,0,6),'k--',...
    'linewidth',2)
axis tight
%grid on
xlabel('Re','fontsize',12)
ylabel('f','fontsize',12)
title('Fatores de Atrito de Fanning','fontsize',14)
legend('Churchill - \epsilon/D=0.0001',...
    'Taitel para regime laminar')
set(gca,'Fontsize',12)

pause

figure,
loglog(Re,fator(Re,1e-3,1),'-.',...
    Re,fator(Re,1e-3,3),'--',...
    Re,fator(Re,1e-3,4),'-',...
    'linewidth',2)
axis tight
%grid on
xlabel('Re','fontsize',12)
ylabel('f','fontsize',12)
title('Fatores de Atrito de Fanning','fontsize',14)
legend('Churchill (1977)',...
       'Haaland (1983)',...
       'Colebrook (1939)',...
    'Churchill para rugosidade 0.01')
set(gca,'Fontsize',12)
ylim([10^-2.4 10^-2])
xlim([10^4 10^8])

pause

vSG = 10.^linspace(0,2,1000);
ix = vSG>5;
f1 = ones(size(vSG));
f2 = f1;
f3 = f1;
f4 = f1;
f5 = f1;

f1(ix) = 1+15.*(vSG(ix)./5-1).*(0.1).^0.5;
f2(ix) = 1+15.*(vSG(ix)./5-1).*(0.2).^0.5;
f3(ix) = 1+15.*(vSG(ix)./5-1).*(0.5).^0.5;

figure,
loglog(vSG,f1,'-',vSG,f2,'-.',vSG,f3,'--','linewidth',2)
axis tight
%grid on
xlabel('Vel. Sup. Gás (m/s)','fontsize',12)
ylabel('fI/fG','fontsize',12)
title('Razão Fator Atrito Interface por Fator Atrito Gás','fontsize',14)
legend('h/D = 0.1','h/D = 0.2','h/D = 0.5')
set(gca,'Fontsize',12)


ReG = 10.^linspace(3,6,1000);
f1 = 1+300.*((1+17500./ReG).*(0.1)-0.0015);
f2 = 1+300.*((1+17500./ReG).*(0.2)-0.0015);
f3 = 1+300.*((1+17500./ReG).*(0.5)-0.0015);

figure,
loglog(ReG,f1,'-',ReG,f2,'-.',ReG,f3,'--','linewidth',2)
axis tight
%grid on
xlabel('Reynolds Fase Gas','fontsize',12)
ylabel('fI/fG','fontsize',12)
title('Razão Fator Atrito Interface por Fator Atrito Gás','fontsize',14)
legend('h/D = 0.1','h/D = 0.2','h/D = 0.5')
set(gca,'Fontsize',12)





function [f]=fator(Re,rugo,correlacao)
% Geraçao dos fatores de atrito de fanning para as fases, onde:
% tensao_G  = -FI_G.*(f_G.*densG.*(v_G).^2./2).
% Sendo o fator de atrito de fanning (f) corresponde a 1/4 do fator de
% atrito de Darcy.
% Lembrando:
% Em regime laminar, f(Fanning) = 16/Re; f(Darcy) = 64/Re;

D=1;

switch correlacao
    case 1 % Churchill (1977)
        % Churchill, S. W. Friction-Factor Equation Spans All Fluid-Flow
        % Regimes. Chemical Engineering, v. 84, n. 24, 0. 91-92, 1977.
        A=(2.457*log(1./(0.27*rugo./D+(7./Re).^0.9))).^16;
        B=(37530./Re).^16;
        f=2.*((8./Re).^12+1./(A+B).^1.5).^(1./12);
    case 2 % Eck (1973)
        % Eck B., (1973) “Technische Stromungslehre”, Springer,
        % New York, 1973.
        f=0.0625./(log10(15./Re+rugo./D./3.715)).^2;
    case 3 % Haaland (1983)
        % Haaland S. E., (1983) “Simple and Explicit Formulas for the
        % Friction Factor in Turbulent Pipe Flow”. Journal of Fluids
        % Engineering, 105, 89-90.
        f=0.07716./(log10(6.9./Re+(rugo./D./3.7).^1.11)).^2;
    case 4 % Colebrook (1939)
        % Colebrook C. F., (1939) “Turbulent Flow in Pipes, With
        % Particular Reference to the Transition between the Smooth and
        % Rough Pipe Laws”. J. Inst. Civ. Eng., 11, 133-156,London, 1939.
        A=(2.457*log(1./(0.27*rugo./D+(7./Re).^0.9))).^16;
        B=(37530./Re).^16;
        f=2.*((8./Re).^12+1./(A+B).^1.5).^(1./12);
        for i=1:size(Re,1)
            for j=1:size(Re,2)
                f(i,j)=fzero(@(x) 1+2*x.^0.5.*log10(2.51./Re(i,j)./...
                    (x.^0.5)+rugo./3.715./D), f(i,j).*4);
            end
        end
        f=f./4; % de Darcy para Fanning
        
    case 5 % Moody (1947)
        % Moody L. F., (1947) “An Approximate Formula for Pipe Friction
        % Factors”. Transactions ASME, 69, 1005.
        f=0.001375.*(1+(1E6.*(0.02.*rugo./D+1./Re)).^(1/3));
        
    case 6 % Taitel Dukler (1976) para regime laminar
        f=16.*Re.^(-1);

    case 7 % Taitel Dukler (1976) para regime turbulento
        f=0.046.*Re.^(-0.2);
        
end
