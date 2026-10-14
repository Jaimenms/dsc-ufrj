function [f,Re]=fator(v,dens,visc,D,rugo)
% Função que gera fatores de atrito de fanning para as fases

% tensao_G  = -FI_G.*(f_G.*densG.*(v_G).^2./2).
% Sendo o fator de atrito de fanning (f) corresponde a 1/4 do fator de
% atrito de Darcy.
% Lembrando:
% Em regime laminar, f(Fanning) = 16/Re; f(Darcy) = 64/Re;

inx=find(v==0);
v(inx)=1e-6;
%D=max(D,1e-6);
Re = D.*abs(v).*dens./visc;
correlacao = 1;
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
        for i=1:size(Re,1)
            for j=1:size(Re,2)
                f(i,j)=fzero(@(x) 1+2*x.^0.5.*log10(2.51./Re(i,j)./...
                    (x.^0.5)+rugo./3.715./D), 64/Re(i,j));
            end
        end
        f=f./4; % de Darcy para Fanning
        
    case 5 % Moody (1947)
        % Moody L. F., (1947) “An Approximate Formula for Pipe Friction
        % Factors”. Transactions ASME, 69, 1005.
        f=0.001375.*(1+(1E6.*(0.02.*rugo./D+1./Re)).^(1/3));

    case 6
        % Taitel e Dykler (1976) - Laminar
        C = 16; n = 1;
        f = C.*Re.^-n;
        
    case 7
        % Taitel e Dykler (1976) - Turbulento
        C = 0.046; n = 0.2;
        f = C.*Re.^-n;

        
end
f(inx)=0;
Re(inx)=0;

